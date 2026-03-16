function reflectIndex(index, length) {
  if (length <= 1) {
    return 0;
  }

  let value = index;
  const limit = length - 1;
  while (value < 0 || value > limit) {
    if (value < 0) {
      value = -value - 1;
    } else {
      value = 2 * length - value - 1;
    }
  }
  return value;
}

function computeStarts(length, tileSize, stride) {
  const positions = [];
  let position = 0;

  while (position + tileSize <= length) {
    positions.push(position);
    position += stride;
  }

  if (positions.length === 0 || positions[positions.length - 1] + tileSize < length) {
    positions.push(Math.max(0, length - tileSize));
  }

  return positions;
}

function padChannel(channel, width, height, paddedWidth, paddedHeight) {
  const output = new Float32Array(paddedWidth * paddedHeight);
  for (let row = 0; row < paddedHeight; row += 1) {
    const sourceRow = reflectIndex(row, height);
    for (let col = 0; col < paddedWidth; col += 1) {
      const sourceCol = reflectIndex(col, width);
      output[row * paddedWidth + col] = channel[sourceRow * width + sourceCol];
    }
  }
  return output;
}

export function createTilePlan(channels, width, height, tileSize = 480, overlap = 0.5) {
  const stride = Math.max(1, Math.round(tileSize * (1 - overlap)));
  const paddedWidth = Math.max(width, tileSize);
  const paddedHeight = Math.max(height, tileSize);
  const rowStarts = computeStarts(paddedHeight, tileSize, stride);
  const colStarts = computeStarts(paddedWidth, tileSize, stride);
  const needHeight = Math.max(height, rowStarts[rowStarts.length - 1] + tileSize);
  const needWidth = Math.max(width, colStarts[colStarts.length - 1] + tileSize);
  const paddedChannels = channels.map((channel) =>
    padChannel(channel, width, height, needWidth, needHeight),
  );

  const origins = [];
  for (const row of rowStarts) {
    for (const col of colStarts) {
      origins.push({ row, col });
    }
  }

  return {
    channels: paddedChannels,
    width,
    height,
    paddedWidth: needWidth,
    paddedHeight: needHeight,
    tileSize,
    overlap,
    origins,
    tileCount: origins.length,
  };
}

export function extractTile(plan, index) {
  const origin = plan.origins[index];
  const output = new Float32Array(plan.channels.length * plan.tileSize * plan.tileSize);
  const planeSize = plan.tileSize * plan.tileSize;

  for (let channelIndex = 0; channelIndex < plan.channels.length; channelIndex += 1) {
    const channel = plan.channels[channelIndex];
    const channelOffset = channelIndex * planeSize;

    for (let row = 0; row < plan.tileSize; row += 1) {
      const sourceOffset = (origin.row + row) * plan.paddedWidth + origin.col;
      const targetOffset = channelOffset + row * plan.tileSize;
      output.set(channel.subarray(sourceOffset, sourceOffset + plan.tileSize), targetOffset);
    }
  }

  return {
    ...origin,
    data: output,
  };
}

export function createAccumulator(width, height) {
  return {
    width,
    height,
    sum: new Float32Array(width * height),
    weight: new Float32Array(width * height),
  };
}

export function accumulateTile(accumulator, tile, row, col, tileSize) {
  const rowEnd = Math.min(row + tileSize, accumulator.height);
  const colEnd = Math.min(col + tileSize, accumulator.width);
  const validRows = rowEnd - row;
  const validCols = colEnd - col;

  for (let localRow = 0; localRow < validRows; localRow += 1) {
    const outputOffset = (row + localRow) * accumulator.width + col;
    const tileOffset = localRow * tileSize;
    for (let localCol = 0; localCol < validCols; localCol += 1) {
      const tileValue = tile[tileOffset + localCol];
      accumulator.sum[outputOffset + localCol] += tileValue;
      accumulator.weight[outputOffset + localCol] += 1;
    }
  }
}

export function finalizeAccumulator(accumulator) {
  const output = new Float32Array(accumulator.sum.length);
  for (let index = 0; index < accumulator.sum.length; index += 1) {
    const weight = accumulator.weight[index] > 0 ? accumulator.weight[index] : 1;
    output[index] = accumulator.sum[index] / weight;
  }
  return output;
}
