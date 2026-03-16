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

function normalize(values) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      continue;
    }
    if (value < min) min = value;
    if (value > max) max = value;
  }

  const output = new Float32Array(values.length);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max - min < 1e-12) {
    return output;
  }

  const scale = 1 / (max - min);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = Math.min(1, Math.max(0, (values[index] - min) * scale));
  }
  return output;
}

function fillInvalid(dem) {
  let sum = 0;
  let count = 0;
  for (let index = 0; index < dem.length; index += 1) {
    const value = dem[index];
    if (Number.isFinite(value)) {
      sum += value;
      count += 1;
    }
  }

  const fallback = count > 0 ? sum / count : 0;
  const output = new Float32Array(dem.length);
  for (let index = 0; index < dem.length; index += 1) {
    const value = dem[index];
    output[index] = Number.isFinite(value) ? value : fallback;
  }
  return output;
}

function boxFilter(dem, width, height, radius) {
  const windowSize = radius * 2 + 1;
  const horizontal = new Float32Array(width * height);

  for (let row = 0; row < height; row += 1) {
    const prefix = new Float64Array(width + 2 * radius + 1);
    for (let extendedCol = 0; extendedCol < width + 2 * radius; extendedCol += 1) {
      const sourceCol = reflectIndex(extendedCol - radius, width);
      prefix[extendedCol + 1] = prefix[extendedCol] + dem[row * width + sourceCol];
    }
    for (let col = 0; col < width; col += 1) {
      const start = col;
      const end = start + windowSize;
      horizontal[row * width + col] = (prefix[end] - prefix[start]) / windowSize;
    }
  }

  const output = new Float32Array(width * height);
  for (let col = 0; col < width; col += 1) {
    const prefix = new Float64Array(height + 2 * radius + 1);
    for (let extendedRow = 0; extendedRow < height + 2 * radius; extendedRow += 1) {
      const sourceRow = reflectIndex(extendedRow - radius, height);
      prefix[extendedRow + 1] = prefix[extendedRow] + horizontal[sourceRow * width + col];
    }
    for (let row = 0; row < height; row += 1) {
      const start = row;
      const end = start + windowSize;
      output[row * width + col] = (prefix[end] - prefix[start]) / windowSize;
    }
  }

  return output;
}

function slidingWindowMax(values, windowSize) {
  const outputLength = values.length - windowSize + 1;
  const output = new Float32Array(outputLength);
  const deque = new Int32Array(values.length);
  let head = 0;
  let tail = 0;

  for (let index = 0; index < values.length; index += 1) {
    while (tail > head && values[deque[tail - 1]] <= values[index]) {
      tail -= 1;
    }
    deque[tail] = index;
    tail += 1;

    if (deque[head] <= index - windowSize) {
      head += 1;
    }

    if (index >= windowSize - 1) {
      output[index - windowSize + 1] = values[deque[head]];
    }
  }

  return output;
}

function maxFilter(dem, width, height, radius) {
  const windowSize = radius * 2 + 1;
  const horizontal = new Float32Array(width * height);

  for (let row = 0; row < height; row += 1) {
    const extended = new Float32Array(width + 2 * radius);
    for (let extendedCol = 0; extendedCol < extended.length; extendedCol += 1) {
      const sourceCol = reflectIndex(extendedCol - radius, width);
      extended[extendedCol] = dem[row * width + sourceCol];
    }
    const maxed = slidingWindowMax(extended, windowSize);
    horizontal.set(maxed, row * width);
  }

  const output = new Float32Array(width * height);
  for (let col = 0; col < width; col += 1) {
    const extended = new Float32Array(height + 2 * radius);
    for (let extendedRow = 0; extendedRow < extended.length; extendedRow += 1) {
      const sourceRow = reflectIndex(extendedRow - radius, height);
      extended[extendedRow] = horizontal[sourceRow * width + col];
    }
    const maxed = slidingWindowMax(extended, windowSize);
    for (let row = 0; row < height; row += 1) {
      output[row * width + col] = maxed[row];
    }
  }

  return output;
}

export function computeSlope(dem, width, height, resolution = 0.5) {
  const output = new Float32Array(width * height);
  const scale = Math.max(resolution, 1e-6);

  for (let row = 0; row < height; row += 1) {
    const up = row > 0 ? row - 1 : row;
    const down = row < height - 1 ? row + 1 : row;

    for (let col = 0; col < width; col += 1) {
      const left = col > 0 ? col - 1 : col;
      const right = col < width - 1 ? col + 1 : col;
      const dx =
        (dem[row * width + right] - dem[row * width + left]) /
        ((right === left ? 1 : 2) * scale);
      const dy =
        (dem[down * width + col] - dem[up * width + col]) /
        ((down === up ? 1 : 2) * scale);
      output[row * width + col] = Math.atan(Math.sqrt(dx * dx + dy * dy)) * (180 / Math.PI);
    }
  }

  return output;
}

export function computeSVF(dem, width, height, resolution = 0.5) {
  const radius = Math.max(3, Math.round(10 * resolution));
  const localMean = boxFilter(dem, width, height, radius);
  const relief = new Float32Array(dem.length);

  for (let index = 0; index < dem.length; index += 1) {
    relief[index] = localMean[index] - dem[index];
  }

  const normalized = normalize(relief);
  const output = new Float32Array(dem.length);
  for (let index = 0; index < dem.length; index += 1) {
    output[index] = 1 - normalized[index];
  }
  return output;
}

export function computeOpenness(dem, width, height, resolution = 0.5) {
  const radius = Math.max(3, Math.round(10 * resolution));
  const localMax = maxFilter(dem, width, height, radius);
  const diff = new Float32Array(dem.length);

  for (let index = 0; index < dem.length; index += 1) {
    diff[index] = localMax[index] - dem[index];
  }

  const normalized = normalize(diff);
  const output = new Float32Array(dem.length);
  for (let index = 0; index < dem.length; index += 1) {
    output[index] = 1 - normalized[index];
  }
  return output;
}

export function computeVisualizations(demInput, width, height, resolution = 0.5) {
  const dem = fillInvalid(demInput);
  const svf = normalize(computeSVF(dem, width, height, resolution));
  const openness = normalize(computeOpenness(dem, width, height, resolution));
  const slope = normalize(computeSlope(dem, width, height, resolution));
  return {
    channels: [svf, openness, slope],
    width,
    height,
  };
}

export function toRgbImageData(channels, width, height) {
  const imageData = new ImageData(width, height);
  const [red, green, blue] = channels;

  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    imageData.data[offset] = Math.round(red[index] * 255);
    imageData.data[offset + 1] = Math.round(green[index] * 255);
    imageData.data[offset + 2] = Math.round(blue[index] * 255);
    imageData.data[offset + 3] = 255;
  }

  return imageData;
}
