import { CLASS_DEFINITIONS, DEFAULTS } from "./constants.js";

function makeEmptyMask(length) {
  return new Uint8Array(length);
}

function dilate(mask, width, height) {
  const output = new Uint8Array(mask.length);

  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const index = row * width + col;
      if (!mask[index]) {
        continue;
      }

      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const nextRow = row + rowOffset;
        if (nextRow < 0 || nextRow >= height) {
          continue;
        }
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          const nextCol = col + colOffset;
          if (nextCol < 0 || nextCol >= width) {
            continue;
          }
          output[nextRow * width + nextCol] = 1;
        }
      }
    }
  }

  return output;
}

function erode(mask, width, height) {
  const output = new Uint8Array(mask.length);

  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      let keep = 1;
      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const nextRow = row + rowOffset;
        if (nextRow < 0 || nextRow >= height) {
          keep = 0;
          break;
        }
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          const nextCol = col + colOffset;
          if (nextCol < 0 || nextCol >= width || !mask[nextRow * width + nextCol]) {
            keep = 0;
            break;
          }
        }
        if (!keep) {
          break;
        }
      }
      output[row * width + col] = keep;
    }
  }

  return output;
}

function closing(mask, width, height) {
  return erode(dilate(mask, width, height), width, height);
}

function opening(mask, width, height) {
  return dilate(erode(mask, width, height), width, height);
}

export function createMask(classes, classId) {
  const output = new Uint8Array(classes.length);
  for (let index = 0; index < classes.length; index += 1) {
    output[index] = classes[index] === classId ? 1 : 0;
  }
  return output;
}

export function labelConnectedComponents(mask, width, height, options = {}) {
  const visited = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  const components = [];

  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index] || visited[index]) {
      continue;
    }

    let head = 0;
    let tail = 0;
    let pixelCount = 0;
    let minRow = height;
    let minCol = width;
    let maxRow = 0;
    let maxCol = 0;
    let sumRow = 0;
    let sumCol = 0;
    const pixels = options.collectPixels ? [] : null;

    queue[tail] = index;
    tail += 1;
    visited[index] = 1;

    while (head < tail) {
      const current = queue[head];
      head += 1;
      const row = Math.floor(current / width);
      const col = current - row * width;
      pixelCount += 1;
      minRow = Math.min(minRow, row);
      minCol = Math.min(minCol, col);
      maxRow = Math.max(maxRow, row);
      maxCol = Math.max(maxCol, col);
      sumRow += row;
      sumCol += col;
      pixels?.push(current);

      for (let rowOffset = -1; rowOffset <= 1; rowOffset += 1) {
        const nextRow = row + rowOffset;
        if (nextRow < 0 || nextRow >= height) {
          continue;
        }
        for (let colOffset = -1; colOffset <= 1; colOffset += 1) {
          const nextCol = col + colOffset;
          if (nextCol < 0 || nextCol >= width) {
            continue;
          }
          const nextIndex = nextRow * width + nextCol;
          if (!mask[nextIndex] || visited[nextIndex]) {
            continue;
          }
          visited[nextIndex] = 1;
          queue[tail] = nextIndex;
          tail += 1;
        }
      }
    }

    components.push({
      pixelCount,
      bbox: [minRow, minCol, maxRow, maxCol],
      centroidRow: sumRow / pixelCount,
      centroidCol: sumCol / pixelCount,
      pixels,
    });
  }

  return components;
}

export function postprocessProbabilityMaps(probabilityMaps, width, height, options = {}) {
  const threshold = options.threshold ?? DEFAULTS.threshold;
  const minBlobSize = options.minBlobSize ?? DEFAULTS.minBlobSize;
  const classes = new Uint8Array(width * height);
  const confidence = new Float32Array(width * height);
  const classOrder = CLASS_DEFINITIONS;

  for (let index = 0; index < classes.length; index += 1) {
    let maxForeground = 0;
    let selectedClassId = 0;
    let selectedConfidence = 0;

    for (const definition of classOrder) {
      const probability = probabilityMaps[definition.key][index];
      if (probability > maxForeground) {
        maxForeground = probability;
        selectedClassId = definition.id;
        selectedConfidence = probability;
      }
    }

    const backgroundProbability = 1 - maxForeground;
    if (backgroundProbability >= selectedConfidence || selectedConfidence < threshold) {
      classes[index] = 0;
      confidence[index] = backgroundProbability;
    } else {
      classes[index] = selectedClassId;
      confidence[index] = selectedConfidence;
    }
  }

  for (const definition of classOrder) {
    let mask = createMask(classes, definition.id);
    if (!mask.some(Boolean)) {
      continue;
    }

    mask = closing(mask, width, height);
    mask = opening(mask, width, height);

    const filtered = makeEmptyMask(mask.length);
    const components = labelConnectedComponents(mask, width, height, { collectPixels: false });
    for (const component of components) {
      if (component.pixelCount < minBlobSize) {
        continue;
      }

      const [minRow, minCol, maxRow, maxCol] = component.bbox;
      for (let row = minRow; row <= maxRow; row += 1) {
        for (let col = minCol; col <= maxCol; col += 1) {
          const index = row * width + col;
          if (mask[index]) {
            filtered[index] = 1;
          }
        }
      }
    }

    for (let index = 0; index < classes.length; index += 1) {
      if (classes[index] === definition.id) {
        classes[index] = 0;
      }
      if (filtered[index]) {
        classes[index] = definition.id;
      }
    }
  }

  for (let index = 0; index < classes.length; index += 1) {
    if (classes[index] === 0) {
      let maxForeground = 0;
      for (const definition of classOrder) {
        maxForeground = Math.max(maxForeground, probabilityMaps[definition.key][index]);
      }
      confidence[index] = 1 - maxForeground;
      continue;
    }

    const selectedDefinition = classOrder.find((definition) => definition.id === classes[index]);
    confidence[index] = selectedDefinition
      ? probabilityMaps[selectedDefinition.key][index]
      : 0;
  }

  return {
    classes,
    confidence,
  };
}
