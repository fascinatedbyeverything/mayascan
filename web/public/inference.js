import { CLASS_DEFINITIONS, DEFAULTS } from "./constants.js";
import { accumulateTile, createAccumulator, extractTile, finalizeAccumulator } from "./tiler.js";

const sessionCache = new Map();
const modelCacheName = "mayascan-models-v1";

function getOrt() {
  const ort = window.ort;
  if (!ort) {
    throw new Error("ONNX Runtime Web failed to load.");
  }

  ort.env.wasm.wasmPaths = "./lib/";
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
  return ort;
}

async function fetchArrayBufferWithCache(url) {
  const cacheApiAvailable = typeof caches !== "undefined";
  if (cacheApiAvailable) {
    try {
      const cache = await caches.open(modelCacheName);
      const cached = await cache.match(url);
      if (cached) {
        return cached.arrayBuffer();
      }
    } catch (error) {
      console.warn("Cache lookup failed; continuing without a cached model.", error);
    }
  }

  const response = await fetch(url, { mode: "cors" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url} (${response.status})`);
  }

  if (cacheApiAvailable) {
    try {
      const cache = await caches.open(modelCacheName);
      await cache.put(url, response.clone());
    } catch (error) {
      console.warn("Cache write failed; continuing without persisting the model.", error);
    }
  }

  return response.arrayBuffer();
}

async function discoverManifest(baseUrl) {
  const normalized = baseUrl.replace(/\/+$/, "");
  const candidates = ["models-manifest.json", "manifest.json"];

  for (const fileName of candidates) {
    try {
      const response = await fetch(`${normalized}/${fileName}`, { mode: "cors" });
      if (!response.ok) {
        continue;
      }
      return await response.json();
    } catch {
      continue;
    }
  }

  return null;
}

function defaultModelPath(baseUrl, classKey, precision) {
  const normalized = baseUrl.replace(/\/+$/, "");
  const suffix = precision === "fp32" ? "" : `_${precision}`;
  return `${normalized}/mayascan_v2_${classKey}_deeplabv3plus_resnet101${suffix}.onnx`;
}

function resolveModelUrl(baseUrl, manifest, classKey, precision) {
  if (!manifest || !manifest.classes || !manifest.classes[classKey]) {
    return defaultModelPath(baseUrl, classKey, precision);
  }

  const entry = manifest.classes[classKey];
  const fileName = entry[precision] || entry.fp16 || entry.fp32 || entry.int8;
  if (!fileName) {
    return defaultModelPath(baseUrl, classKey, precision);
  }
  return `${baseUrl.replace(/\/+$/, "")}/${fileName}`;
}

async function createSession(url, preferredProvider) {
  const cacheKey = `${preferredProvider}:${url}`;
  if (sessionCache.has(cacheKey)) {
    return sessionCache.get(cacheKey);
  }

  const ort = getOrt();
  const modelBytes = await fetchArrayBufferWithCache(url);
  const commonOptions = {
    graphOptimizationLevel: "all",
    enableCpuMemArena: true,
  };

  let session;
  let provider = "wasm";

  if (preferredProvider === "webgpu" && navigator.gpu) {
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        ...commonOptions,
        executionProviders: ["webgpu"],
      });
      provider = "webgpu";
    } catch (error) {
      console.warn("WebGPU session creation failed. Falling back to WASM.", error);
    }
  }

  if (!session) {
    session = await ort.InferenceSession.create(modelBytes, {
      ...commonOptions,
      executionProviders: ["wasm"],
    });
    provider = "wasm";
  }

  const wrapped = { session, provider, url };
  sessionCache.set(cacheKey, wrapped);
  return wrapped;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function transformCoordinates(row, col, size, rotation, flip) {
  let transformedRow = row;
  let transformedCol = col;

  switch (rotation) {
    case 1:
      transformedRow = col;
      transformedCol = size - 1 - row;
      break;
    case 2:
      transformedRow = size - 1 - row;
      transformedCol = size - 1 - col;
      break;
    case 3:
      transformedRow = size - 1 - col;
      transformedCol = row;
      break;
    default:
      break;
  }

  if (flip) {
    transformedCol = size - 1 - transformedCol;
  }

  return { row: transformedRow, col: transformedCol };
}

function inverseTransformCoordinates(row, col, size, rotation, flip) {
  let transformedRow = row;
  let transformedCol = col;

  if (flip) {
    transformedCol = size - 1 - transformedCol;
  }

  switch ((4 - rotation) % 4) {
    case 1:
      return { row: transformedCol, col: size - 1 - transformedRow };
    case 2:
      return { row: size - 1 - transformedRow, col: size - 1 - transformedCol };
    case 3:
      return { row: size - 1 - transformedCol, col: transformedRow };
    default:
      return { row: transformedRow, col: transformedCol };
  }
}

function transformTile(tile, channels, size, rotation, flip) {
  const planeSize = size * size;
  const output = new Float32Array(tile.length);

  for (let channelIndex = 0; channelIndex < channels; channelIndex += 1) {
    const channelOffset = channelIndex * planeSize;
    for (let row = 0; row < size; row += 1) {
      for (let col = 0; col < size; col += 1) {
        const sourceIndex = channelOffset + row * size + col;
        const targetCoords = transformCoordinates(row, col, size, rotation, flip);
        const targetIndex = channelOffset + targetCoords.row * size + targetCoords.col;
        output[targetIndex] = tile[sourceIndex];
      }
    }
  }

  return output;
}

function inverseTransformPrediction(prediction, size, rotation, flip) {
  const output = new Float32Array(prediction.length);
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const sourceIndex = row * size + col;
      const targetCoords = inverseTransformCoordinates(row, col, size, rotation, flip);
      const targetIndex = targetCoords.row * size + targetCoords.col;
      output[targetIndex] = prediction[sourceIndex];
    }
  }
  return output;
}

async function runSingleTile(session, tileData, tileSize) {
  const ort = getOrt();
  const inputName = session.inputNames[0] || "input";
  const outputName = session.outputNames[0] || "output";
  const tensor = new ort.Tensor("float32", tileData, [1, 3, tileSize, tileSize]);
  const results = await session.run({ [inputName]: tensor });
  const output = results[outputName];
  const raw = output.data;
  const probabilities = new Float32Array(tileSize * tileSize);

  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] = sigmoid(raw[index]);
  }

  return probabilities;
}

async function runTileWithTta(session, tileData, tileSize, useTta) {
  if (!useTta) {
    return runSingleTile(session, tileData, tileSize);
  }

  const output = new Float32Array(tileSize * tileSize);
  let count = 0;

  for (let rotation = 0; rotation < 4; rotation += 1) {
    for (const flip of [false, true]) {
      const augmentedTile = transformTile(tileData, 3, tileSize, rotation, flip);
      const prediction = await runSingleTile(session, augmentedTile, tileSize);
      const restored = inverseTransformPrediction(prediction, tileSize, rotation, flip);
      for (let index = 0; index < output.length; index += 1) {
        output[index] += restored[index];
      }
      count += 1;
    }
  }

  for (let index = 0; index < output.length; index += 1) {
    output[index] /= count;
  }

  return output;
}

function choosePrecision(preference) {
  if (preference !== "auto") {
    return preference;
  }
  return navigator.gpu ? "fp16" : "int8";
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function cropChannel(channel, paddedWidth, width, height) {
  const output = new Float32Array(width * height);
  for (let row = 0; row < height; row += 1) {
    const sourceOffset = row * paddedWidth;
    const targetOffset = row * width;
    output.set(channel.subarray(sourceOffset, sourceOffset + width), targetOffset);
  }
  return output;
}

function smooth3x3(channel, width, height) {
  const output = new Float32Array(channel.length);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      let sum = 0;
      let count = 0;
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
          sum += channel[nextRow * width + nextCol];
          count += 1;
        }
      }
      output[row * width + col] = sum / count;
    }
  }
  return output;
}

function buildHeuristicProbabilities(tilePlan) {
  const svf = cropChannel(tilePlan.channels[0], tilePlan.paddedWidth, tilePlan.width, tilePlan.height);
  const openness = cropChannel(
    tilePlan.channels[1],
    tilePlan.paddedWidth,
    tilePlan.width,
    tilePlan.height,
  );
  const slope = cropChannel(tilePlan.channels[2], tilePlan.paddedWidth, tilePlan.width, tilePlan.height);

  const building = new Float32Array(tilePlan.width * tilePlan.height);
  const platform = new Float32Array(tilePlan.width * tilePlan.height);
  const aguada = new Float32Array(tilePlan.width * tilePlan.height);

  for (let index = 0; index < building.length; index += 1) {
    const concavity = 1 - svf[index];
    const sheltered = 1 - openness[index];
    const flatness = 1 - Math.min(1, Math.abs(slope[index] - 0.22) / 0.22);

    building[index] = clamp01(openness[index] * 0.48 + slope[index] * 0.6 + svf[index] * 0.18);
    platform[index] = clamp01(openness[index] * 0.62 + flatness * 0.42 + svf[index] * 0.16);
    aguada[index] = clamp01(concavity * 0.7 + sheltered * 0.42 + (1 - slope[index]) * 0.3);
  }

  return {
    building: smooth3x3(building, tilePlan.width, tilePlan.height),
    platform: smooth3x3(platform, tilePlan.width, tilePlan.height),
    aguada: smooth3x3(aguada, tilePlan.width, tilePlan.height),
  };
}

export async function runInference(tilePlan, options = {}) {
  const modelBaseUrl = options.modelBaseUrl || DEFAULTS.modelBaseUrl;
  const precision = choosePrecision(options.precision || "auto");
  try {
    const manifest = await discoverManifest(modelBaseUrl);
    if (!manifest && modelBaseUrl.replace(/\/+$/, "") === DEFAULTS.modelBaseUrl) {
      return {
        probabilityMaps: buildHeuristicProbabilities(tilePlan),
        provider: "demo",
        precision: "n/a",
        fallbackReason: "No local models-manifest.json found under ./models.",
      };
    }

    const preferredProvider = navigator.gpu ? "webgpu" : "wasm";

    const models = [];
    for (const definition of CLASS_DEFINITIONS) {
      const url = resolveModelUrl(modelBaseUrl, manifest, definition.key, precision);
      const loaded = await createSession(url, preferredProvider);
      models.push({ ...definition, ...loaded });
    }

    const provider = models[0]?.provider || "wasm";
    const totalSteps = tilePlan.tileCount * models.length;
    let completedSteps = 0;

    const probabilityMaps = {};
    for (const [modelIndex, model] of models.entries()) {
      const accumulator = createAccumulator(tilePlan.width, tilePlan.height);

      for (let tileIndex = 0; tileIndex < tilePlan.tileCount; tileIndex += 1) {
        const tile = extractTile(tilePlan, tileIndex);
        const prediction = await runTileWithTta(
          model.session,
          tile.data,
          tilePlan.tileSize,
          options.useTta ?? true,
        );
        accumulateTile(accumulator, prediction, tile.row, tile.col, tilePlan.tileSize);

        completedSteps += 1;
        options.onProgress?.({
          provider,
          precision,
          modelKey: model.key,
          modelLabel: model.label,
          modelIndex: modelIndex + 1,
          modelCount: models.length,
          tileIndex: tileIndex + 1,
          tileCount: tilePlan.tileCount,
          completedSteps,
          totalSteps,
        });
      }

      probabilityMaps[model.key] = finalizeAccumulator(accumulator);
    }

    return {
      probabilityMaps,
      provider,
      precision,
      fallbackReason: null,
    };
  } catch (error) {
    console.warn("Model inference unavailable. Falling back to heuristic terrain scoring.", error);
    options.onProgress?.({
      provider: "demo",
      precision: "n/a",
      modelKey: "demo",
      modelLabel: "Heuristic terrain scorer",
      modelIndex: 1,
      modelCount: 1,
      tileIndex: tilePlan.tileCount,
      tileCount: tilePlan.tileCount,
      completedSteps: 1,
      totalSteps: 1,
    });

    return {
      probabilityMaps: buildHeuristicProbabilities(tilePlan),
      provider: "demo",
      precision: "n/a",
      fallbackReason: error.message,
    };
  }
}
