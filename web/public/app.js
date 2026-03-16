import { CLASS_DEFINITIONS, DEFAULTS } from "./constants.js";
import { computeVisualizations, toRgbImageData } from "./terrain.js";
import { createTilePlan } from "./tiler.js";
import { runInference } from "./inference.js";
import { postprocessProbabilityMaps } from "./postprocess.js";
import {
  buildCsv,
  buildGeoJson,
  canvasToBlob,
  createOverlayCanvas,
  extractFeatures,
  mapProject,
  summarizeFeatures,
  triggerDownload,
} from "./export.js";

const state = {
  raster: null,
  map: null,
  mapReady: null,
  overlaySourceReady: false,
  featureSourceReady: false,
  latest: null,
};

const elements = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  runButton: document.getElementById("run-button"),
  sampleButton: document.getElementById("sample-button"),
  threshold: document.getElementById("threshold"),
  thresholdValue: document.getElementById("threshold-value"),
  opacity: document.getElementById("opacity"),
  opacityValue: document.getElementById("opacity-value"),
  tta: document.getElementById("tta-toggle"),
  precision: document.getElementById("precision-mode"),
  modelBaseUrl: document.getElementById("model-base-url"),
  fileMeta: document.getElementById("file-meta"),
  progressBar: document.getElementById("progress-bar"),
  progressLabel: document.getElementById("progress-label"),
  gpuStatus: document.getElementById("gpu-status"),
  mapStatus: document.getElementById("map-status"),
  statsPanel: document.getElementById("stats-panel"),
  featureList: document.getElementById("feature-list"),
  terrainCanvas: document.getElementById("terrain-canvas"),
  overlayCanvas: document.getElementById("overlay-canvas"),
  exportGeojson: document.getElementById("export-geojson"),
  exportCsv: document.getElementById("export-csv"),
  exportPng: document.getElementById("export-png"),
};

function setStatus(message) {
  elements.progressLabel.textContent = message;
}

function setProgress(value) {
  elements.progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function formatFileSize(size) {
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function makeSatelliteStyle() {
  return {
    version: 8,
    sources: {
      imagery: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "Esri",
      },
    },
    layers: [
      {
        id: "imagery",
        type: "raster",
        source: "imagery",
      },
    ],
  };
}

function initializeMap() {
  const maplibregl = window.maplibregl;
  if (!maplibregl) {
    console.warn("MapLibre GL JS failed to load.");
    state.mapReady = Promise.resolve();
    elements.mapStatus.textContent = "Map library unavailable in this browser context.";
    return;
  }

  try {
    state.map = new maplibregl.Map({
      container: "map",
      style: makeSatelliteStyle(),
      center: [-89.6, 19.5],
      zoom: 11,
      attributionControl: false,
    });

    state.map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    state.map.addControl(new maplibregl.AttributionControl({ compact: true }));
    state.mapReady = new Promise((resolve) => {
      state.map.on("load", resolve);
    });
  } catch (error) {
    console.warn("Map initialization failed.", error);
    state.map = null;
    state.mapReady = Promise.resolve();
    elements.mapStatus.textContent = "Map unavailable here. Preview canvases and exports still work.";
  }
}

function drawImageData(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const context = canvas.getContext("2d");
  context.putImageData(imageData, 0, 0);
}

function renderOverlayPreview(overlayCanvas) {
  const target = elements.overlayCanvas;
  target.width = overlayCanvas.width;
  target.height = overlayCanvas.height;
  const context = target.getContext("2d");
  context.clearRect(0, 0, target.width, target.height);
  context.drawImage(overlayCanvas, 0, 0);
}

function updateFileMeta(raster) {
  const parts = [
    `<strong>${raster.name}</strong>`,
    `${raster.width} x ${raster.height}px`,
    `${formatFileSize(raster.sizeBytes || raster.width * raster.height * 4)}`,
    `${(raster.geo?.resolution || DEFAULTS.resolution).toFixed(2)} m/px`,
  ];

  if (raster.geo?.crs) {
    parts.push(raster.geo.crs);
  }

  elements.fileMeta.innerHTML = parts.join(" <span class=\"dot\">•</span> ");
}

function ensureExportButtons(enabled) {
  [elements.exportCsv, elements.exportGeojson, elements.exportPng].forEach((button) => {
    button.disabled = !enabled;
  });
}

function renderStats(summary) {
  const blocks = [
    `<div class="stat-card"><span class="label">Total features</span><strong>${summary.totalCount}</strong></div>`,
    `<div class="stat-card"><span class="label">Mapped area</span><strong>${summary.totalAreaM2.toFixed(
      0,
    )} m²</strong></div>`,
  ];

  for (const definition of CLASS_DEFINITIONS) {
    const classSummary = summary.byClass[definition.key];
    blocks.push(`
      <div class="stat-card">
        <span class="label">${classSummary.label}</span>
        <strong>${classSummary.count}</strong>
        <div class="mini">${classSummary.areaM2.toFixed(0)} m²</div>
        <div class="mini">mean conf ${(classSummary.meanConfidence * 100).toFixed(1)}%</div>
        <div class="dist">
          ${classSummary.distribution
            .map((item) => `<span>${item.label}: ${item.count}</span>`)
            .join("")}
        </div>
      </div>
    `);
  }

  elements.statsPanel.innerHTML = blocks.join("");
}

function renderFeatureList(features) {
  if (!features.length) {
    elements.featureList.innerHTML = "<p class=\"empty\">No features survived the confidence and blob filters.</p>";
    return;
  }

  elements.featureList.innerHTML = features
    .slice(0, 80)
    .map((feature) => {
      const centroid = feature.centroid.lngLat
        ? `${feature.centroid.lngLat[1].toFixed(5)}, ${feature.centroid.lngLat[0].toFixed(5)}`
        : `${feature.centroid.map.y.toFixed(2)}, ${feature.centroid.map.x.toFixed(2)}`;
      return `
        <button class="feature-row" data-feature-id="${feature.id}">
          <span class="chip chip-${feature.classKey}">${feature.classLabel}</span>
          <strong>${feature.areaM2.toFixed(1)} m²</strong>
          <span>${(feature.confidence * 100).toFixed(1)}%</span>
          <span>${centroid}</span>
        </button>
      `;
    })
    .join("");

  elements.featureList.querySelectorAll("[data-feature-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const feature = state.latest?.features.find((item) => item.id === button.dataset.featureId);
      if (!feature) {
        return;
      }
      focusFeature(feature);
    });
  });
}

function buildMapCollection(features) {
  const projectedFeatures = buildGeoJson(features, state.raster?.geo, true);
  if (!projectedFeatures.features.length) {
    return null;
  }
  return projectedFeatures;
}

function getImageCoordinates(bounds) {
  return [
    [bounds.west, bounds.north],
    [bounds.east, bounds.north],
    [bounds.east, bounds.south],
    [bounds.west, bounds.south],
  ];
}

async function updateMap(latest) {
  if (!state.map) {
    elements.mapStatus.textContent = "Map unavailable here. Preview canvases and exports still work.";
    return;
  }

  await state.mapReady;
  const collection = buildMapCollection(latest.features);
  if (!collection) {
    elements.mapStatus.textContent =
      "Projection unsupported for basemap overlay. Preview canvases and exports remain available.";
    return;
  }

  const lngs = [];
  const lats = [];
  collection.features.forEach((feature) => {
    feature.geometry.coordinates[0].forEach(([lng, lat]) => {
      lngs.push(lng);
      lats.push(lat);
    });
  });

  const bounds = {
    west: Math.min(...lngs),
    east: Math.max(...lngs),
    south: Math.min(...lats),
    north: Math.max(...lats),
  };

  const map = state.map;
  const overlayUrl = latest.overlayCanvas.toDataURL("image/png");

  if (!state.overlaySourceReady) {
    map.addSource("detections-overlay", {
      type: "image",
      url: overlayUrl,
      coordinates: getImageCoordinates(bounds),
    });
    map.addLayer({
      id: "detections-overlay",
      type: "raster",
      source: "detections-overlay",
      paint: {
        "raster-opacity": Number(elements.opacity.value),
      },
    });
    state.overlaySourceReady = true;
  } else {
    const overlaySource = map.getSource("detections-overlay");
    overlaySource.updateImage({
      url: overlayUrl,
      coordinates: getImageCoordinates(bounds),
    });
  }

  if (!state.featureSourceReady) {
    map.addSource("detection-features", {
      type: "geojson",
      data: collection,
    });

    map.addLayer({
      id: "detection-polygons",
      type: "fill",
      source: "detection-features",
      paint: {
        "fill-color": [
          "match",
          ["get", "class"],
          "Building",
          "#ff5444",
          "Platform",
          "#5cd278",
          "Aguada",
          "#4c92ff",
          "#ffffff",
        ],
        "fill-opacity": 0.16,
      },
    });

    map.addLayer({
      id: "detection-lines",
      type: "line",
      source: "detection-features",
      paint: {
        "line-color": [
          "match",
          ["get", "class"],
          "Building",
          "#ff7967",
          "Platform",
          "#93e6a6",
          "Aguada",
          "#8eb5ff",
          "#ffffff",
        ],
        "line-width": 1.4,
      },
    });

    map.on("click", "detection-polygons", (event) => {
      const feature = event.features?.[0];
      if (!feature) {
        return;
      }

      const coordinates = event.lngLat;
      const popupHtml = `
        <div class="popup">
          <strong>${feature.properties.class}</strong>
          <div>Area: ${feature.properties.area_m2} m²</div>
          <div>Confidence: ${(feature.properties.confidence * 100).toFixed(1)}%</div>
          <div>Centroid: ${feature.properties.centroid_y}, ${feature.properties.centroid_x}</div>
        </div>
      `;
      new window.maplibregl.Popup({ closeButton: false })
        .setLngLat([coordinates.lng, coordinates.lat])
        .setHTML(popupHtml)
        .addTo(map);
    });

    state.featureSourceReady = true;
  } else {
    map.getSource("detection-features").setData(collection);
  }

  map.fitBounds(
    [
      [bounds.west, bounds.south],
      [bounds.east, bounds.north],
    ],
    { padding: 48, duration: 800 },
  );

  elements.mapStatus.textContent = "Detections projected onto the basemap.";
}

function focusFeature(feature) {
  if (!feature.centroid.lngLat || !state.map) {
    return;
  }

  state.map.flyTo({
    center: feature.centroid.lngLat,
    zoom: Math.max(state.map.getZoom(), 15),
    essential: true,
  });
}

function updateOpacity() {
  elements.opacityValue.textContent = Number(elements.opacity.value).toFixed(2);
  if (state.map?.getLayer("detections-overlay")) {
    state.map.setPaintProperty(
      "detections-overlay",
      "raster-opacity",
      Number(elements.opacity.value),
    );
  }
}

function updateThresholdLabel() {
  elements.thresholdValue.textContent = Number(elements.threshold.value).toFixed(2);
}

function parseCrs(geoKeys) {
  if (geoKeys?.ProjectedCSTypeGeoKey) {
    return `EPSG:${geoKeys.ProjectedCSTypeGeoKey}`;
  }
  if (geoKeys?.GeographicTypeGeoKey) {
    return `EPSG:${geoKeys.GeographicTypeGeoKey}`;
  }
  return null;
}

function buildGeoMetadata(image) {
  const fileDirectory = image.getFileDirectory?.() || {};
  const geoKeys = image.getGeoKeys?.() || {};
  const tiepoint = fileDirectory.ModelTiepoint || fileDirectory.ModelTiepointTag;
  const scale = fileDirectory.ModelPixelScale || fileDirectory.ModelPixelScaleTag;
  const width = image.getWidth();
  const height = image.getHeight();
  let transform = null;

  if (tiepoint?.length >= 6 && scale?.length >= 2) {
    const [pixelX, pixelY, , mapX, mapY] = tiepoint;
    transform = [
      scale[0],
      0,
      mapX - pixelX * scale[0],
      0,
      -scale[1],
      mapY + pixelY * scale[1],
    ];
  }

  let bounds = null;
  if (transform) {
    const west = transform[2];
    const north = transform[5];
    const east = west + transform[0] * width;
    const south = north + transform[4] * height;
    bounds = [west, south, east, north];
  }

  return {
    crs: parseCrs(geoKeys),
    transform,
    bounds,
    resolution: Math.abs(scale?.[0] || DEFAULTS.resolution),
  };
}

async function readGeoTiffFromArrayBuffer(arrayBuffer, name, sizeBytes) {
  const GeoTIFF = window.GeoTIFF;
  if (!GeoTIFF?.fromArrayBuffer) {
    throw new Error("geotiff.js failed to load.");
  }

  const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage();
  const rasters = await image.readRasters({ interleave: false });
  const width = image.getWidth();
  const height = image.getHeight();
  const geo = buildGeoMetadata(image);

  const samples = Array.isArray(rasters)
    ? rasters
    : Array.from({ length: rasters.length || 0 }, (_, index) => rasters[index]);
  const channels = samples.map((sample) => Float32Array.from(sample));
  const singleBand = channels.length === 1;
  const precomputedVisualization = channels.length >= 3;

  return {
    name,
    sizeBytes,
    width,
    height,
    geo,
    dem: singleBand ? channels[0] : null,
    channels: precomputedVisualization ? channels.slice(0, 3) : null,
  };
}

async function loadRasterFromFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  return readGeoTiffFromArrayBuffer(arrayBuffer, file.name, file.size);
}

async function loadSampleRaster() {
  const metaResponse = await fetch(DEFAULTS.sampleMetaUrl);
  if (!metaResponse.ok) {
    throw new Error("Sample metadata missing.");
  }

  const meta = await metaResponse.json();
  const dataResponse = await fetch(meta.dataUrl);
  if (!dataResponse.ok) {
    throw new Error("Sample DEM missing.");
  }

  const buffer = await dataResponse.arrayBuffer();
  const dem = new Float32Array(buffer);
  return {
    name: meta.name,
    sizeBytes: buffer.byteLength,
    width: meta.width,
    height: meta.height,
    geo: {
      crs: meta.crs,
      transform: meta.transform,
      bounds: meta.bounds,
      resolution: meta.resolution,
    },
    dem,
    channels: null,
  };
}

async function prepareRaster(raster) {
  updateFileMeta(raster);
  state.raster = raster;
  elements.runButton.disabled = false;

  if (raster.channels) {
    const imageData = toRgbImageData(raster.channels, raster.width, raster.height);
    drawImageData(elements.terrainCanvas, imageData);
    return;
  }

  setStatus("Preparing terrain visualization preview...");
  const visualization = computeVisualizations(
    raster.dem,
    raster.width,
    raster.height,
    raster.geo?.resolution || DEFAULTS.resolution,
  );
  const imageData = toRgbImageData(visualization.channels, raster.width, raster.height);
  drawImageData(elements.terrainCanvas, imageData);
}

async function runPipeline() {
  if (!state.raster) {
    return;
  }

  ensureExportButtons(false);
  setProgress(4);
  setStatus("Computing terrain visualization...");

  const visualization = state.raster.channels
    ? { channels: state.raster.channels }
    : computeVisualizations(
        state.raster.dem,
        state.raster.width,
        state.raster.height,
        state.raster.geo?.resolution || DEFAULTS.resolution,
      );

  const imageData = toRgbImageData(
    visualization.channels,
    state.raster.width,
    state.raster.height,
  );
  drawImageData(elements.terrainCanvas, imageData);

  setStatus("Generating overlapping tiles...");
  setProgress(12);
  const tilePlan = createTilePlan(
    visualization.channels,
    state.raster.width,
    state.raster.height,
    DEFAULTS.tileSize,
    DEFAULTS.overlap,
  );

  setStatus("Loading ONNX models...");
  const inferenceResult = await runInference(tilePlan, {
    modelBaseUrl: elements.modelBaseUrl.value.trim() || DEFAULTS.modelBaseUrl,
    precision: elements.precision.value,
    useTta: elements.tta.checked,
    onProgress: (progress) => {
      const percent = 12 + (progress.completedSteps / progress.totalSteps) * 70;
      setProgress(percent);
      setStatus(
        `Running ${progress.modelLabel} model (${progress.modelIndex}/${progress.modelCount}) - tile ${progress.tileIndex}/${progress.tileCount}`,
      );
      if (progress.provider === "webgpu") {
        elements.gpuStatus.textContent = `WebGPU active (${progress.precision.toUpperCase()})`;
      } else if (progress.provider === "wasm") {
        elements.gpuStatus.textContent = `WASM fallback (${progress.precision.toUpperCase()})`;
      } else {
        elements.gpuStatus.textContent = "Demo fallback (models unavailable)";
      }
    },
  });

  setStatus("Post-processing detections...");
  setProgress(86);
  const postprocessed = postprocessProbabilityMaps(
    inferenceResult.probabilityMaps,
    state.raster.width,
    state.raster.height,
    {
      threshold: Number(elements.threshold.value),
      minBlobSize: DEFAULTS.minBlobSize,
    },
  );

  const overlayCanvas = createOverlayCanvas(
    postprocessed.classes,
    state.raster.width,
    state.raster.height,
  );
  renderOverlayPreview(overlayCanvas);

  const features = extractFeatures(
    postprocessed.classes,
    postprocessed.confidence,
    state.raster.width,
    state.raster.height,
    state.raster.geo,
  );
  const summary = summarizeFeatures(features);

  state.latest = {
    ...postprocessed,
    visualization,
    overlayCanvas,
    features,
    geoJson: buildGeoJson(features, state.raster.geo),
  };

  renderStats(summary);
  renderFeatureList(features);
  await updateMap(state.latest);

  setProgress(100);
  setStatus(
    inferenceResult.fallbackReason
      ? `Completed ${features.length} feature detections using heuristic fallback.`
      : `Completed ${features.length} feature detections.`,
  );
  ensureExportButtons(true);
}

async function handleFileSelection(file) {
  const raster = await loadRasterFromFile(file);
  await prepareRaster(raster);
}

function bindExportHandlers() {
  elements.exportGeojson.addEventListener("click", () => {
    if (!state.latest) {
      return;
    }
    const blob = new Blob([JSON.stringify(state.latest.geoJson, null, 2)], {
      type: "application/geo+json",
    });
    triggerDownload(blob, "mayascan-detections.geojson");
  });

  elements.exportCsv.addEventListener("click", () => {
    if (!state.latest) {
      return;
    }
    const blob = new Blob([buildCsv(state.latest.features)], { type: "text/csv" });
    triggerDownload(blob, "mayascan-detections.csv");
  });

  elements.exportPng.addEventListener("click", async () => {
    if (!state.latest) {
      return;
    }
    const blob = await canvasToBlob(state.latest.overlayCanvas);
    triggerDownload(blob, "mayascan-overlay.png");
  });
}

function bindEvents() {
  elements.threshold.addEventListener("input", updateThresholdLabel);
  elements.opacity.addEventListener("input", updateOpacity);
  elements.runButton.addEventListener("click", () => {
    runPipeline().catch((error) => {
      console.error(error);
      setStatus(error.message);
      setProgress(0);
    });
  });

  elements.sampleButton.addEventListener("click", async () => {
    try {
      const raster = await loadSampleRaster();
      await prepareRaster(raster);
      await runPipeline();
    } catch (error) {
      console.error(error);
      setStatus(error.message);
    }
  });

  elements.fileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    try {
      await handleFileSelection(file);
    } catch (error) {
      console.error(error);
      setStatus(error.message);
    }
  });

  elements.dropzone.addEventListener("click", () => elements.fileInput.click());
  elements.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("dragging");
  });
  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("dragging");
  });
  elements.dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("dragging");
    const [file] = event.dataTransfer.files;
    if (!file) {
      return;
    }
    try {
      await handleFileSelection(file);
    } catch (error) {
      console.error(error);
      setStatus(error.message);
    }
  });
}

function seedDefaults() {
  elements.modelBaseUrl.value = DEFAULTS.modelBaseUrl;
  updateThresholdLabel();
  updateOpacity();
  ensureExportButtons(false);
  elements.gpuStatus.textContent = navigator.gpu ? "WebGPU available" : "WebGPU unavailable";
  elements.mapStatus.textContent = "Upload a DEM or try the bundled sample to populate the map.";
  setStatus("Ready for GeoTIFF upload.");
}

async function maybeBootSample() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("sample") !== "1") {
    return;
  }

  try {
    const raster = await loadSampleRaster();
    await prepareRaster(raster);
    await runPipeline();
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  }
}

initializeMap();
seedDefaults();
bindEvents();
bindExportHandlers();
maybeBootSample();
