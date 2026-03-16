export const CLASS_DEFINITIONS = [
  { id: 1, key: "building", label: "Building", color: [255, 84, 68, 224] },
  { id: 2, key: "platform", label: "Platform", color: [92, 210, 120, 224] },
  { id: 3, key: "aguada", label: "Aguada", color: [76, 146, 255, 224] },
];

export const DEFAULTS = {
  tileSize: 480,
  overlap: 0.5,
  threshold: 0.5,
  minBlobSize: 50,
  resolution: 0.5,
  sampleMetaUrl: "./sample/sample-dem.json",
  modelBaseUrl: "./models",
};

export const CONFIDENCE_BUCKETS = [
  { label: "Low", min: 0.0, max: 0.6 },
  { label: "Medium", min: 0.6, max: 0.8 },
  { label: "High", min: 0.8, max: 1.01 },
];
