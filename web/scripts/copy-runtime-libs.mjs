import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const publicLibDir = path.join(rootDir, "public", "lib");

mkdirSync(publicLibDir, { recursive: true });

function firstExisting(candidates) {
  for (const candidate of candidates) {
    const fullPath = path.join(rootDir, candidate);
    if (existsSync(fullPath)) {
      return fullPath;
    }
  }
  return null;
}

function copyFile(label, candidates, destinationName) {
  const source = firstExisting(candidates);
  if (!source) {
    throw new Error(`Could not find ${label}. Tried: ${candidates.join(", ")}`);
  }

  const destination = path.join(publicLibDir, destinationName);
  cpSync(source, destination, { force: true });
  console.log(`Copied ${label}: ${path.relative(rootDir, source)} -> public/lib/${destinationName}`);
}

copyFile(
  "ONNX Runtime Web bundle",
  ["node_modules/onnxruntime-web/dist/ort.min.js"],
  "ort.min.js",
);

copyFile(
  "GeoTIFF browser bundle",
  [
    "node_modules/geotiff/dist-browser/geotiff.js",
    "node_modules/geotiff/dist-browser/geotiff.min.js",
    "node_modules/geotiff/dist/geotiff.js",
  ],
  "geotiff.min.js",
);

copyFile(
  "MapLibre GL JS bundle",
  ["node_modules/maplibre-gl/dist/maplibre-gl.js"],
  "maplibre-gl.js",
);

copyFile(
  "MapLibre GL stylesheet",
  ["node_modules/maplibre-gl/dist/maplibre-gl.css"],
  "maplibre-gl.css",
);

const ortDistDir = path.join(rootDir, "node_modules", "onnxruntime-web", "dist");
if (!existsSync(ortDistDir)) {
  throw new Error("Missing onnxruntime-web dist directory.");
}

for (const fileName of readdirSync(publicLibDir)) {
  if (fileName.startsWith("ort-wasm") && fileName.endsWith(".wasm")) {
    rmSync(path.join(publicLibDir, fileName), { force: true });
  }
}

for (const fileName of readdirSync(ortDistDir)) {
  if (
    fileName === "ort-wasm-simd-threaded.wasm" ||
    fileName === "ort-wasm-simd-threaded.jsep.wasm" ||
    fileName === "ort-wasm-simd-threaded.jspi.wasm"
  ) {
    cpSync(path.join(ortDistDir, fileName), path.join(publicLibDir, fileName), { force: true });
    console.log(`Copied ORT runtime: ${fileName}`);
  }
}
