import { CLASS_DEFINITIONS, CONFIDENCE_BUCKETS, DEFAULTS } from "./constants.js";
import { createMask, labelConnectedComponents } from "./postprocess.js";

function pixelToMap(row, col, geo) {
  if (geo?.transform) {
    const [a, b, c, d, e, f] = geo.transform;
    return {
      x: c + col * a + row * b,
      y: f + col * d + row * e,
    };
  }

  const resolution = geo?.resolution || DEFAULTS.resolution;
  return {
    x: col * resolution,
    y: row * resolution,
  };
}

function parseUtm(crs) {
  if (!crs) {
    return null;
  }
  const match = /EPSG:(326|327)(\d{2})/.exec(crs);
  if (!match) {
    return null;
  }

  return {
    north: match[1] === "326",
    zone: Number(match[2]),
  };
}

function mercatorToLngLat(x, y) {
  const lng = (x / 20037508.34) * 180;
  let lat = (y / 20037508.34) * 180;
  lat = (180 / Math.PI) * (2 * Math.atan(Math.exp((lat * Math.PI) / 180)) - Math.PI / 2);
  return [lng, lat];
}

function utmToLngLat(x, y, zone, northernHemisphere) {
  const a = 6378137;
  const e = 0.081819191;
  const e1sq = 0.006739497;
  const k0 = 0.9996;

  const xOffset = x - 500000.0;
  let yOffset = y;
  if (!northernHemisphere) {
    yOffset -= 10000000.0;
  }

  const longOrigin = (zone - 1) * 6 - 180 + 3;
  const m = yOffset / k0;
  const mu =
    m /
    (a *
      (1 -
        (e * e) / 4 -
        (3 * e ** 4) / 64 -
        (5 * e ** 6) / 256));

  const e1 = (1 - Math.sqrt(1 - e * e)) / (1 + Math.sqrt(1 - e * e));
  const j1 = (3 * e1) / 2 - (27 * e1 ** 3) / 32;
  const j2 = (21 * e1 ** 2) / 16 - (55 * e1 ** 4) / 32;
  const j3 = (151 * e1 ** 3) / 96;
  const j4 = (1097 * e1 ** 4) / 512;
  const fp =
    mu +
    j1 * Math.sin(2 * mu) +
    j2 * Math.sin(4 * mu) +
    j3 * Math.sin(6 * mu) +
    j4 * Math.sin(8 * mu);

  const sinFp = Math.sin(fp);
  const cosFp = Math.cos(fp);
  const tanFp = Math.tan(fp);
  const c1 = e1sq * cosFp ** 2;
  const t1 = tanFp ** 2;
  const r1 = (a * (1 - e * e)) / Math.pow(1 - e * e * sinFp ** 2, 1.5);
  const n1 = a / Math.sqrt(1 - e * e * sinFp ** 2);
  const d = xOffset / (n1 * k0);

  let lat =
    fp -
    ((n1 * tanFp) / r1) *
      (d ** 2 / 2 -
        ((5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e1sq) * d ** 4) / 24 +
        ((61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e1sq - 3 * c1 ** 2) * d ** 6) /
          720);
  lat = (lat * 180) / Math.PI;

  let lon =
    (d -
      ((1 + 2 * t1 + c1) * d ** 3) / 6 +
      ((5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e1sq + 24 * t1 ** 2) * d ** 5) / 120) /
    cosFp;
  lon = longOrigin + (lon * 180) / Math.PI;

  return [lon, lat];
}

export function mapProject(point, geo) {
  if (!geo?.crs) {
    return null;
  }

  if (geo.crs === "EPSG:4326") {
    return [point.x, point.y];
  }

  if (geo.crs === "EPSG:3857") {
    return mercatorToLngLat(point.x, point.y);
  }

  const utm = parseUtm(geo.crs);
  if (utm) {
    return utmToLngLat(point.x, point.y, utm.zone, utm.north);
  }

  return null;
}

function convexHull(points) {
  if (points.length < 3) {
    return null;
  }

  const sorted = [...points].sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
  const cross = (origin, a, b) =>
    (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]);

  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }

  const upper = [];
  for (let index = sorted.length - 1; index >= 0; index -= 1) {
    const point = sorted[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }

  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function polygonFromComponent(component, geo) {
  const points = component.pixels.map((index) => {
    const row = Math.floor(index / component.width);
    const col = index - row * component.width;
    const point = pixelToMap(row, col, geo);
    return [point.x, point.y];
  });

  const hull = convexHull(points);
  if (!hull) {
    const [minRow, minCol, maxRow, maxCol] = component.bbox;
    const corners = [
      pixelToMap(minRow, minCol, geo),
      pixelToMap(minRow, maxCol + 1, geo),
      pixelToMap(maxRow + 1, maxCol + 1, geo),
      pixelToMap(maxRow + 1, minCol, geo),
    ];
    return corners.map((corner) => [corner.x, corner.y]);
  }

  return hull;
}

export function extractFeatures(classes, confidence, width, height, geo) {
  const features = [];
  const resolution = geo?.resolution || DEFAULTS.resolution;

  for (const definition of CLASS_DEFINITIONS) {
    const mask = createMask(classes, definition.id);
    const components = labelConnectedComponents(mask, width, height, { collectPixels: true });

    components.forEach((component, componentIndex) => {
      if (!component.pixels?.length) {
        return;
      }

      component.width = width;
      const centroidMap = pixelToMap(component.centroidRow, component.centroidCol, geo);
      const centroidLngLat = mapProject(centroidMap, geo);

      let confidenceSum = 0;
      component.pixels.forEach((pixelIndex) => {
        confidenceSum += confidence[pixelIndex];
      });

      const polygonMap = polygonFromComponent(component, geo);
      const polygonClosed = [...polygonMap, polygonMap[0]];
      const polygonLngLat = polygonMap
        .map(([x, y]) => mapProject({ x, y }, geo))
        .filter(Boolean);

      features.push({
        id: `${definition.key}-${componentIndex + 1}`,
        classId: definition.id,
        classKey: definition.key,
        classLabel: definition.label,
        color: definition.color,
        pixelCount: component.pixelCount,
        areaM2: component.pixelCount * resolution * resolution,
        confidence: confidenceSum / component.pixelCount,
        centroid: {
          row: component.centroidRow,
          col: component.centroidCol,
          map: centroidMap,
          lngLat: centroidLngLat,
        },
        bbox: component.bbox,
        polygonMap: polygonClosed,
        polygonLngLat: polygonLngLat.length === polygonMap.length ? [...polygonLngLat, polygonLngLat[0]] : null,
      });
    });
  }

  return features.sort((left, right) => right.areaM2 - left.areaM2);
}

export function summarizeFeatures(features) {
  const summary = {
    totalCount: features.length,
    totalAreaM2: features.reduce((total, feature) => total + feature.areaM2, 0),
    byClass: {},
  };

  for (const definition of CLASS_DEFINITIONS) {
    const classFeatures = features.filter((feature) => feature.classId === definition.id);
    const distribution = CONFIDENCE_BUCKETS.map((bucket) => ({
      label: bucket.label,
      count: classFeatures.filter(
        (feature) => feature.confidence >= bucket.min && feature.confidence < bucket.max,
      ).length,
    }));

    summary.byClass[definition.key] = {
      label: definition.label,
      count: classFeatures.length,
      areaM2: classFeatures.reduce((total, feature) => total + feature.areaM2, 0),
      meanConfidence:
        classFeatures.length > 0
          ? classFeatures.reduce((total, feature) => total + feature.confidence, 0) /
            classFeatures.length
          : 0,
      distribution,
    };
  }

  return summary;
}

export function buildGeoJson(features, geo, projected = false) {
  const items = features
    .map((feature) => {
      const polygon = projected ? feature.polygonLngLat : feature.polygonMap;
      const centroid = projected ? feature.centroid.lngLat : [feature.centroid.map.x, feature.centroid.map.y];
      if (!polygon || !centroid) {
        return null;
      }

      return {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [polygon],
        },
        properties: {
          id: feature.id,
          class: feature.classLabel,
          class_id: feature.classId,
          area_m2: Number(feature.areaM2.toFixed(2)),
          confidence: Number(feature.confidence.toFixed(4)),
          centroid_x: Number(centroid[0].toFixed(6)),
          centroid_y: Number(centroid[1].toFixed(6)),
          pixel_count: feature.pixelCount,
        },
      };
    })
    .filter(Boolean);

  const collection = {
    type: "FeatureCollection",
    features: items,
  };

  if (geo?.crs && !projected) {
    collection.crs = {
      type: "name",
      properties: {
        name: geo.crs,
      },
    };
  }

  return collection;
}

export function buildCsv(features) {
  const rows = [
    [
      "id",
      "class",
      "class_id",
      "area_m2",
      "confidence",
      "pixel_count",
      "centroid_x",
      "centroid_y",
    ],
  ];

  for (const feature of features) {
    rows.push([
      feature.id,
      feature.classLabel,
      String(feature.classId),
      feature.areaM2.toFixed(2),
      feature.confidence.toFixed(4),
      String(feature.pixelCount),
      feature.centroid.map.x.toFixed(6),
      feature.centroid.map.y.toFixed(6),
    ]);
  }

  return rows.map((row) => row.join(",")).join("\n");
}

export function createOverlayCanvas(classes, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const imageData = context.createImageData(width, height);

  for (let index = 0; index < classes.length; index += 1) {
    const definition = CLASS_DEFINITIONS.find((item) => item.id === classes[index]);
    const offset = index * 4;
    if (!definition) {
      imageData.data[offset + 3] = 0;
      continue;
    }
    imageData.data[offset] = definition.color[0];
    imageData.data[offset + 1] = definition.color[1];
    imageData.data[offset + 2] = definition.color[2];
    imageData.data[offset + 3] = definition.color[3];
  }

  context.putImageData(imageData, 0, 0);
  return canvas;
}

export async function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Canvas export failed."));
        return;
      }
      resolve(blob);
    }, "image/png");
  });
}

export function triggerDownload(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}
