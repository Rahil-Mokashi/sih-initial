export interface SuspectVessel {
  id: string;
  rank: number;
  name: string;
  mmsi: string;
  imo: string;
  flag: string;
  countryCode: string;
  vesselType: string;
  matchScore: number;
  rawScore?: number;
  distFromOriginKm: number;
  timeGapHours: number;
  aisStatus: string;
  aisGapIntentional?: boolean;
  aisGapHours?: number;
  unaccountedMovementKm?: number;
  closestApproachKm: number;
  gfwPresenceRecords: number;
  // Not returned by the GFW presence API this pipeline actually calls --
  // left optional rather than fabricated. See VesselDossierModal's
  // "Vessel Identity" tab, which only renders fields that are present.
  speedKnots?: number;
  draftMeters?: number;
  lengthMeters?: number;
  destination?: string;
  owner?: string;
  evidenceDate?: string;
  entryTimestamp?: string;
  exitTimestamp?: string;
  lastSeenTime: string;
  evidenceTags: string[];
  behaviorSummary: string;
  coordinates: [number, number]; // [lat, lng] -- GFW's grid-cell centroid at the requested spatial resolution, not an exact GPS fix
  historicalPath: { time: string; lat: number; lng: number; isDark?: boolean; speed?: number }[];
  // Real pipeline (src/attribution/score_vessels.py) does not compute
  // per-sensor corroboration percentages -- omitted rather than invented.
  // Kept optional so any leftover mock fixture still type-checks.
  sensorCorroboration?: {
    sarWakeAlignment: number;
    windDataCorrelation: number;
    trajectoryConsistency: number;
    approachConfidence: number;
  };
}

export interface GeometricCharacterization {
  areaPx: number;
  lengthPx: number;
  widthPx: number;
  orientationDeg: number;
  elongation: number;
  connectedComponents: number;
  groundTruthAreaPx?: number;
  groundTruthDimensions?: string;
  groundTruthOrientation?: string;
  groundTruthElongation?: string;
}

export interface EnvironmentalData {
  driftBackDistanceKm: number;
  vesselsEvaluated: number;
  topMatchScore: number;
  // Only populated when both ERA5 and NCEP/NCAR wind sources are present
  // for this case (src/dashboard/build_dashboard.py's own wind_compare_stat
  // has the same "only if both sources exist" condition).
  environmentalDisagreementKm?: number;
  varianceNote?: string;
  // Real per-source wind speed/heading isn't surfaced by
  // src/drift/wind_era5.py / wind_ncep.py in a form the dashboard reads
  // today (only per-source origin centroids) -- left optional rather than
  // fabricated.
  oceanCurrentSpeedMs?: number;
  oceanCurrentHeadingDeg?: number;
  windSpeedMs?: number;
  windHeadingDeg?: number;
  era5OriginCoords?: [number, number];
  ncepOriginCoords?: [number, number];
  consensusOriginCoords: [number, number];
  detectionCoords: [number, number];
}

export interface DriftParticle {
  id: number;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  timeOffset: number; // -72 to 0
  weight: number;
  model: 'ERA5' | 'HYCOM' | 'NCEP';
}

export interface CaseRecord {
  id: string;
  code: string;
  name: string;
  locationName: string;
  region: string;
  detectionTime: string;
  originEstimatedTime: string;
  status: 'In Progress' | 'Confirmed' | 'Review Required';
  detectionStatus: 'Confirmed' | 'In Progress' | 'Validating';
  driftStatus: 'Confirmed' | 'Simulating' | 'Pending';
  attributionStatus: 'In Progress' | 'Ranked' | 'Verified';
  summary: string;
  satelliteSensor: string;
  sarTileUrl: string;
  sarMaskUrl: string;
  mapBgUrl: string;
  // Real Leaflet/folium drift map for this case (src/dashboard/build_map.py's
  // map.html / map_{case}.html), served by the backend for an <iframe>.
  mapUrl: string;
  coordinates: {
    lat: number;
    lng: number;
  };
  environmental: EnvironmentalData;
  // Real per-case slick geometry isn't produced by this pipeline yet --
  // data/processed/dashboard/detection_geometry.json is a single global
  // demo tile's ground-truth/prediction comparison, not specific to any
  // drift case. See DetectionDemoResult below for that real (but
  // case-independent) data instead. Kept optional so a future per-case
  // detection pipeline can populate it without a type change.
  geometry?: GeometricCharacterization;
  topSuspect: SuspectVessel;
  rankedCandidates: SuspectVessel[];
  nCandidatesTotal: number;
  provenance: {
    trainingDataset: string;
    trainingTiles: string;
    detectionModel: string;
    detectionValDice: string;
    driftModels: string;
    driftLastComputed: string;
    gfwApiQuota: string;
    gfwRequestsUsed: string;
  };
  evidenceTimeline: {
    time: string;
    source: string;
    event: string;
    type: 'sar' | 'ais' | 'drift' | 'weather';
    confidence: 'high' | 'medium' | 'critical';
  }[];
}

export type ActiveTab = 'detection' | 'drift' | 'attribution' | 'evidence';

// The one real (but case-independent) detection example the pipeline has
// pre-rendered: data/processed/dashboard/{detection_overlay.png,
// detection_geometry.json}, produced by scripts/render_detection_overlay.py
// against the real trained checkpoint on a held-out Zenodo Part III tile.
export interface DetectionDemoResult {
  demoImage: string;
  overlayUrl: string;
  groundTruth: GeometricCharacterization | null;
  prediction: GeometricCharacterization | null;
  checkpoint: string | null;
}

// Response shape for POST /api/detect (the upload feature): a real
// forward pass of an uploaded SAR tile through the trained checkpoint.
export interface DetectionUploadResult {
  overlayPngBase64: string;
  geometry: GeometricCharacterization;
  checkpoint: string;
  inputFormat: 'geotiff_dB' | 'generic_image_best_effort';
  resizedTo: [number, number];
  originalSize: [number, number];
  note: string;
}
