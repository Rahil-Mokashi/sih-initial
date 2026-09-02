import React, { useEffect, useRef, useState } from 'react';
import { CaseRecord, DetectionDemoResult, DetectionUploadResult, GeometricCharacterization } from '../types';
import { fetchDetectionDemo, uploadForDetection } from '../lib/api';
import { Satellite, UploadCloud, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

interface SarDetectionModuleProps {
  currentCase: CaseRecord;
}

function GeometryCells({ label, g }: { label: string; g: GeometricCharacterization | null | undefined }) {
  if (!g || !g.areaPx) {
    return (
      <div className="bg-[#f4f4f0] p-3 rounded border border-[#e6dfd8] border-dashed text-xs text-[#6c6a64] italic">
        {label}: no detected pixels
      </div>
    );
  }
  return (
    <div className="bg-[#f4f4f0] p-3 rounded border border-[#e6dfd8] space-y-1.5 font-mono text-xs">
      <div className="text-[10px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">{label}</div>
      <div className="flex justify-between"><span className="text-[#6c6a64]">Area</span><span className="font-bold text-[#141413]">{g.areaPx.toLocaleString()} px ({g.connectedComponents} comp.)</span></div>
      <div className="flex justify-between"><span className="text-[#6c6a64]">Length × Width</span><span className="font-bold text-[#141413]">{g.lengthPx?.toFixed(0)} × {g.widthPx?.toFixed(0)} px</span></div>
      <div className="flex justify-between"><span className="text-[#6c6a64]">Orientation</span><span className="font-bold text-[#141413]">{g.orientationDeg?.toFixed(0)}°</span></div>
      <div className="flex justify-between"><span className="text-[#6c6a64]">Elongation</span><span className="font-bold text-[#141413]">{g.elongation?.toFixed(2)}x</span></div>
    </div>
  );
}

export const SarDetectionModule: React.FC<SarDetectionModuleProps> = ({ currentCase }) => {
  const [demo, setDemo] = useState<DetectionDemoResult | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);

  const [uploadResult, setUploadResult] = useState<DetectionUploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchDetectionDemo()
      .then(setDemo)
      .catch((err) => setDemoError(err.message ?? String(err)));
  }, []);

  const handleFile = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const result = await uploadForDetection(file);
      setUploadResult(result);
    } catch (err: any) {
      setUploadError(err.message ?? String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="border-b border-[#e6dfd8] pb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-serif text-xl sm:text-2xl font-bold text-[#141413]">
            Detection & Geometric Characterization
          </h3>
          <p className="text-xs text-[#6c6a64] mt-0.5">
            Real trained U-Net/ResNet18 checkpoint · upload your own SAR tile or review the pre-rendered demo below
          </p>
        </div>
        <span className="text-xs font-mono bg-[#efeeea] px-2.5 py-1 rounded text-[#141413] border border-[#e6dfd8]">
          Sensor: {currentCase.satelliteSensor}
        </span>
      </div>

      {/* Upload Feature */}
      <div className="bg-[#1f1e1b] rounded-xl p-4 sm:p-5 border border-[#484644] shadow-md">
        <div className="flex items-center gap-2 mb-3">
          <UploadCloud className="w-4 h-4 text-[#ffb59d]" />
          <h4 className="font-sans font-bold text-sm text-[#faf9f5]">Upload a SAR Tile for Real Detection</h4>
        </div>
        <p className="text-xs text-[#cac6c2] mb-3">
          Runs an actual forward pass through the trained checkpoint. Best results with a calibrated Sigma0-dB
          GeoTIFF (the same format the training data ships as) -- a plain PNG/JPG is accepted too, but rescaled
          onto the model's expected range as a rough best-effort approximation only.
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
            e.target.value = '';
          }}
        />
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 bg-[#8f482f] hover:bg-[#a9583e] disabled:opacity-50 text-white px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all"
          >
            {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UploadCloud className="w-4 h-4" />}
            <span>{uploading ? 'Running inference…' : 'Choose SAR tile…'}</span>
          </button>
          {uploadResult && (
            <span className="flex items-center gap-1.5 text-xs text-[#5db872] font-mono">
              <CheckCircle2 className="w-3.5 h-3.5" /> Detection complete (checkpoint: {uploadResult.checkpoint})
            </span>
          )}
          {uploadError && (
            <span className="flex items-center gap-1.5 text-xs text-[#ffb59d] font-mono">
              <AlertCircle className="w-3.5 h-3.5" /> {uploadError}
            </span>
          )}
        </div>

        {uploadResult && (
          <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-lg overflow-hidden border border-[#54433e]">
              <img
                src={`data:image/png;base64,${uploadResult.overlayPngBase64}`}
                alt="Uploaded tile with predicted oil mask overlay"
                className="w-full block"
              />
            </div>
            <div className="space-y-2">
              <GeometryCells label="Model Prediction" g={uploadResult.geometry} />
              <p className="text-[11px] text-[#cac6c2] leading-relaxed">
                {uploadResult.note} Original size {uploadResult.originalSize[0]}×{uploadResult.originalSize[1]}px,
                resized to {uploadResult.resizedTo[0]}×{uploadResult.resizedTo[1]}px for inference.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Pre-rendered real demo (scripts/render_detection_overlay.py) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#1f1e1b] rounded-xl p-4 sm:p-5 border border-[#484644] flex flex-col shadow-md">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <Satellite className="w-4 h-4 text-[#ffb59d]" />
              <h4 className="font-sans font-bold text-sm text-[#faf9f5]">Pre-rendered Demo (Held-out Test Tile)</h4>
            </div>
            {demo?.checkpoint && (
              <span className="bg-[#e8a55a] text-[#141413] font-mono text-[10px] uppercase font-bold px-2 py-0.5 rounded">
                {demo.checkpoint}
              </span>
            )}
          </div>

          {demoError && <p className="text-xs text-[#ffb59d]">{demoError}</p>}
          {demo?.overlayUrl ? (
            <img src={demo.overlayUrl} alt="SAR tile, ground truth, and predicted mask" className="w-full rounded-lg" />
          ) : (
            !demoError && <p className="text-xs text-[#cac6c2]">Loading demo…</p>
          )}
          <p className="text-[11px] text-[#cac6c2] mt-3">
            Real Zenodo Part III (held-out test set) tile, never seen during training -- see
            scripts/render_detection_overlay.py. Not specific to case {currentCase.code}.
          </p>
        </div>

        <div className="flex flex-col gap-3">
          <GeometryCells label="Real Ground Truth (Zenodo mask)" g={demo?.groundTruth} />
          <GeometryCells label="Model Prediction" g={demo?.prediction} />
          <p className="text-[11px] text-[#6c6a64] italic">
            Pixel units only -- this dataset's Sentinel-1 product type/ground resolution isn't documented by its
            source, so no km² conversion is assumed.
          </p>
        </div>
      </div>
    </div>
  );
};
