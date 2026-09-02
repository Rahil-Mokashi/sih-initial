import React, { useState } from 'react';
import { CaseRecord, SuspectVessel } from '../types';
import { RotateCcw, ExternalLink, Ship } from 'lucide-react';

interface DriftMapWorkspaceProps {
  currentCase: CaseRecord;
  onSelectVessel: (vessel: SuspectVessel) => void;
  selectedVessel?: SuspectVessel | null;
}

// Renders the REAL Leaflet/folium drift map (src/dashboard/build_map.py's
// map.html / map_{case}.html -- real basemap, real backward-advection
// particle tracks, real candidate vessel markers) via <iframe>, instead of
// the original mock's hand-placed SVG paths and hardcoded vessel names.
// Reimplementing real per-particle Lagrangian tracks as a second,
// less-verified renderer isn't worth it when a real one already exists and
// is exercised by the Python dashboard build.
export const DriftMapWorkspace: React.FC<DriftMapWorkspaceProps> = ({
  currentCase,
  onSelectVessel,
}) => {
  const [iframeKey, setIframeKey] = useState(0);
  const topCandidates = currentCase.rankedCandidates.slice(0, 5);

  return (
    <div className="flex-1 bg-[#181715] rounded-xl border border-[#484644] flex flex-col overflow-hidden shadow-md relative min-h-[520px] lg:min-h-[580px]">
      {/* Map Header bar */}
      <div className="bg-[#1f1e1b] px-4 py-2.5 flex justify-between items-center border-b border-[#484644] shrink-0 z-20">
        <div className="flex items-center gap-3">
          <span className="font-serif font-bold text-sm sm:text-base text-[#faf9f5]">
            Drift Reconstruction — {currentCase.locationName}
          </span>
          <span className="font-mono text-[11px] text-[#77d7ca] bg-[#005049] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-[#5db872] animate-pulse"></span>
            Real ERA5/NCEP backward advection
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs text-[#cac6c2]">
          <a
            href={currentCase.mapUrl}
            target="_blank"
            rel="noopener noreferrer"
            title="Open full map in a new tab"
            className="p-1 rounded hover:bg-[#252320] text-[#faf9f5] transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
          <button
            onClick={() => setIframeKey((k) => k + 1)}
            title="Reload map"
            className="p-1 rounded hover:bg-[#252320] text-[#faf9f5] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Real Leaflet map, iframed from the backend (build_map.py's own HTML
          output -- it already has its own real drift-animation scrubber
          inside it, see src/dashboard/build_map.py's render_drift_animation) */}
      <div className="flex-1 relative bg-[#081320]">
        <iframe
          key={iframeKey}
          src={currentCase.mapUrl}
          title={`Drift map for ${currentCase.code}`}
          className="absolute inset-0 w-full h-full border-0"
        />
      </div>

      {/* Real ranked-vessel quick list (click to open dossier) */}
      <div className="bg-[#1f1e1b] border-t border-[#484644] p-3 sm:p-4 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-mono uppercase tracking-wider text-[#cac6c2]">
            Top candidates on this map
          </span>
          <span className="text-[11px] font-mono text-[#6c6a64]">
            {currentCase.nCandidatesTotal} evaluated total
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {topCandidates.map((v) => (
            <button
              key={v.id}
              onClick={() => onSelectVessel(v)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                v.rank === 1
                  ? 'bg-[#8f482f] text-white font-bold'
                  : 'bg-[#252320] text-[#cac6c2] hover:bg-[#33312d]'
              }`}
              title={`Open ${v.name}'s dossier`}
            >
              <Ship className="w-3 h-3" />
              #{v.rank} {v.name} · {v.matchScore}%
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
