import React from 'react';
import { CaseRecord, SuspectVessel } from '../types';
import { 
  FolderGit2, 
  Satellite, 
  Waves, 
  Wind, 
  Radio, 
  ShieldAlert, 
  FileCheck2, 
  Clock, 
  Calendar,
  ExternalLink,
  ChevronRight
} from 'lucide-react';

interface EvidenceHubViewProps {
  currentCase: CaseRecord;
  onOpenDossier: (vessel: SuspectVessel) => void;
}

export const EvidenceHubView: React.FC<EvidenceHubViewProps> = ({
  currentCase,
  onOpenDossier,
}) => {
  const { topSuspect, evidenceTimeline } = currentCase;

  return (
    <div className="space-y-6">
      {/* Evidence Hub Header */}
      <div className="border-b border-[#e6dfd8] pb-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase px-2 py-0.5 rounded bg-[#8f482f] text-white font-bold">
              CLASSIFIED EVIDENCE REPOSITORY
            </span>
            <span className="text-xs text-[#6c6a64]">NTRO Forensic Case #{currentCase.code}</span>
          </div>
          <h2 className="font-serif text-2xl sm:text-3xl font-bold text-[#141413]">
            Multi-Sensor Evidence Hub
          </h2>
          <p className="text-xs sm:text-sm text-[#6c6a64] mt-1 max-w-3xl">
            Real evidence from three independent sources: Sentinel-1 SAR satellite detection, ERA5/NCEP-NCAR backward drift reconstruction, and Global Fishing Watch AIS vessel telemetry.
          </p>
        </div>

        <button
          onClick={() => onOpenDossier(topSuspect)}
          className="flex items-center gap-2 bg-[#8f482f] hover:bg-[#a9583e] text-white px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all shadow-xs"
        >
          <ShieldAlert className="w-4 h-4" />
          <span>Inspect Top Suspect ({topSuspect.name})</span>
        </button>
      </div>

      {/* 3 Core Intelligence Pillars -- real per-case numbers */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Pillar 1: Satellite Intelligence */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 sm:p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 text-[#8f482f] mb-2 font-bold text-sm">
              <Satellite className="w-5 h-5" />
              <span>Layer 1 — Satellite Detection</span>
            </div>
            <p className="text-xs text-[#3d3d3a] leading-relaxed">
              {currentCase.satelliteSensor} detection recorded at {currentCase.detectionTime}, at{' '}
              {currentCase.coordinates.lat.toFixed(3)}, {currentCase.coordinates.lng.toFixed(3)}.
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-[#e6dfd8] flex justify-between text-[11px] font-mono text-[#6c6a64]">
            <span>Status: {currentCase.detectionStatus}</span>
            <span>Sensor: Sentinel-1</span>
          </div>
        </div>

        {/* Pillar 2: Oceanographic Hindcast */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 sm:p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 text-[#0f8378] mb-2 font-bold text-sm">
              <Waves className="w-5 h-5" />
              <span>Layer 2 — Oceanographic Hindcast</span>
            </div>
            <p className="text-xs text-[#3d3d3a] leading-relaxed">
              Backward drift simulation traces {currentCase.environmental.driftBackDistanceKm} km back to the
              estimated origin ({currentCase.originEstimatedTime}).
              {currentCase.environmental.environmentalDisagreementKm != null &&
                ` ERA5 and NCEP/NCAR agree within ${currentCase.environmental.environmentalDisagreementKm} km.`}
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-[#e6dfd8] flex justify-between text-[11px] font-mono text-[#6c6a64]">
            <span>Status: {currentCase.driftStatus}</span>
            <span>Sources: ERA5 / NCEP</span>
          </div>
        </div>

        {/* Pillar 3: Maritime Dark-Vessel Attribution */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 sm:p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2.5 text-[#d4a017] mb-2 font-bold text-sm">
              <Radio className="w-5 h-5" />
              <span>Layer 3 — Maritime AIS Intelligence</span>
            </div>
            <p className="text-xs text-[#3d3d3a] leading-relaxed">
              Evaluated {currentCase.nCandidatesTotal} vessels across the search radius.
              {topSuspect.aisGapIntentional
                ? ` Identified a ${topSuspect.aisGapHours?.toFixed(0)}-hour likely-intentional AIS gap on ${topSuspect.name}.`
                : ` Top match: ${topSuspect.name}.`}
            </p>
          </div>
          <div className="mt-4 pt-3 border-t border-[#e6dfd8] flex justify-between text-[11px] font-mono text-[#6c6a64]">
            <span>Top Match: {topSuspect.matchScore}%</span>
            <span>Source: GFW API</span>
          </div>
        </div>
      </div>

      {/* Forensic Intelligence Timeline */}
      <div className="bg-[#faf9f5] border border-[#e6dfd8] rounded-xl p-5 sm:p-6 shadow-xs">
        <div className="flex items-center justify-between mb-5 border-b border-[#e6dfd8] pb-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-[#8f482f]" />
            <h3 className="font-serif text-lg font-bold text-[#141413]">
              Chronological Chain of Custody & Incident Timeline
            </h3>
          </div>
          <span className="text-xs font-mono text-[#6c6a64]">
            {evidenceTimeline.length} Verified Events
          </span>
        </div>

        <div className="relative pl-6 space-y-6 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-[#e6dfd8]">
          {evidenceTimeline.map((item, idx) => (
            <div key={idx} className="relative group">
              {/* Dot */}
              <div
                className={`absolute -left-6 top-1 w-4 h-4 rounded-full border-2 border-white shadow-xs flex items-center justify-center ${
                  item.confidence === 'critical'
                    ? 'bg-[#c64545]'
                    : item.confidence === 'high'
                    ? 'bg-[#8f482f]'
                    : 'bg-[#5db8a6]'
                }`}
              />

              <div className="bg-white border border-[#e6dfd8] rounded-lg p-3.5 hover:border-[#8f482f] transition-all shadow-xs">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                  <span className="font-mono text-xs font-bold text-[#141413]">
                    {item.time}
                  </span>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-[#efeeea] text-[#54433e] font-semibold">
                    {item.source}
                  </span>
                </div>
                <p className="text-xs sm:text-sm text-[#3d3d3a] font-medium leading-snug">
                  {item.event}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
