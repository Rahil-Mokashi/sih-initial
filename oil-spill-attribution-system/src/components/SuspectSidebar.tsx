import React from 'react';
import { CaseRecord, SuspectVessel } from '../types';
import { 
  CheckCircle2, 
  AlertTriangle, 
  Satellite, 
  Wind, 
  Compass, 
  Route, 
  FileText, 
  Flag, 
  Maximize2,
  TrendingDown
} from 'lucide-react';

interface SuspectSidebarProps {
  currentCase: CaseRecord;
  onOpenDossier: (vessel: SuspectVessel) => void;
  onSelectVessel?: (vessel: SuspectVessel) => void;
}

export const SuspectSidebar: React.FC<SuspectSidebarProps> = ({
  currentCase,
  onOpenDossier,
}) => {
  const suspect = currentCase.topSuspect;

  return (
    <div className="w-full lg:w-[400px] border-l border-[#e6dfd8] bg-[#faf9f5] flex flex-col h-full overflow-y-auto shrink-0 z-10">
      {/* Panel Header */}
      <div className="p-4 sm:p-5 border-b border-[#e6dfd8] bg-[#f5f0e8]">
        <h3 className="font-serif text-lg font-bold text-[#141413]">
          Top Suspect Profile
        </h3>
        <p className="text-xs text-[#6c6a64] mt-0.5">
          Analyzing evidence correlated with drift back trajectory.
        </p>
      </div>

      {/* Panel Content Scroll Area */}
      <div className="p-4 sm:p-5 flex-1 space-y-5">
        {/* Suspect Primary Card */}
        <div className="bg-white border-2 border-[#e8a55a]/60 rounded-xl p-4 sm:p-5 shadow-xs relative overflow-hidden">
          <div className="absolute top-0 left-0 w-1.5 h-full bg-[#e8a55a]"></div>
          
          <div className="flex justify-between items-start mb-3">
            <div>
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-[#e8a55a]/20 text-[#8f482f] mb-1 inline-block">
                TOP SUSPECT #1
              </span>
              <h4 className="font-serif text-xl sm:text-2xl font-bold text-[#141413] leading-tight">
                {suspect.name}
              </h4>
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-[#6c6a64] font-mono mt-1">
                <span>IMO: {suspect.imo}</span>
                <span>·</span>
                <span>MMSI: {suspect.mmsi}</span>
              </div>
              <p className="text-xs text-[#6c6a64] mt-1 flex items-center gap-1">
                <Flag className="w-3.5 h-3.5 text-[#8f482f]" />
                <span>{suspect.flag} ({suspect.countryCode})</span>
              </p>
            </div>

            <div className="bg-[#5db872]/15 text-[#0f8378] px-2.5 py-1 rounded-md flex items-center gap-1 font-mono text-xs font-bold border border-[#5db872]/30 shadow-xs">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{suspect.matchScore}% Match</span>
            </div>
          </div>

          {/* Behavior / AIS gap metrics */}
          <div className="space-y-2 mt-4 pt-3 border-t border-[#e6dfd8]">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#6c6a64] flex items-center gap-1.5">
                <Satellite className="w-3.5 h-3.5 text-[#d4a017]" />
                <span>AIS Signal Status</span>
              </span>
              <span className="font-mono text-xs sm:text-sm font-bold text-[#d4a017] bg-[#d4a017]/10 px-2 py-0.5 rounded">
                {suspect.aisStatus}
              </span>
            </div>

            {suspect.unaccountedMovementKm != null && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-[#6c6a64] flex items-center gap-1.5">
                  <Route className="w-3.5 h-3.5 text-[#d4a017]" />
                  <span>Movement During AIS Gap</span>
                </span>
                <span className="font-mono text-xs sm:text-sm font-bold text-[#d4a017]">
                  {suspect.unaccountedMovementKm.toFixed(0)} km unaccounted
                </span>
              </div>
            )}
          </div>

          {/* View Full Dossier Button */}
          <button
            onClick={() => onOpenDossier(suspect)}
            className="w-full mt-4 py-2.5 px-3 border border-[#8f482f] bg-[#8f482f] hover:bg-[#a9583e] active:scale-98 text-white rounded-lg text-xs sm:text-sm font-semibold transition-all flex items-center justify-center gap-2 shadow-xs"
          >
            <FileText className="w-4 h-4" />
            <span>View Full Dossier</span>
          </button>
        </div>

        {/* Real Scoring Evidence Cards -- src/attribution/score_vessels.py's
            actual inputs, not fabricated sensor-fusion percentages. */}
        <div className="space-y-2.5">
          <h5 className="text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
            Scoring Evidence (real GFW/drift data)
          </h5>

          {/* Evidence 1: Proximity */}
          <div className="bg-white border border-[#e6dfd8] rounded-xl p-3 flex items-start gap-3 shadow-xs">
            <div className="w-9 h-9 rounded-lg bg-[#efeeea] flex items-center justify-center text-[#8f482f] shrink-0">
              <Satellite className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs sm:text-sm font-bold text-[#141413]">Proximity to Estimated Origin</p>
              <p className="text-xs text-[#6c6a64] mt-0.5">
                Closest approach {suspect.closestApproachKm} km from the drift-estimated origin.
              </p>
            </div>
          </div>

          {/* Evidence 2: Timing */}
          <div className="bg-white border border-[#e6dfd8] rounded-xl p-3 flex items-start gap-3 shadow-xs">
            <div className="w-9 h-9 rounded-lg bg-[#efeeea] flex items-center justify-center text-[#5db8a6] shrink-0">
              <Route className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs sm:text-sm font-bold text-[#141413]">Temporal Window</p>
              <p className="text-xs text-[#6c6a64] mt-0.5">
                {suspect.timeGapHours === 0
                  ? 'Present exactly at the estimated origin time.'
                  : `${suspect.timeGapHours.toFixed(1)}h from the estimated origin time.`}
                {' '}{suspect.gfwPresenceRecords} real GFW presence records in the search window.
              </p>
            </div>
          </div>

          {/* Evidence 3: AIS behavior */}
          <div className="bg-white border border-[#e6dfd8] rounded-xl p-3 flex items-start gap-3 shadow-xs">
            <div className="w-9 h-9 rounded-lg bg-[#efeeea] flex items-center justify-center text-[#0f8378] shrink-0">
              <Wind className="w-5 h-5" />
            </div>
            <div>
              <p className="text-xs sm:text-sm font-bold text-[#141413]">AIS Behavior</p>
              <p className="text-xs text-[#6c6a64] mt-0.5">
                {suspect.aisGapIntentional
                  ? `Likely-intentional ${suspect.aisGapHours?.toFixed(1)}h AIS gap covering ${suspect.unaccountedMovementKm?.toFixed(0)} km.`
                  : 'No intentional AIS gap detected for this vessel.'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Provenance Stamp */}
      <div className="p-3 border-t border-[#e6dfd8] bg-[#f4f4f0]">
        <p className="text-[10px] font-mono text-[#6c6a64] text-center opacity-85">
          First-pass heuristic scoring (distance + timing + AIS-gap behavior) -- not calibrated against labeled ground truth
        </p>
      </div>
    </div>
  );
};
