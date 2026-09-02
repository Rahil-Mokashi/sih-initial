import React, { useState } from 'react';
import { SuspectVessel, CaseRecord } from '../types';
import { 
  X, 
  CheckCircle2, 
  AlertTriangle, 
  Satellite, 
  Radio, 
  Wind, 
  Route, 
  FileText, 
  ShieldAlert, 
  Flag, 
  Ship, 
  Anchor, 
  Clock, 
  Download,
  Share2
} from 'lucide-react';

interface VesselDossierModalProps {
  vessel: SuspectVessel | null;
  currentCase: CaseRecord;
  onClose: () => void;
}

export const VesselDossierModal: React.FC<VesselDossierModalProps> = ({
  vessel,
  currentCase,
  onClose,
}) => {
  if (!vessel) return null;

  const [activeDossierTab, setActiveDossierTab] = useState<'overview' | 'ais' | 'identity'>('overview');
  const [isFlagged, setIsFlagged] = useState(false);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/60 backdrop-blur-xs animate-fadeIn">
      <div className="bg-[#faf9f5] w-full max-w-4xl max-h-[90vh] rounded-2xl border border-[#e6dfd8] shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="p-4 sm:p-6 bg-[#efe9de] border-b border-[#e6dfd8] flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-[#8f482f] text-white flex items-center justify-center shadow-xs">
              <Ship className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 bg-[#e8a55a] text-[#141413] rounded">
                  RANK #{vessel.rank}
                </span>
                <h3 className="font-serif text-xl sm:text-2xl font-bold text-[#141413]">
                  {vessel.name}
                </h3>
              </div>
              <p className="text-xs text-[#6c6a64] font-mono mt-0.5">
                IMO: {vessel.imo} · MMSI: {vessel.mmsi} · Flag: {vessel.flag} ({vessel.countryCode})
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="bg-[#5db872]/20 text-[#0f8378] px-3 py-1.5 rounded-lg border border-[#5db872]/40 font-mono text-sm font-bold flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" />
              <span>{vessel.matchScore}% Match Score</span>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-full hover:bg-[#e8e0d2] text-[#6c6a64] hover:text-[#141413] transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="px-4 sm:px-6 bg-[#f4f4f0] border-b border-[#e6dfd8] flex gap-2 overflow-x-auto shrink-0">
          {[
            { id: 'overview', label: 'Attribution Overview' },
            { id: 'ais', label: 'AIS Gap Evidence' },
            { id: 'identity', label: 'Vessel Identity' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveDossierTab(tab.id as any)}
              className={`py-3 px-3 text-xs sm:text-sm font-sans font-medium border-b-2 whitespace-nowrap transition-colors ${
                activeDossierTab === tab.id
                  ? 'border-[#8f482f] text-[#8f482f] font-bold'
                  : 'border-transparent text-[#6c6a64] hover:text-[#141413]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Modal Scrollable Content Body */}
        <div className="p-4 sm:p-6 flex-1 overflow-y-auto space-y-6">
          {/* Tab 1: Attribution Overview */}
          {activeDossierTab === 'overview' && (
            <div className="space-y-5">
              <div className="bg-white border border-[#e6dfd8] rounded-xl p-4 sm:p-5 shadow-xs">
                <h4 className="font-serif font-bold text-base text-[#141413] mb-2">
                  Forensic Summary & Causal Attribution
                </h4>
                <p className="text-xs sm:text-sm text-[#3d3d3a] leading-relaxed">
                  {vessel.behaviorSummary}
                </p>

                {/* Real Scoring Inputs -- score_vessels.py's actual composite
                    score components, not fabricated confidence percentages. */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4 pt-4 border-t border-[#e6dfd8]">
                  <div className="bg-[#f4f4f0] p-3 rounded-lg border border-[#e6dfd8]">
                    <span className="text-[10px] font-mono text-[#6c6a64] uppercase">Closest Approach</span>
                    <p className="font-mono text-lg font-bold text-[#141413]">
                      {vessel.closestApproachKm} km
                    </p>
                    <span className="text-[10px] text-[#6c6a64]">Distance from estimated origin</span>
                  </div>
                  <div className="bg-[#f4f4f0] p-3 rounded-lg border border-[#e6dfd8]">
                    <span className="text-[10px] font-mono text-[#6c6a64] uppercase">Time Gap</span>
                    <p className="font-mono text-lg font-bold text-[#141413]">
                      +{vessel.timeGapHours}h
                    </p>
                    <span className="text-[10px] text-[#6c6a64]">vs. estimated origin time</span>
                  </div>
                  <div className="bg-[#f4f4f0] p-3 rounded-lg border border-[#e6dfd8]">
                    <span className="text-[10px] font-mono text-[#6c6a64] uppercase">AIS Behavior</span>
                    <p className={`font-mono text-lg font-bold ${vessel.aisGapIntentional ? 'text-[#c64545]' : 'text-[#0f8378]'}`}>
                      {vessel.aisGapIntentional ? 'Gap Flagged' : 'Compliant'}
                    </p>
                    <span className="text-[10px] text-[#6c6a64]">{vessel.aisStatus}</span>
                  </div>
                </div>
              </div>

              {/* Presence Record Log (real entry/exit from GFW's presence API --
                  grid-cell centroid, not a continuous GPS track) */}
              <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4">
                <h5 className="font-serif font-bold text-sm text-[#141413] mb-1">
                  GFW Presence Record
                </h5>
                <p className="text-[10px] text-[#6c6a64] mb-3">
                  Entry/exit of this vessel's presence window in the search area -- a grid-cell centroid position, not a continuous GPS track.
                </p>
                <div className="space-y-2">
                  {vessel.historicalPath.map((pt, i) => (
                    <div
                      key={i}
                      className="bg-white p-2.5 rounded border border-[#e6dfd8] flex flex-wrap items-center justify-between text-xs font-mono"
                    >
                      <span className="text-[#6c6a64]">{pt.time || '—'}</span>
                      <span className="text-[#141413] font-bold">{pt.lat.toFixed(2)}°N, {pt.lng.toFixed(2)}°E</span>
                      {pt.isDark ? (
                        <span className="text-[10px] bg-[#ffdad6] text-[#c64545] px-2 py-0.5 rounded font-bold">
                          AIS Gap Window
                        </span>
                      ) : (
                        <span className="text-[10px] bg-[#5db872]/20 text-[#0f8378] px-2 py-0.5 rounded">
                          AIS Broadcasting
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: AIS Gap Evidence */}
          {activeDossierTab === 'ais' && (
            <div className="space-y-4">
              {vessel.aisGapIntentional ? (
                <div className="bg-[#ffdad6]/40 border border-[#c64545]/40 rounded-xl p-4 sm:p-5">
                  <div className="flex items-center gap-2 text-[#c64545] font-bold text-sm mb-2">
                    <AlertTriangle className="w-5 h-5" />
                    <span>Likely-Intentional AIS Transponder Gap Detected</span>
                  </div>
                  <p className="text-xs sm:text-sm text-[#3d3d3a] leading-relaxed">
                    Global Fishing Watch flagged a {vessel.aisGapHours?.toFixed(1)}-hour transmission gap.
                    During this window the vessel's presence record shows {vessel.unaccountedMovementKm?.toFixed(0)} km
                    of unaccounted movement relative to the search area.
                  </p>
                </div>
              ) : (
                <div className="bg-[#5db872]/10 border border-[#5db872]/30 rounded-xl p-4 sm:p-5">
                  <p className="text-xs sm:text-sm text-[#3d3d3a] leading-relaxed">
                    No likely-intentional AIS gap was detected for this vessel in Global Fishing Watch's gap-events dataset.
                  </p>
                </div>
              )}

              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="bg-white p-4 rounded-xl border border-[#e6dfd8]">
                  <p className="text-[#6c6a64] text-[10px] uppercase">Presence Entry</p>
                  <p className="text-sm font-bold text-[#141413] mt-1">
                    {vessel.coordinates[0].toFixed(2)}° N, {vessel.coordinates[1].toFixed(2)}° E
                  </p>
                  <p className="text-[10px] text-[#6c6a64] mt-0.5">{vessel.entryTimestamp || 'n/a'}</p>
                </div>
                <div className="bg-white p-4 rounded-xl border border-[#e6dfd8]">
                  <p className="text-[#6c6a64] text-[10px] uppercase">Presence Exit</p>
                  <p className="text-sm font-bold text-[#141413] mt-1">
                    {vessel.coordinates[0].toFixed(2)}° N, {vessel.coordinates[1].toFixed(2)}° E
                  </p>
                  <p className="text-[10px] text-[#6c6a64] mt-0.5">{vessel.exitTimestamp || 'n/a'}</p>
                </div>
              </div>
              <p className="text-[10px] text-[#6c6a64] italic">
                Coordinates are GFW's grid-cell centroid at the requested spatial resolution, not an exact GPS fix.
              </p>
            </div>
          )}

          {/* Tab 3: Vessel Identity -- only fields the GFW presence API
              actually returns; length/draft/speed/owner/destination aren't
              in that response, so they're omitted rather than invented. */}
          {activeDossierTab === 'identity' && (
            <div className="bg-white border border-[#e6dfd8] rounded-xl p-5 shadow-xs">
              <h4 className="font-serif font-bold text-base text-[#141413] mb-1">
                Vessel Identity (Global Fishing Watch)
              </h4>
              <p className="text-[10px] text-[#6c6a64] mb-4">
                Registry details such as owner, destination, length, draft, and cruising speed are not part of GFW's presence-report API and aren't shown here.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs font-mono">
                <div>
                  <span className="text-[#6c6a64] text-[10px] uppercase font-sans">MMSI</span>
                  <p className="font-bold text-[#141413] text-sm mt-0.5">{vessel.mmsi}</p>
                </div>
                <div>
                  <span className="text-[#6c6a64] text-[10px] uppercase font-sans">IMO</span>
                  <p className="font-bold text-[#141413] text-sm mt-0.5">{vessel.imo}</p>
                </div>
                <div>
                  <span className="text-[#6c6a64] text-[10px] uppercase font-sans">Vessel Type</span>
                  <p className="font-bold text-[#141413] text-sm mt-0.5">{vessel.vesselType}</p>
                </div>
                <div>
                  <span className="text-[#6c6a64] text-[10px] uppercase font-sans">Flag</span>
                  <p className="font-bold text-[#141413] text-sm mt-0.5">{vessel.flag}</p>
                </div>
                <div>
                  <span className="text-[#6c6a64] text-[10px] uppercase font-sans">GFW Presence Records</span>
                  <p className="font-bold text-[#141413] text-sm mt-0.5">{vessel.gfwPresenceRecords}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer Controls */}
        <div className="p-4 sm:p-5 bg-[#efe9de] border-t border-[#e6dfd8] flex flex-wrap items-center justify-between gap-3 shrink-0">
          <button
            onClick={() => setIsFlagged(!isFlagged)}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all flex items-center gap-2 ${
              isFlagged
                ? 'bg-[#c64545] text-white shadow-md'
                : 'bg-white border border-[#c64545] text-[#c64545] hover:bg-[#ffdad6]/40'
            }`}
          >
            <ShieldAlert className="w-4 h-4" />
            <span>{isFlagged ? '✓ Flagged for Maritime Interception' : 'Flag for Port Authority Interception'}</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={() => window.print()}
              className="px-4 py-2 bg-white border border-[#e6dfd8] hover:bg-[#f4f4f0] text-[#141413] rounded-lg text-xs sm:text-sm font-medium transition-colors flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              <span>Print Dossier</span>
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-[#8f482f] hover:bg-[#a9583e] text-white rounded-lg text-xs sm:text-sm font-semibold transition-colors"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
