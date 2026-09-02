import React from 'react';
import { CaseRecord } from '../types';
import { X, Printer, Download, ShieldCheck, FileCheck, CheckCircle2 } from 'lucide-react';

interface PdfReportModalProps {
  currentCase: CaseRecord;
  onClose: () => void;
}

export const PdfReportModal: React.FC<PdfReportModalProps> = ({ currentCase, onClose }) => {
  const handlePrint = () => {
    window.print();
  };

  const { topSuspect, environmental, provenance } = currentCase;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-6 bg-black/70 backdrop-blur-xs animate-fadeIn overflow-y-auto">
      <div className="bg-white w-full max-w-4xl max-h-[92vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-[#e6dfd8]">
        {/* Modal Action Bar (Hidden on print) */}
        <div className="no-print p-4 bg-[#efe9de] border-b border-[#e6dfd8] flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-[#8f482f]" />
            <span className="font-serif font-bold text-sm text-[#141413]">
              Official Intelligence Report Preview — {currentCase.code}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="flex items-center gap-2 bg-[#8f482f] hover:bg-[#a9583e] text-white px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all shadow-xs"
            >
              <Printer className="w-4 h-4" />
              <span>Print / Save as PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-full hover:bg-[#e8e0d2] text-[#6c6a64] hover:text-[#141413] transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable Report Document Body */}
        <div className="p-6 sm:p-10 flex-1 overflow-y-auto bg-white text-[#141413] space-y-6 font-sans">
          {/* Formal Report Header */}
          <div className="border-b-2 border-[#8f482f] pb-5 flex flex-wrap justify-between items-start gap-4">
            <div>
              <div className="flex items-center gap-2 text-[#8f482f] font-bold text-xs uppercase tracking-widest font-mono">
                <span>NATIONAL TECHNICAL RESEARCH ORGANISATION</span>
              </div>
              <h1 className="font-serif text-2xl sm:text-3xl font-bold text-[#141413] mt-1">
                Oil Spill Forensic Attribution Dossier
              </h1>
              <p className="text-xs text-[#6c6a64] font-mono mt-0.5">
                SIH26143 / NTRO Maritime Intelligence · Security Classification: RESTRICTED
              </p>
            </div>

            <div className="bg-[#faf9f5] border border-[#e6dfd8] rounded-lg p-3 text-right text-xs font-mono">
              <p className="text-[#6c6a64]">Case Number</p>
              <p className="font-bold text-base text-[#8f482f]">{currentCase.code}</p>
              <p className="text-[10px] text-[#6c6a64] mt-1">Generated: {new Date().toLocaleDateString()}</p>
            </div>
          </div>

          {/* Executive Overview */}
          <div className="bg-[#faf9f5] border border-[#e6dfd8] rounded-xl p-4 sm:p-5 space-y-2">
            <h3 className="font-serif font-bold text-base text-[#8f482f] uppercase tracking-wider text-xs">
              Executive Incident Synopsis
            </h3>
            <p className="text-xs sm:text-sm text-[#3d3d3a] leading-relaxed">
              On <strong>{currentCase.detectionTime}</strong>, Sentinel-1 Synthetic Aperture Radar (SAR) detected an extensive mineral hydrocarbon slick in the <strong>{currentCase.locationName}</strong> region ({currentCase.coordinates.lat}°N, {currentCase.coordinates.lng}°E). Reverse Lagrangian numerical particle hindcasting over a 72-hour drift window established a consensus spill origin at <strong>{currentCase.originEstimatedTime}</strong>.
            </p>
          </div>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="bg-[#faf9f5] p-3 rounded-lg border border-[#e6dfd8]">
              <span className="text-[10px] text-[#6c6a64] uppercase font-sans">Drift Distance</span>
              <p className="font-bold text-base text-[#141413] mt-0.5">{environmental.driftBackDistanceKm} km</p>
            </div>
            <div className="bg-[#faf9f5] p-3 rounded-lg border border-[#e6dfd8]">
              <span className="text-[10px] text-[#6c6a64] uppercase font-sans">Model Variance</span>
              <p className="font-bold text-base text-[#d4a017] mt-0.5">{environmental.environmentalDisagreementKm} km</p>
            </div>
            <div className="bg-[#faf9f5] p-3 rounded-lg border border-[#e6dfd8]">
              <span className="text-[10px] text-[#6c6a64] uppercase font-sans">Candidate Pool</span>
              <p className="font-bold text-base text-[#141413] mt-0.5">{environmental.vesselsEvaluated} vessels</p>
            </div>
            <div className="bg-[#faf9f5] p-3 rounded-lg border border-[#e6dfd8]">
              <span className="text-[10px] text-[#6c6a64] uppercase font-sans">Primary Suspect Match</span>
              <p className="font-bold text-base text-[#0f8378] mt-0.5">{topSuspect.matchScore}%</p>
            </div>
          </div>

          {/* Primary Suspect Findings */}
          <div className="border border-[#e6dfd8] rounded-xl p-5 space-y-4">
            <div className="flex justify-between items-start border-b border-[#e6dfd8] pb-3">
              <div>
                <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded bg-[#e8a55a] text-[#141413]">
                  PRIMARY ATTRIBUTED VESSEL
                </span>
                <h2 className="font-serif text-xl font-bold text-[#141413] mt-1">
                  {topSuspect.name} (IMO: {topSuspect.imo})
                </h2>
                <p className="text-xs text-[#6c6a64] font-mono mt-0.5">
                  MMSI: {topSuspect.mmsi} · Flag: {topSuspect.flag}
                </p>
              </div>
              <span className="bg-[#5db872]/20 text-[#0f8378] font-mono font-bold text-sm px-3 py-1 rounded border border-[#5db872]/40">
                {topSuspect.matchScore}% Match
              </span>
            </div>

            <div className="space-y-2 text-xs text-[#3d3d3a] leading-relaxed">
              <p>
                <strong>Corroborating Evidence:</strong> {topSuspect.behaviorSummary}
              </p>
            </div>
          </div>

          {/* Data Provenance & Certification Block */}
          <div className="bg-[#faf9f5] border border-[#e6dfd8] rounded-xl p-4 text-xs space-y-2 font-mono">
            <p className="font-sans font-bold text-xs text-[#6c6a64] uppercase tracking-wider">
              Data Provenance & Scientific Validation
            </p>
            <p className="text-[#3d3d3a]">
              • Satellite Sensor: {currentCase.satelliteSensor}
            </p>
            <p className="text-[#3d3d3a]">
              • Numerical Ocean/Atmo: {provenance.driftModels}
            </p>
            <p className="text-[#3d3d3a]">
              • Maritime Tracking: Global Fishing Watch API (real AIS presence logs)
            </p>
          </div>

          {/* Sign-off block */}
          <div className="pt-6 border-t border-[#e6dfd8] flex justify-between items-end text-xs">
            <div>
              <p className="font-serif font-bold text-sm text-[#141413]">
                NTRO Maritime Forensic Investigation Team
              </p>
              <p className="text-[#6c6a64] font-mono mt-0.5">
                Auto-generated prototype report -- not an official NTRO document
              </p>
            </div>
            <div className="text-right">
              <div className="w-32 border-b border-[#141413] mb-1"></div>
              <p className="text-[#6c6a64] font-mono">Lead Intelligence Officer</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
