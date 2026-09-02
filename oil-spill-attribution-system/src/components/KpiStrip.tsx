import React from 'react';
import { CaseRecord } from '../types';
import { CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';

interface KpiStripProps {
  currentCase: CaseRecord;
}

export const KpiStrip: React.FC<KpiStripProps> = ({ currentCase }) => {
  const { environmental } = currentCase;

  return (
    <div className="flex-none border-b border-[#e6dfd8] bg-[#f5f0e8]">
      {/* Case Header & Status row */}
      <div className="px-4 sm:px-8 pt-5 pb-4 border-b border-[#e6dfd8]/60 bg-[#faf9f5]">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-[#e8a55a] text-[#141413]">
                PROTOTYPE
              </span>
              <span className="text-xs text-[#6c6a64] font-medium">
                Real Drift + AIS Data · {currentCase.locationName}
              </span>
            </div>
            <h2 className="font-serif text-2xl sm:text-3xl font-bold text-[#141413] tracking-tight">
              Drift Reconstruction Dashboard — {currentCase.code}
            </h2>
          </div>

          {/* Status indicators */}
          <div className="flex items-center gap-3 text-xs">
            <span className="font-mono bg-[#efeeea] px-2.5 py-1 rounded text-[#1b1c1a] border border-[#e6dfd8] font-medium">
              {currentCase.code}
            </span>
            <span className="flex items-center gap-1.5 text-[#54433e]">
              <span className="w-2 h-2 rounded-full bg-[#e8a55a] animate-pulse"></span>
              Detection: <strong className="font-semibold">{currentCase.detectionStatus}</strong>
            </span>
            <span className="flex items-center gap-1.5 text-[#54433e]">
              <span className="w-2 h-2 rounded-full bg-[#5db872]"></span>
              Drift: <strong className="font-semibold">{currentCase.driftStatus}</strong>
            </span>
          </div>
        </div>

        <p className="mt-2 text-xs sm:text-sm text-[#6c6a64] max-w-4xl leading-relaxed">
          {currentCase.summary}
        </p>
      </div>

      {/* 4-Column KPI Grid */}
      <div className="px-4 sm:px-8 py-4 grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        {/* KPI 1: Drift Back Distance */}
        <div className="bg-[#efe9de] rounded-lg p-3.5 sm:p-4 border border-[#e6dfd8] shadow-xs flex flex-col justify-between">
          <p className="text-[11px] font-sans font-semibold tracking-wider text-[#6c6a64] uppercase mb-1">
            Drift Back Distance
          </p>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-2xl sm:text-3xl font-bold text-[#141413]">
              {environmental.driftBackDistanceKm}
            </span>
            <span className="font-sans text-sm text-[#6c6a64] font-medium">km</span>
          </div>
          <p className="text-[10px] text-[#6c6a64] mt-1">Satellite Slick → Estimated Origin</p>
        </div>

        {/* KPI 2: Vessels Evaluated */}
        <div className="bg-[#efe9de] rounded-lg p-3.5 sm:p-4 border border-[#e6dfd8] shadow-xs flex flex-col justify-between">
          <p className="text-[11px] font-sans font-semibold tracking-wider text-[#6c6a64] uppercase mb-1">
            Vessels Evaluated
          </p>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-2xl sm:text-3xl font-bold text-[#141413]">
              {environmental.vesselsEvaluated}
            </span>
            <span className="font-sans text-xs text-[#6c6a64]">total candidates</span>
          </div>
          <p className="text-[10px] text-[#6c6a64] mt-1">Filtered across temporal search cone</p>
        </div>

        {/* KPI 3: Top Match Score */}
        <div className="bg-[#efe9de] rounded-lg p-3.5 sm:p-4 border border-[#5db872]/40 relative overflow-hidden shadow-xs flex flex-col justify-between">
          <div className="absolute inset-0 bg-[#5db872]/10 pointer-events-none"></div>
          <p className="text-[11px] font-sans font-semibold tracking-wider text-[#0f8378] uppercase mb-1 relative z-10">
            Top Match Score
          </p>
          <div className="flex items-center gap-2 relative z-10">
            <span className="font-mono text-2xl sm:text-3xl font-bold text-[#0f8378]">
              {environmental.topMatchScore}%
            </span>
            <span className="flex items-center gap-0.5 text-xs text-[#0f8378] font-medium bg-[#5db872]/20 px-1.5 py-0.5 rounded">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Verified
            </span>
          </div>
          <p className="text-[10px] text-[#0f8378] font-medium mt-1 relative z-10">
            {currentCase.topSuspect.name} (Rank #1)
          </p>
        </div>

        {/* KPI 4: Environmental Disagreement */}
        <div className="bg-[#efe9de] rounded-lg p-3.5 sm:p-4 border border-[#e6dfd8] shadow-xs flex flex-col justify-between">
          <p className="text-[11px] font-sans font-semibold tracking-wider text-[#6c6a64] uppercase mb-1">
            Environmental Disagreement
          </p>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-2xl sm:text-3xl font-bold text-[#d4a017]">
              {environmental.environmentalDisagreementKm}
            </span>
            <span className="font-sans text-sm text-[#6c6a64]">km</span>
          </div>
          <p className="text-[10px] text-[#6c6a64] italic mt-1 truncate" title={environmental.varianceNote}>
            {environmental.varianceNote}
          </p>
        </div>
      </div>
    </div>
  );
};
