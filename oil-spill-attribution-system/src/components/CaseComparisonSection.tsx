import React from 'react';
import { CaseRecord } from '../types';
import { ArrowRight, CheckCircle2 } from 'lucide-react';

interface CaseComparisonSectionProps {
  cases: CaseRecord[];
  currentCaseId: string;
  onSelectCase: (id: string) => void;
}

export const CaseComparisonSection: React.FC<CaseComparisonSectionProps> = ({
  cases,
  currentCaseId,
  onSelectCase,
}) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-serif text-xl font-bold text-[#141413]">
          Case Comparison Matrix
        </h3>
        <span className="text-xs font-mono text-[#6c6a64]">
          {cases.length} Registered Cases
        </span>
      </div>

      <div className="overflow-x-auto bg-[#faf9f5] border border-[#e6dfd8] rounded-xl shadow-xs">
        <table className="w-full text-left border-collapse whitespace-nowrap">
          <thead>
            <tr className="bg-[#f5f0e8] border-b border-[#e6dfd8] text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
              <th className="py-3 px-4 sm:px-6">Case</th>
              <th className="py-3 px-4">Region / Location</th>
              <th className="py-3 px-4">Detection Time</th>
              <th className="py-3 px-4">Detection → Origin</th>
              <th className="py-3 px-4">ERA5 vs NCEP Disagreement</th>
              <th className="py-3 px-4">Vessels Evaluated</th>
              <th className="py-3 px-4 sm:px-6">Top Suspect</th>
              <th className="py-3 px-4 text-right">Switch</th>
            </tr>
          </thead>
          <tbody className="text-xs divide-y divide-[#e6dfd8]">
            {cases.map((c) => {
              const isActive = c.id === currentCaseId;
              return (
                <tr
                  key={c.id}
                  onClick={() => onSelectCase(c.id)}
                  className={`cursor-pointer transition-colors ${
                    isActive
                      ? 'bg-[#e8e0d2] font-semibold'
                      : 'hover:bg-[#efe9de] bg-white'
                  }`}
                >
                  <td className="py-3.5 px-4 sm:px-6 font-mono font-bold text-sm text-[#8f482f]">
                    {c.code}
                  </td>
                  <td className="py-3.5 px-4 text-[#141413]">
                    {c.locationName}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-[#6c6a64]">
                    {c.detectionTime}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-[#141413]">
                    {c.environmental.driftBackDistanceKm} km
                  </td>
                  <td className="py-3.5 px-4 font-mono text-[#d4a017]">
                    {c.environmental.environmentalDisagreementKm} km
                  </td>
                  <td className="py-3.5 px-4 font-mono text-[#141413]">
                    {c.environmental.vesselsEvaluated}
                  </td>
                  <td className="py-3.5 px-4 sm:px-6">
                    <span className="font-serif font-bold text-[#141413]">
                      {c.topSuspect.name}
                    </span>
                    <span className="font-mono ml-2 font-bold text-[#0f8378]">
                      {c.topSuspect.matchScore}%
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    {isActive ? (
                      <span className="inline-flex items-center gap-1 text-[11px] font-mono text-[#8f482f] bg-[#8f482f]/10 px-2 py-0.5 rounded font-bold">
                        <CheckCircle2 className="w-3 h-3" /> Active
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-[#6c6a64] hover:text-[#8f482f]">
                        Load <ArrowRight className="w-3 h-3" />
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
