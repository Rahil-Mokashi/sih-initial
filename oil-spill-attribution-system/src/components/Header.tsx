import React from 'react';
import { CaseRecord } from '../types';
import { User, Settings, FileText, CheckCircle2, AlertCircle } from 'lucide-react';

interface HeaderProps {
  currentCase: CaseRecord;
  cases: CaseRecord[];
  onSelectCase: (caseId: string) => void;
  onOpenPdfReport: () => void;
  onOpenSystemStatus: () => void;
  onOpenHelp: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentCase,
  cases,
  onSelectCase,
  onOpenPdfReport,
  onOpenSystemStatus,
  onOpenHelp,
}) => {
  return (
    <header className="flex-none px-4 sm:px-8 py-3.5 border-b border-[#e6dfd8] bg-[#faf9f5] flex flex-wrap justify-between items-center gap-4 z-20">
      {/* Left: Brand / Title and status badges */}
      <div className="flex items-center gap-4 sm:gap-6">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-serif text-xl sm:text-2xl font-bold text-[#141413] tracking-tight">
              Oil Spill Attribution System
            </h1>
            <span className="hidden lg:inline-flex text-[11px] font-mono uppercase px-2 py-0.5 rounded bg-[#e8a55a]/20 text-[#8f482f] border border-[#e8a55a]/40 font-semibold tracking-wider">
              NTRO v1.4
            </span>
          </div>
          <p className="text-xs text-[#6c6a64] hidden sm:block">
            Satellite SAR detection · Reverse Lagrangian drift hindcast · AIS/GFW dark vessel correlation
          </p>
        </div>
      </div>

      {/* Center / Right: Case Tabs & Actions */}
      <div className="flex items-center gap-3 sm:gap-5 flex-wrap">
        {/* Case Switcher Tabs */}
        <div className="flex items-center gap-1 bg-[#efe9de] p-1 rounded-lg border border-[#e6dfd8]">
          {cases.map((c) => (
            <button
              key={c.id}
              onClick={() => onSelectCase(c.id)}
              className={`px-3 py-1 rounded-md text-xs sm:text-sm font-mono font-medium transition-all ${
                currentCase.id === c.id
                  ? 'bg-[#8f482f] text-white shadow-xs font-bold'
                  : 'text-[#6c6a64] hover:text-[#141413] hover:bg-[#e8e0d2]'
              }`}
            >
              {c.code}
            </button>
          ))}
        </div>

        {/* Generate PDF Report Button */}
        <button
          onClick={onOpenPdfReport}
          className="flex items-center gap-2 bg-[#8f482f] hover:bg-[#a9583e] active:scale-98 text-white px-3.5 py-1.5 rounded text-xs sm:text-sm font-medium transition-all shadow-xs"
        >
          <FileText className="w-4 h-4" />
          <span>Generate PDF Report</span>
        </button>

        {/* Action icons */}
        <div className="flex items-center gap-1.5 text-[#6c6a64]">
          <button
            onClick={onOpenHelp}
            title="Help Center & Methodology"
            className="w-8 h-8 rounded-full border border-[#e6dfd8] flex items-center justify-center hover:bg-[#efe9de] hover:text-[#141413] transition-colors"
          >
            <User className="w-4 h-4" />
          </button>
          <button
            onClick={onOpenSystemStatus}
            title="System Telemetry & Settings"
            className="w-8 h-8 rounded-full border border-[#e6dfd8] flex items-center justify-center hover:bg-[#efe9de] hover:text-[#141413] transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
