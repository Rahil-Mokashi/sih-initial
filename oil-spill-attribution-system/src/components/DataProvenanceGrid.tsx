import React from 'react';
import { CaseRecord } from '../types';
import { Database, Cpu, Wind, ShieldCheck } from 'lucide-react';

interface DataProvenanceGridProps {
  currentCase: CaseRecord;
}

export const DataProvenanceGrid: React.FC<DataProvenanceGridProps> = ({ currentCase }) => {
  const { provenance } = currentCase;

  return (
    <div className="space-y-3">
      <h3 className="font-serif text-xl font-bold text-[#141413]">
        Data Provenance & Pipeline Integrity
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Card 1: Training Dataset */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
              Training Dataset
            </span>
            <Database className="w-4 h-4 text-[#8f482f]" />
          </div>
          <p className="font-serif text-base sm:text-lg font-bold text-[#141413]">
            {provenance.trainingDataset}
          </p>
          <p className="text-xs text-[#6c6a64] mt-1.5 font-mono">
            {provenance.trainingTiles}
          </p>
        </div>

        {/* Card 2: Detection Model */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
              Detection Model
            </span>
            <Cpu className="w-4 h-4 text-[#8f482f]" />
          </div>
          <p className="font-serif text-base sm:text-lg font-bold text-[#141413]">
            {provenance.detectionModel}
          </p>
          <p className="text-xs text-[#0f8378] font-mono font-medium mt-1.5">
            {provenance.detectionValDice}
          </p>
        </div>

        {/* Card 3: Drift Data */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
              Drift MetOcean Solver
            </span>
            <Wind className="w-4 h-4 text-[#5db8a6]" />
          </div>
          <p className="font-serif text-base sm:text-lg font-bold text-[#141413]">
            {provenance.driftModels}
          </p>
          <p className="text-xs text-[#6c6a64] font-mono mt-1.5">
            {provenance.driftLastComputed}
          </p>
        </div>

        {/* Card 4: GFW API Quota */}
        <div className="bg-[#efe9de] border border-[#e6dfd8] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
              GFW AIS API Quota
            </span>
            <ShieldCheck className="w-4 h-4 text-[#5db872]" />
          </div>
          <p className="font-serif text-base sm:text-lg font-bold text-[#0f8378]">
            {provenance.gfwApiQuota}
          </p>
          <p className="text-xs text-[#6c6a64] font-mono mt-1.5">
            {provenance.gfwRequestsUsed}
          </p>
        </div>
      </div>
    </div>
  );
};
