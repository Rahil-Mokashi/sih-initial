import React from 'react';
import { X, Activity, Server, Cpu, Database, CheckCircle2, ShieldCheck } from 'lucide-react';

interface SystemStatusModalProps {
  onClose: () => void;
}

export const SystemStatusModal: React.FC<SystemStatusModalProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-fadeIn">
      <div className="bg-[#faf9f5] w-full max-w-2xl max-h-[85vh] rounded-2xl border border-[#e6dfd8] shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-4 sm:p-5 bg-[#efe9de] border-b border-[#e6dfd8] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Activity className="w-5 h-5 text-[#8f482f]" />
            <h3 className="font-serif text-lg sm:text-xl font-bold text-[#141413]">
              System Pipeline Telemetry
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full hover:bg-[#e8e0d2] text-[#6c6a64]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 sm:p-6 flex-1 overflow-y-auto space-y-4 text-xs font-mono">
          <div className="space-y-2">
            {[
              { name: 'Copernicus Sentinel-1 Hub API', status: 'Operational', latency: '42ms', color: 'text-[#0f8378]' },
              { name: 'ECMWF ERA5 Meteorological Ingestion', status: 'Active (Hourly)', latency: '118ms', color: 'text-[#0f8378]' },
              { name: 'HYCOM Ocean Current Numerical Solver', status: 'Synchronized', latency: '89ms', color: 'text-[#0f8378]' },
              { name: 'Global Fishing Watch AIS Dark Vessel Pipe', status: 'Connected (Quota: 98.4% free)', latency: '65ms', color: 'text-[#0f8378]' },
              { name: 'OpenDrift Lagrangian Particle Engine', status: 'Ready (v2.1.4)', latency: '12ms', color: 'text-[#0f8378]' },
            ].map((node, i) => (
              <div key={i} className="bg-white p-3 rounded-lg border border-[#e6dfd8] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-[#5db872]" />
                  <span className="font-sans font-bold text-[#141413]">{node.name}</span>
                </div>
                <div className="text-right">
                  <span className={`font-bold ${node.color}`}>{node.status}</span>
                  <span className="text-[#6c6a64] ml-2">({node.latency})</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-[#efe9de] border-t border-[#e6dfd8] text-right">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#8f482f] hover:bg-[#a9583e] text-white rounded-lg text-xs font-semibold"
          >
            Close Telemetry
          </button>
        </div>
      </div>
    </div>
  );
};
