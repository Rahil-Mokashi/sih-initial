import React from 'react';
import { X, BookOpen, Waves, Satellite, Radio, ShieldCheck, CheckCircle2 } from 'lucide-react';

interface HelpCenterModalProps {
  onClose: () => void;
}

export const HelpCenterModal: React.FC<HelpCenterModalProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-fadeIn">
      <div className="bg-[#faf9f5] w-full max-w-3xl max-h-[85vh] rounded-2xl border border-[#e6dfd8] shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-4 sm:p-5 bg-[#efe9de] border-b border-[#e6dfd8] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <BookOpen className="w-5 h-5 text-[#8f482f]" />
            <h3 className="font-serif text-lg sm:text-xl font-bold text-[#141413]">
              System Methodology & Attribution Guide
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
        <div className="p-5 sm:p-6 flex-1 overflow-y-auto space-y-5 text-xs sm:text-sm text-[#3d3d3a]">
          <div className="space-y-2">
            <h4 className="font-serif font-bold text-base text-[#141413] flex items-center gap-2">
              <Satellite className="w-4 h-4 text-[#8f482f]" />
              1. Satellite SAR Slick Detection & Filtering
            </h4>
            <p className="leading-relaxed text-[#6c6a64]">
              Sentinel-1 Synthetic Aperture Radar (C-band VV) scans surface backscatter dampening caused by oil capillary wave suppression. A deep convolutional segmentation model extracts slick boundaries, rejecting biogenic lookalikes (algae, low-wind sheens) through dampening contrast variance.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-serif font-bold text-base text-[#141413] flex items-center gap-2">
              <Waves className="w-4 h-4 text-[#0f8378]" />
              2. Backward Lagrangian Drift Hindcasting
            </h4>
            <p className="leading-relaxed text-[#6c6a64]">
              Using Copernicus ERA5 10m wind fields, HYCOM hydrodynamic surface currents, and Stokes wave drift vectors, OpenDrift reverses time step-by-step (-72 hours) tracking 10,000 virtual particles to identify the precise geographical origin of discharge.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-serif font-bold text-base text-[#141413] flex items-center gap-2">
              <Radio className="w-4 h-4 text-[#d4a017]" />
              3. AIS Dark Vessel Correlation & Multi-Factor Scoring
            </h4>
            <p className="leading-relaxed text-[#6c6a64]">
              Vessels in the spatiotemporal search cone are evaluated with composite weighting:
            </p>
            <ul className="list-disc list-inside space-y-1 text-xs font-mono text-[#54433e]">
              <li><strong>Spatial Proximity (40%):</strong> Distance from particle origin centroid.</li>
              <li><strong>Temporal Consistency (30%):</strong> Intersecting time delta of ship position.</li>
              <li><strong>AIS Dark Gap Anomaly (30%):</strong> Flagging deliberate transponder blackouts.</li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-[#efe9de] border-t border-[#e6dfd8] text-right">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#8f482f] hover:bg-[#a9583e] text-white rounded-lg text-xs font-semibold"
          >
            Got It
          </button>
        </div>
      </div>
    </div>
  );
};
