import React from 'react';
import { ActiveTab, CaseRecord } from '../types';
import { 
  Radar, 
  Waves, 
  History, 
  FolderGit2, 
  Download, 
  HelpCircle, 
  Activity, 
  ShieldCheck,
  Menu,
  X
} from 'lucide-react';

interface SidebarProps {
  currentCase: CaseRecord;
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  onExportCaseFile: () => void;
  onOpenHelp: () => void;
  onOpenSystemStatus: () => void;
  mobileOpen?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentCase,
  activeTab,
  onTabChange,
  onExportCaseFile,
  onOpenHelp,
  onOpenSystemStatus,
  mobileOpen = false,
  onCloseMobile,
}) => {
  const navItems = [
    { id: 'detection' as ActiveTab, label: 'Detection', icon: Radar },
    { id: 'drift' as ActiveTab, label: 'Drift Analysis', icon: Waves },
    { id: 'attribution' as ActiveTab, label: 'Attribution', icon: History },
    { id: 'evidence' as ActiveTab, label: 'Evidence Hub', icon: FolderGit2 },
  ];

  const sidebarContent = (
    <div className="flex flex-col h-full p-5 justify-between">
      {/* Brand Header */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-[#8f482f] flex items-center justify-center text-white shadow-xs">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <h2 className="font-sans text-base font-bold text-[#8f482f] leading-tight tracking-tight">
                NTRO Intelligence
              </h2>
              <p className="font-mono text-xs text-[#6c6a64] uppercase tracking-wider">
                Case ID: {currentCase.code}
              </p>
            </div>
          </div>
          {onCloseMobile && (
            <button
              onClick={onCloseMobile}
              className="md:hidden p-1 text-[#6c6a64] hover:text-[#141413]"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Navigation items */}
        <nav className="space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onTabChange(item.id);
                  if (onCloseMobile) onCloseMobile();
                }}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm transition-all text-left ${
                  isActive
                    ? 'bg-[#e8e0d2] text-[#8f482f] font-bold shadow-xs'
                    : 'text-[#6c6a64] hover:bg-[#efeeea] hover:text-[#141413]'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-[#8f482f]' : 'text-[#6c6a64]'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer controls */}
      <div className="space-y-4 pt-6 border-t border-[#e6dfd8]">
        <button
          onClick={onExportCaseFile}
          className="w-full py-2.5 px-3 bg-[#8f482f] hover:bg-[#a9583e] active:scale-98 text-white rounded text-sm font-medium transition-all flex items-center justify-center gap-2 shadow-xs"
        >
          <Download className="w-4 h-4" />
          <span>Export Case File</span>
        </button>

        <div className="space-y-1 text-xs text-[#6c6a64]">
          <button
            onClick={onOpenHelp}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-[#efeeea] hover:text-[#141413] transition-colors text-left"
          >
            <HelpCircle className="w-4 h-4 text-[#6c6a64]" />
            <span>Help Center</span>
          </button>
          <button
            onClick={onOpenSystemStatus}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-[#efeeea] hover:text-[#141413] transition-colors text-left"
          >
            <Activity className="w-4 h-4 text-[#6c6a64]" />
            <span>System Status</span>
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop docked sidebar */}
      <aside className="hidden md:flex flex-col w-64 h-full border-r border-[#e6dfd8] bg-[#f4f4f0] shrink-0 z-10">
        {sidebarContent}
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          <div className="fixed inset-0 bg-black/40 backdrop-blur-xs" onClick={onCloseMobile} />
          <div className="relative w-64 max-w-[80%] bg-[#f4f4f0] h-full shadow-xl z-10">
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
};
