import React, { useEffect, useState } from 'react';
import { fetchCases } from './lib/api';
import { ActiveTab, CaseRecord, SuspectVessel } from './types';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { KpiStrip } from './components/KpiStrip';
import { DriftMapWorkspace } from './components/DriftMapWorkspace';
import { SuspectSidebar } from './components/SuspectSidebar';
import { RankedLeadsTable } from './components/RankedLeadsTable';
import { SarDetectionModule } from './components/SarDetectionModule';
import { CaseComparisonSection } from './components/CaseComparisonSection';
import { DataProvenanceGrid } from './components/DataProvenanceGrid';
import { EvidenceHubView } from './components/EvidenceHubView';
import { VesselDossierModal } from './components/VesselDossierModal';
import { PdfReportModal } from './components/PdfReportModal';
import { HelpCenterModal } from './components/HelpCenterModal';
import { SystemStatusModal } from './components/SystemStatusModal';
import { Menu } from 'lucide-react';

export function App() {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentCaseId, setCurrentCaseId] = useState<string>('ow-0001');
  const [activeTab, setActiveTab] = useState<ActiveTab>('drift');
  const [selectedDossierVessel, setSelectedDossierVessel] = useState<SuspectVessel | null>(null);
  const [selectedMapVessel, setSelectedMapVessel] = useState<SuspectVessel | null>(null);

  // Modals state
  const [pdfModalOpen, setPdfModalOpen] = useState(false);
  const [helpModalOpen, setHelpModalOpen] = useState(false);
  const [systemStatusModalOpen, setSystemStatusModalOpen] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Real case data, fetched from backend/main.py (FastAPI) -- built from
  // data/processed/dashboard/{drift,vessel_ranking}_*.json, the same real
  // files src/dashboard/build_dashboard.py's static HTML build reads.
  useEffect(() => {
    fetchCases()
      .then((fetched) => {
        setCases(fetched);
        if (fetched.length > 0 && !fetched.some((c) => c.id === currentCaseId)) {
          setCurrentCaseId(fetched[0].id);
        }
      })
      .catch((err) => setLoadError(err.message ?? String(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loadError) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#faf9f5] text-[#3d3d3a] p-8">
        <div className="max-w-lg text-center space-y-3">
          <h1 className="font-serif text-xl font-bold">Could not reach the backend API</h1>
          <p className="text-sm text-[#6c6a64]">{loadError}</p>
          <p className="text-xs text-[#6c6a64]">
            Start it with: <code className="bg-[#efeeea] px-1.5 py-0.5 rounded">venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000</code> (from the repo root)
          </p>
        </div>
      </div>
    );
  }

  if (cases.length === 0) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#faf9f5] text-[#6c6a64]">
        Loading real case data&hellip;
      </div>
    );
  }

  // Active Case Record
  const currentCase: CaseRecord = cases.find((c) => c.id === currentCaseId) || cases[0];

  const handleExportCaseFile = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentCase, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `NTRO_CASE_${currentCase.code}_FORENSIC_EXPORT.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#faf9f5] text-[#3d3d3a] font-sans antialiased">
      {/* Top Header */}
      <Header
        currentCase={currentCase}
        cases={cases}
        onSelectCase={(id) => setCurrentCaseId(id)}
        onOpenPdfReport={() => setPdfModalOpen(true)}
        onOpenSystemStatus={() => setSystemStatusModalOpen(true)}
        onOpenHelp={() => setHelpModalOpen(true)}
      />

      {/* Main Body Area: Sidebar + Scrollable View */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Responsive Mobile Header Button */}
        <button
          onClick={() => setMobileSidebarOpen(true)}
          className="md:hidden fixed bottom-4 left-4 z-40 bg-[#8f482f] text-white p-3 rounded-full shadow-lg flex items-center justify-center"
          title="Open Menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Sidebar */}
        <Sidebar
          currentCase={currentCase}
          activeTab={activeTab}
          onTabChange={(tab) => setActiveTab(tab)}
          onExportCaseFile={handleExportCaseFile}
          onOpenHelp={() => setHelpModalOpen(true)}
          onOpenSystemStatus={() => setSystemStatusModalOpen(true)}
          mobileOpen={mobileSidebarOpen}
          onCloseMobile={() => setMobileSidebarOpen(false)}
        />

        {/* Scrollable Center Work Area */}
        <main className="flex-1 flex flex-col overflow-y-auto bg-[#faf9f5]">
          {/* KPI Strip at Top of Main View */}
          <KpiStrip currentCase={currentCase} />

          {/* Dynamic Content View based on Tab */}
          <div className="p-4 sm:p-8 space-y-8 max-w-[1600px] w-full mx-auto">
            {/* TAB: DRIFT ANALYSIS (Default Primary Forensic Workspace) */}
            {activeTab === 'drift' && (
              <>
                {/* 2-Column Split: Interactive Tactical Map + Top Suspect Profile */}
                <div className="flex flex-col lg:flex-row gap-6 items-stretch">
                  <DriftMapWorkspace
                    currentCase={currentCase}
                    onSelectVessel={(vessel) => setSelectedDossierVessel(vessel)}
                    selectedVessel={selectedMapVessel}
                  />
                  <SuspectSidebar
                    currentCase={currentCase}
                    onOpenDossier={(vessel) => setSelectedDossierVessel(vessel)}
                    onSelectVessel={(vessel) => setSelectedMapVessel(vessel)}
                  />
                </div>

                {/* Ranked Investigation Leads Table */}
                <RankedLeadsTable
                  candidates={currentCase.rankedCandidates}
                  nCandidatesTotal={currentCase.nCandidatesTotal}
                  onOpenDossier={(vessel) => setSelectedDossierVessel(vessel)}
                  onSelectVessel={(vessel) => setSelectedMapVessel(vessel)}
                />

                {/* Case Comparison Matrix */}
                <CaseComparisonSection
                  cases={cases}
                  currentCaseId={currentCase.id}
                  onSelectCase={(id) => setCurrentCaseId(id)}
                />

                {/* Data Provenance 4-Card Grid */}
                <DataProvenanceGrid currentCase={currentCase} />
              </>
            )}

            {/* TAB: DETECTION (SAR & Neural Segmentation) */}
            {activeTab === 'detection' && (
              <>
                <SarDetectionModule currentCase={currentCase} />
                <DataProvenanceGrid currentCase={currentCase} />
              </>
            )}

            {/* TAB: ATTRIBUTION (Deep Suspect Profiling & Ranking) */}
            {activeTab === 'attribution' && (
              <>
                <div className="flex flex-col lg:flex-row gap-6">
                  <div className="flex-1">
                    <RankedLeadsTable
                      candidates={currentCase.rankedCandidates}
                  nCandidatesTotal={currentCase.nCandidatesTotal}
                      onOpenDossier={(vessel) => setSelectedDossierVessel(vessel)}
                      onSelectVessel={(vessel) => setSelectedMapVessel(vessel)}
                    />
                  </div>
                  <SuspectSidebar
                    currentCase={currentCase}
                    onOpenDossier={(vessel) => setSelectedDossierVessel(vessel)}
                  />
                </div>
                <CaseComparisonSection
                  cases={cases}
                  currentCaseId={currentCase.id}
                  onSelectCase={(id) => setCurrentCaseId(id)}
                />
              </>
            )}

            {/* TAB: EVIDENCE HUB (Multi-Sensor Repository) */}
            {activeTab === 'evidence' && (
              <>
                <EvidenceHubView
                  currentCase={currentCase}
                  onOpenDossier={(vessel) => setSelectedDossierVessel(vessel)}
                />
                <DataProvenanceGrid currentCase={currentCase} />
              </>
            )}
          </div>
        </main>
      </div>

      {/* Modals & Dialogs */}
      {selectedDossierVessel && (
        <VesselDossierModal
          vessel={selectedDossierVessel}
          currentCase={currentCase}
          onClose={() => setSelectedDossierVessel(null)}
        />
      )}

      {pdfModalOpen && (
        <PdfReportModal
          currentCase={currentCase}
          onClose={() => setPdfModalOpen(false)}
        />
      )}

      {helpModalOpen && (
        <HelpCenterModal onClose={() => setHelpModalOpen(false)} />
      )}

      {systemStatusModalOpen && (
        <SystemStatusModal onClose={() => setSystemStatusModalOpen(false)} />
      )}
    </div>
  );
}

export default App;
