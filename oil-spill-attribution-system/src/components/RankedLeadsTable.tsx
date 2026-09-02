import React, { useState } from 'react';
import { SuspectVessel } from '../types';
import { Search, Filter, ArrowUpDown, ExternalLink, Ship, AlertTriangle } from 'lucide-react';

interface RankedLeadsTableProps {
  candidates: SuspectVessel[];
  nCandidatesTotal?: number;
  onOpenDossier: (vessel: SuspectVessel) => void;
  onSelectVessel: (vessel: SuspectVessel) => void;
}

export const RankedLeadsTable: React.FC<RankedLeadsTableProps> = ({
  candidates,
  nCandidatesTotal,
  onOpenDossier,
  onSelectVessel,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'rank' | 'score' | 'distance'>('rank');
  const [showAll, setShowAll] = useState(false);

  // Filter tags pool
  const allTags = ['all', 'Near Origin', 'Speed Anomaly', 'AIS Gap', 'Course Change'];

  const filteredCandidates = candidates.filter((v) => {
    const matchesSearch = 
      v.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      v.mmsi.includes(searchQuery) ||
      v.imo.includes(searchQuery);
    
    const matchesTag = selectedTag === 'all' || v.evidenceTags.some(t => t.toLowerCase().includes(selectedTag.toLowerCase()));

    return matchesSearch && matchesTag;
  });

  const sortedCandidates = [...filteredCandidates].sort((a, b) => {
    if (sortBy === 'score') return b.matchScore - a.matchScore;
    if (sortBy === 'distance') return a.distFromOriginKm - b.distFromOriginKm;
    return a.rank - b.rank;
  });

  const displayedList = showAll ? sortedCandidates : sortedCandidates.slice(0, 5);

  return (
    <div className="bg-[#efe9de] rounded-xl border border-[#e6dfd8] overflow-hidden shadow-xs">
      {/* Table Header */}
      <div className="bg-[#e9e8e4] px-4 sm:px-6 py-3.5 border-b border-[#e6dfd8] flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-serif text-lg font-bold text-[#141413]">
            Ranked Investigation Leads
          </h2>
          <p className="text-xs text-[#6c6a64]">
            Composite scoring based on spatial proximity, temporal window, and dark vessel behavior
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-[#6c6a64] bg-[#f4f4f0] px-2.5 py-1 rounded border border-[#e6dfd8]">
            Showing {candidates.length}{nCandidatesTotal ? ` of ${nCandidatesTotal} real GFW candidates` : ''}
          </span>
        </div>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="p-3 sm:px-6 bg-[#faf9f5] border-b border-[#e6dfd8] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-1 max-w-sm">
          <div className="relative w-full">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#6c6a64]" />
            <input
              type="text"
              placeholder="Search vessel by name, MMSI, or IMO..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 text-xs bg-white border border-[#e6dfd8] rounded-lg focus:outline-none focus:border-[#8f482f]"
            />
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Quick Tag Filter Pills */}
          <div className="flex items-center gap-1">
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => setSelectedTag(tag)}
                className={`text-[11px] font-mono px-2 py-1 rounded transition-colors ${
                  selectedTag === tag
                    ? 'bg-[#8f482f] text-white font-bold'
                    : 'bg-[#efeeea] text-[#6c6a64] hover:bg-[#e8e0d2]'
                }`}
              >
                {tag === 'all' ? 'All Leads' : tag}
              </button>
            ))}
          </div>

          {/* Sort Selector */}
          <select
            value={sortBy}
            onChange={(e: any) => setSortBy(e.target.value)}
            className="text-xs bg-white border border-[#e6dfd8] rounded-lg px-2.5 py-1 text-[#3d3d3a] focus:outline-none"
          >
            <option value="rank">Sort: Rank</option>
            <option value="score">Sort: Match Score</option>
            <option value="distance">Sort: Distance to Origin</option>
          </select>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-[#f4f4f0] border-b border-[#e6dfd8] text-[11px] font-sans font-bold text-[#6c6a64] uppercase tracking-wider">
              <th className="py-3 px-4 sm:px-6">Rank</th>
              <th className="py-3 px-4">Vessel Name / Identity</th>
              <th className="py-3 px-4">Match Score</th>
              <th className="py-3 px-4">Dist. from Origin</th>
              <th className="py-3 px-4">Time Gap</th>
              <th className="py-3 px-4 sm:px-6">Evidence Tags</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="text-xs divide-y divide-[#e6dfd8] bg-white">
            {displayedList.map((v) => {
              const isTop = v.rank === 1;
              return (
                <tr
                  key={v.id}
                  onClick={() => onSelectVessel(v)}
                  className="hover:bg-[#faf9f5] transition-colors cursor-pointer group"
                >
                  {/* Rank */}
                  <td className="py-3.5 px-4 sm:px-6 font-mono font-bold text-sm text-[#8f482f]">
                    #{v.rank}
                  </td>

                  {/* Vessel Name & MMSI */}
                  <td className="py-3.5 px-4">
                    <div className="font-serif font-bold text-sm text-[#141413] flex items-center gap-1.5">
                      <span>{v.name}</span>
                      {isTop && (
                        <span className="text-[9px] font-mono uppercase px-1.5 py-0.2 bg-[#e8a55a] text-[#141413] rounded font-bold">
                          Top Suspect
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-[#6c6a64] font-mono mt-0.5">
                      MMSI {v.mmsi} · {v.countryCode} ({v.flag})
                    </div>
                  </td>

                  {/* Match Score Bar */}
                  <td className="py-3.5 px-4 min-w-[140px]">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-[#efeeea] rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            v.matchScore >= 90
                              ? 'bg-[#5db872]'
                              : v.matchScore >= 70
                              ? 'bg-[#e8a55a]'
                              : 'bg-[#d4a017]'
                          }`}
                          style={{ width: `${v.matchScore}%` }}
                        />
                      </div>
                      <span className="font-mono font-bold text-xs text-[#141413]">
                        {v.matchScore}%
                      </span>
                    </div>
                  </td>

                  {/* Distance */}
                  <td className="py-3.5 px-4 font-mono text-[#141413]">
                    {v.distFromOriginKm} km
                  </td>

                  {/* Time Gap */}
                  <td className="py-3.5 px-4 font-mono text-[#6c6a64]">
                    +{v.timeGapHours}h
                  </td>

                  {/* Evidence Tags */}
                  <td className="py-3.5 px-4 sm:px-6">
                    <div className="flex flex-wrap gap-1">
                      {v.evidenceTags.map((tag, idx) => (
                        <span
                          key={idx}
                          className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                            tag.includes('Blackout') || tag.includes('AIS Gap') || tag.includes('Dark')
                              ? 'bg-[#ffdad6] text-[#c64545] border border-[#c64545]/20 font-bold'
                              : tag.includes('Speed')
                              ? 'bg-[#e8a55a]/20 text-[#8f482f]'
                              : 'bg-[#efeeea] text-[#54433e]'
                          }`}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </td>

                  {/* Action */}
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onOpenDossier(v);
                      }}
                      className="inline-flex items-center gap-1 text-xs font-semibold text-[#8f482f] hover:text-[#a9583e] hover:underline"
                    >
                      <span>Dossier</span>
                      <ExternalLink className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Load More Candidates Button */}
      {candidates.length > 5 && (
        <div className="bg-[#f4f4f0] p-3 text-center border-t border-[#e6dfd8]">
          <button
            onClick={() => setShowAll(!showAll)}
            className="text-xs font-sans font-bold text-[#8f482f] hover:underline"
          >
            {showAll ? 'Show Top 5 Candidates Only' : `Load More Candidates (${candidates.length - 5} remaining)`}
          </button>
        </div>
      )}
    </div>
  );
};
