import React, { useState, useEffect, useMemo } from 'react';
import { Clock } from 'lucide-react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ResultGrid from './components/ResultGrid';
import ResultDetailModal from './components/ResultDetailModal';
import ClipboardPanel from './components/ClipboardPanel';
import { MOCK_RESULTS } from './data/mockResults';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8090';

const createDefaultPanel = (idNumber = 1) => ({
  id: `panel-${Date.now()}-${idNumber}`,
  type: 'text',
  query: '',
  enabled: true,
});

function formatHistoryTimestamp(now = new Date()) {
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())} ${pad(now.getDate())}/${pad(now.getMonth() + 1)}`;
}

export default function App() {
  const [panels, setPanels] = useState([createDefaultPanel(1)]);
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useState(10);
  const [apiError, setApiError] = useState(null);
  const [searchError, setSearchError] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ reachable: false, models_loaded: false });
  const [queryTime, setQueryTime] = useState(null);
  const [startTime, setStartTime] = useState(null);

  const [clipboardTab, setClipboardTab] = useState('note');
  const [noteText, setNoteText] = useState('');
  const [rankingList, setRankingList] = useState([]);
  const [queryHistory, setQueryHistory] = useState([]);

  const rankingKeyframeIds = useMemo(() => {
    const s = new Set();
    rankingList.forEach((it) => {
      if (it.keyframe_id) s.add(it.keyframe_id);
      if (it.id) s.add(it.id);
    });
    return s;
  }, [rankingList]);

  // Hook to handle real-time query execution timer
  useEffect(() => {
    let intervalId;
    if (isSearching) {
      const start = performance.now();
      setStartTime(start);
      setQueryTime('0.00');
      intervalId = setInterval(() => {
        const current = performance.now();
        setQueryTime(((current - start) / 1000).toFixed(2));
      }, 50);
    } else {
      if (startTime) {
        const end = performance.now();
        setQueryTime(((end - startTime) / 1000).toFixed(2));
      }
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isSearching]);

  // Reset entire application state to default
  const handleReset = () => {
    setPanels([createDefaultPanel(1)]);
    setResults([]);
    setSelectedResult(null);
    setSearchError(null);
    setApiError(null);
    setQueryTime(null);
    setStartTime(null);
    setRankingList([]);
  };

  // Panel updates
  const handleUpdatePanel = (updatedPanel) => {
    setPanels((prev) =>
      prev.map((p) => (p.id === updatedPanel.id ? updatedPanel : p))
    );
  };

  // Add new panel to bottom
  const handleAddPanel = () => {
    setPanels((prev) => [...prev, createDefaultPanel(prev.length + 1)]);
  };

  // Remove last panel (keep at least 1)
  const handleRemovePanel = () => {
    setPanels((prev) => {
      if (prev.length <= 1) return prev;
      return prev.slice(0, prev.length - 1);
    });
  };

  // Ranking handlers
  const handleAddToRanking = (item) => {
    setRankingList((prev) => {
      const kfId = item.keyframe_id || item.id;
      const exists = prev.some((it) => (it.keyframe_id || it.id) === kfId);
      if (exists) return prev;
      return [...prev, { ...item }];
    });
  };

  const handleMoveRankingUp = (idx) => {
    if (idx <= 0) return;
    setRankingList((prev) => {
      const next = [...prev];
      [next[idx - 1], next[idx]] = [next[idx], next[idx - 1]];
      return next;
    });
  };

  const handleMoveRankingDown = (idx) => {
    setRankingList((prev) => {
      if (idx >= prev.length - 1) return prev;
      const next = [...prev];
      [next[idx + 1], next[idx]] = [next[idx], next[idx + 1]];
      return next;
    });
  };

  const handleRemoveRankingItem = (idx) => {
    setRankingList((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSelectRankingItem = (item) => {
    setSelectedResult(item);
  };

  // Query history handlers
  const pushHistoryEntry = (queryText, resultCount = 0) => {
    if (!queryText || !queryText.trim()) return;
    const entry = {
      id: `h-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      query: queryText.trim(),
      resultCount,
      timestamp: formatHistoryTimestamp(),
    };
    setQueryHistory((prev) => [entry, ...prev].slice(0, 50));
  };

  const handleReplayQuery = (entry) => {
    if (!entry || !entry.query) return;
    const q = entry.query;
    setPanels((prev) => {
      if (prev.length === 0) return [{ ...createDefaultPanel(1), query: q }];
      return prev.map((p, i) => (i === 0 ? { ...p, query: q, enabled: true } : p));
    });
  };

  const handleClearHistory = () => {
    setQueryHistory([]);
  };

  // Perform search (filter mock results based on enabled panels)
  const handleSearch = async () => {
    setIsSearching(true);
    setApiError(null);
    setSearchError(null);

    const activePanels = panels.filter((p) => p.enabled && p.query.trim() !== '');
    const queries = activePanels.map((p) => p.query.trim()).filter(Boolean);

    if (queries.length === 0) {
      setResults(MOCK_RESULTS.slice(0, topK));
      console.debug('Search: using MOCK_RESULTS (no query provided)');
      setIsSearching(false);
      return;
    }

    const combinedQuery = queries.join(' | ');
    const payload = {
      query: combinedQuery,
      top_k: topK,
      rrf_k: 60,
      video_list: [],
      use_ocr: activePanels.some((p) => p.type === 'ocr'),
      use_asr: activePanels.some((p) => p.type === 'asr'),
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || response.statusText || 'Search request failed');
      }

      if (!Array.isArray(data.results) || data.results.length === 0) {
        setSearchError('Không tìm thấy kết quả phù hợp cho truy vấn này.');
        setResults([]);
        pushHistoryEntry(combinedQuery, 0);
        return;
      }

      const mappedResults = data.results.map((item, idx) => ({
        id: item.keyframe_id || `res-${idx}`,
        keyframe_id: item.keyframe_id || '',
        rank: item.rank ?? idx + 1,
        videoId: item.video_id || '',
        videoTitle: item.video_id || 'Keyframe result',
        frameId: item.frame_idx ?? '',
        timestamp: item.timestamp !== undefined && item.timestamp !== null ? item.timestamp : (item.pts_time !== undefined ? Math.round(item.pts_time * 1000) : ''),
        pts_time: item.pts_time,
        score: item.score ?? 0,
        matchedType: activePanels[0]?.type || 'text',
        caption: item.keyframe_id || item.video_id || 'Kết quả tìm kiếm',
        thumbnailUrl: item.image_url,
      }));

      const finalResults = mappedResults.slice(0, topK);
      setResults(finalResults);
      pushHistoryEntry(combinedQuery, finalResults.length);
    } catch (err) {
      console.error('Search API error:', err);
      const errorMessage = err.message ?? 'Lỗi khi gọi backend';
      setApiError(errorMessage);
      setSearchError(errorMessage);
      setResults([]);
      pushHistoryEntry(combinedQuery, 0);
    } finally {
      setIsSearching(false);
    }
  };

  // Check backend health on mount (shows whether frontend will use live backend)
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (!mounted) return;
        if (!res.ok) {
          setBackendStatus({ reachable: false, models_loaded: false });
          return;
        }
        const json = await res.json();
        setBackendStatus({ reachable: true, models_loaded: !!json.models_loaded });
      } catch (e) {
        if (!mounted) return;
        setBackendStatus({ reachable: false, models_loaded: false });
      }
    };
    check();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-slate-100 font-sans text-slate-800">
      {/* Header */}
      <Header onReset={handleReset} topK={topK} onChangeTopK={setTopK} />
      {/* Backend health & query status bar */}
      <div className="px-4 py-2 flex items-center justify-between bg-white border-b border-slate-200">
        <span className={`inline-flex items-center gap-2 rounded-md px-2 py-1 text-xs font-medium ${backendStatus.reachable ? (backendStatus.models_loaded ? 'bg-emerald-100 text-emerald-800' : 'bg-yellow-100 text-yellow-800') : 'bg-rose-100 text-rose-800'}`}>
          Backend: {backendStatus.reachable ? (backendStatus.models_loaded ? 'Live (models ready)' : 'Live (loading models)') : 'Unavailable (using mock)'}
        </span>
        
        <div className="flex items-center gap-2">
          {isSearching && (
            <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-50 border border-amber-200 px-2.5 py-1 text-xs font-semibold text-amber-700 shadow-sm animate-pulse">
              <svg className="animate-spin h-3.5 w-3.5 text-amber-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Đang tìm kiếm... <span className="font-mono font-bold text-amber-950">{queryTime}s</span>
            </span>
          )}

          {!isSearching && queryTime !== null && (
            <span className="inline-flex items-center gap-1.5 rounded-md bg-sky-50 border border-sky-200 px-2.5 py-1 text-xs font-semibold text-sky-700 shadow-sm transition-all duration-300">
              <Clock className="w-3.5 h-3.5 text-sky-500" />
              Thời gian thực hiện: <span className="font-mono text-sky-950 font-bold">{queryTime}s</span>
            </span>
          )}
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          panels={panels}
          onUpdatePanel={handleUpdatePanel}
          onAddPanel={handleAddPanel}
          onRemovePanel={handleRemovePanel}
          onSearch={handleSearch}
          isSearching={isSearching}
        />

        {/* Center Main Content: Result Grid */}
        <main className="flex-1 flex flex-col min-w-0 bg-slate-100 overflow-y-auto">
          <ResultGrid
            results={results}
            error={searchError}
            onSelectResult={(item) => setSelectedResult(item)}
            isSearching={isSearching}
            onAddToRanking={handleAddToRanking}
            rankingKeyframeIds={rankingKeyframeIds}
          />
        </main>

        {/* Right Sidebar: Clipboard Panel */}
        <ClipboardPanel
          activeTab={clipboardTab}
          onTabChange={setClipboardTab}
          noteText={noteText}
          onChangeNoteText={setNoteText}
          rankingList={rankingList}
          onMoveRankingUp={handleMoveRankingUp}
          onMoveRankingDown={handleMoveRankingDown}
          onRemoveRankingItem={handleRemoveRankingItem}
          onSelectRankingItem={handleSelectRankingItem}
          queryHistory={queryHistory}
          onReplayQuery={handleReplayQuery}
          onClearHistory={handleClearHistory}
        />
      </div>

      {apiError && (
        <div className="fixed bottom-4 right-4 max-w-sm rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 shadow-lg">
          <strong className="block font-semibold">Lỗi kết nối backend</strong>
          <p>{apiError}</p>
        </div>
      )}

      {/* Detail Modal */}
      {selectedResult && (
        <ResultDetailModal
          item={selectedResult}
          onClose={() => setSelectedResult(null)}
        />
      )}
    </div>
  );
}
