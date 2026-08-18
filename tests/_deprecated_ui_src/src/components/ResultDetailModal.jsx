import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { X, Film, Clock, Hash, Tag, ChevronLeft, ChevronRight, Send, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8090';

// Global cache for video keyframes & submitted entries
window.__videoKeyframesCache = window.__videoKeyframesCache || {};
window.__dresSubmittedSet = window.__dresSubmittedSet || new Set();

function formatMilliseconds(ms) {
  if (ms === null || ms === undefined || ms === '' || isNaN(ms)) return '00:00:00.000';
  const numMs = parseInt(ms, 10);
  const totalSec = Math.floor(numMs / 1000);
  const msec = String(numMs % 1000).padStart(3, '0');
  const hrs = String(Math.floor(totalSec / 3600)).padStart(2, '0');
  const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, '0');
  const secs = String(totalSec % 60).padStart(2, '0');
  return `${hrs}:${mins}:${secs}.${msec}`;
}

export default function ResultDetailModal({ item, onClose }) {
  if (!item) return null;

  const videoId = item.videoId;
  const [keyframes, setKeyframes] = useState(window.__videoKeyframesCache[videoId] || []);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loadingList, setLoadingList] = useState(!window.__videoKeyframesCache[videoId]);

  const [imgState, setImgState] = useState('loading');
  const [curSrc, setCurSrc] = useState(item.thumbnailUrl);
  const [retryCount, setRetryCount] = useState(0);

  const [submitStatus, setSubmitStatus] = useState('idle'); // 'idle' | 'submitting' | 'success' | 'error'
  const [submitResult, setSubmitResult] = useState(null);

  const [mode, setMode] = useState('semantic'); // 'semantic' | 'vqa'
  const [vqaAnswer, setVqaAnswer] = useState('');
  const [vqaSubmitStatus, setVqaSubmitStatus] = useState('idle'); // 'idle' | 'submitting' | 'success' | 'error'
  const [vqaSubmitResult, setVqaSubmitResult] = useState(null);

  // Preload adjacent images
  const preloadNeighbors = useCallback((list, activeIdx) => {
    if (!list || list.length === 0) return;
    const offsets = [-3, -2, -1, 1, 2, 3];
    offsets.forEach((offset) => {
      const targetIdx = activeIdx + offset;
      if (targetIdx >= 0 && targetIdx < list.length) {
        const kf = list[targetIdx];
        const rawUrl = kf.image_url || '';
        const fullUrl = rawUrl.startsWith('http')
          ? rawUrl
          : `${API_BASE_URL}${rawUrl.startsWith('/') ? '' : '/'}${rawUrl}`;
        const img = new Image();
        img.src = fullUrl;
      }
    });
  }, []);

  // Fetch video keyframes once and cache
  useEffect(() => {
    let isMounted = true;

    const resolveStartIdx = (list) => {
      const targetKid = item.keyframe_id || item.id || '';
      let idx = -1;
      if (targetKid) {
        idx = list.findIndex((k) => k.keyframe_id === targetKid);
      }
      if (idx < 0) {
        const fid = String(item.frameId);
        if (fid) {
          idx = list.findIndex((k) => String(k.frame_idx) === fid);
        }
      }
      return idx >= 0 ? idx : 0;
    };

    if (window.__videoKeyframesCache[videoId]) {
      const list = window.__videoKeyframesCache[videoId];
      setKeyframes(list);
      setLoadingList(false);
      const startIdx = resolveStartIdx(list);
      setCurrentIndex(startIdx);
      preloadNeighbors(list, startIdx);
    } else {
      setLoadingList(true);
      fetch(`${API_BASE_URL}/api/video/${videoId}/keyframes`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data) => {
          if (!isMounted) return;
          const list = data.keyframes || [];
          window.__videoKeyframesCache[videoId] = list;
          setKeyframes(list);
          setLoadingList(false);
          const startIdx = resolveStartIdx(list);
          setCurrentIndex(startIdx);
          preloadNeighbors(list, startIdx);
        })
        .catch((err) => {
          if (!isMounted) return;
          console.warn('Could not fetch video keyframes:', err);
          setLoadingList(false);
        });
    }
    return () => { isMounted = false; };
  }, [videoId, item.id, item.frameId, item.keyframe_id, preloadNeighbors]);

  // Determine active frame
  const activeKf = (keyframes.length > 0 && currentIndex >= 0 && currentIndex < keyframes.length)
    ? keyframes[currentIndex]
    : null;

  const activeFrameId = activeKf ? activeKf.frame_idx : item.frameId;
  const activeKeyframeId = activeKf ? activeKf.keyframe_id : (item.keyframe_id || item.id || '');
  const activeTimestamp = activeKf
    ? activeKf.timestamp
    : (item.timestamp !== '' && item.timestamp !== undefined ? item.timestamp : 0);
  const activePts = activeKf
    ? activeKf.pts_time
    : (item.pts_time !== undefined ? item.pts_time : (activeTimestamp / 1000.0));

  const activeImageSrc = useMemo(() => {
    if (activeKf && activeKf.image_url) {
      const u = activeKf.image_url;
      return u.startsWith('http') ? u : `${API_BASE_URL}${u.startsWith('/') ? '' : '/'}${u}`;
    }
    const u = item.thumbnailUrl || '';
    return u.startsWith('http') || u === '' ? u : `${API_BASE_URL}${u.startsWith('/') ? '' : '/'}${u}`;
  }, [activeKf, item.thumbnailUrl]);

  useEffect(() => {
    setImgState('loading');
    setRetryCount(0);
    setCurSrc(activeImageSrc);
    if (keyframes.length > 0) {
      preloadNeighbors(keyframes, currentIndex);
    }
  }, [activeImageSrc, currentIndex, keyframes, preloadNeighbors]);

  // Navigation
  const goPrev = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  }, [currentIndex]);

  const goNext = useCallback(() => {
    if (currentIndex < keyframes.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  }, [currentIndex, keyframes.length]);

  // Duplicate submission key
  const submissionKey = `${videoId}_${activeFrameId}_${activeTimestamp}`;
  const isAlreadySubmitted = window.__dresSubmittedSet.has(submissionKey);

  // Submit Search / KIS to DRES
  const handleSubmit = useCallback(async () => {
    if (submitStatus === 'submitting') return;
    setSubmitStatus('submitting');
    setSubmitResult(null);

    const payload = {
      video_id: String(videoId),
      timestamp: parseInt(activeTimestamp, 10) || 0,
      timestamp_ms: parseInt(activeTimestamp, 10) || 0,
      start: parseInt(activeTimestamp, 10) || 0,
      end: parseInt(activeTimestamp, 10) || 0,
      mode: 'search',
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/dres/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok || data.status === false) {
        setSubmitStatus('error');
        setSubmitResult({
          status: false,
          description: data.detail || data.description || 'Submission failed',
        });
      } else {
        setSubmitStatus('success');
        window.__dresSubmittedSet.add(submissionKey);
        setSubmitResult({
          status: true,
          description: data.description || 'Submitted',
        });
      }
    } catch (err) {
      console.error('DRES submit error:', err);
      setSubmitStatus('error');
      setSubmitResult({
        status: false,
        description: `Submission failed: ${err.message}`,
      });
    }
  }, [submitStatus, videoId, activeTimestamp, submissionKey]);

  // Submit VQA to DRES
  const handleVqaSubmit = useCallback(async () => {
    if (vqaSubmitStatus === 'submitting' || !vqaAnswer.trim()) return;
    setVqaSubmitStatus('submitting');
    setVqaSubmitResult(null);

    const payload = {
      video_id: String(videoId),
      timestamp_ms: parseInt(activeTimestamp, 10) || 0,
      timestamp: parseInt(activeTimestamp, 10) || 0,
      answer: vqaAnswer.trim(),
      mode: 'vqa',
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/dres/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok || data.status === false) {
        setVqaSubmitStatus('error');
        setVqaSubmitResult({
          status: false,
          description: data.detail || data.description || 'Submission failed',
        });
      } else {
        setVqaSubmitStatus('success');
        setVqaSubmitResult({
          status: true,
          description: data.description || 'Submitted',
        });
      }
    } catch (err) {
      console.error('VQA submit error:', err);
      setVqaSubmitStatus('error');
      setVqaSubmitResult({
        status: false,
        description: `Submission failed: ${err.message}`,
      });
    }
  }, [vqaSubmitStatus, vqaAnswer, videoId, activeTimestamp]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      const isInput = e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA');
      if (isInput) {
        if (e.key === 'Escape') {
          e.preventDefault();
          onClose();
        }
        return;
      }

      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        goPrev();
      } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        e.preventDefault();
        goNext();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (mode === 'semantic') {
          handleSubmit();
        } else if (mode === 'vqa') {
          handleVqaSubmit();
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goPrev, goNext, handleSubmit, handleVqaSubmit, mode, onClose]);

  const handleLoad = () => setImgState('loaded');
  const handleError = () => {
    if (retryCount < 2) {
      const next = retryCount + 1;
      setRetryCount(next);
      const sep = curSrc.includes('?') ? '&' : '?';
      setCurSrc(activeImageSrc + sep + '__retry=' + next + '_' + Date.now());
      setImgState('loading');
    } else {
      setImgState('error');
    }
  };

  const manualRetry = (e) => {
    e.stopPropagation();
    setRetryCount(0);
    setImgState('loading');
    const sep = activeImageSrc.includes('?') ? '&' : '?';
    setCurSrc(activeImageSrc + sep + '__retry=m_' + Date.now());
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/75 backdrop-blur-md animate-fade-in"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl overflow-hidden border border-slate-700/30 flex flex-col max-h-[95vh]">
        
        {/* Header Bar */}
        <div className="bg-slate-900 px-4 py-2.5 flex items-center justify-between text-white border-b border-slate-800">
          <div className="flex items-center space-x-2.5">
            <span className="bg-sky-500/20 text-sky-400 border border-sky-500/30 px-2 py-0.5 rounded text-xs font-mono font-bold">
              {videoId}
            </span>
            <span className="text-slate-400 text-xs font-mono">
              {activeKeyframeId}
            </span>
            {isAlreadySubmitted && mode === 'semantic' && (
              <span className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-[10px] px-2 py-0.5 rounded-full font-bold flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Đã nộp
              </span>
            )}
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex items-center bg-slate-800 p-0.5 rounded-lg border border-slate-700">
            <button
              onClick={() => setMode('semantic')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                mode === 'semantic'
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Semantic
            </button>
            <button
              onClick={() => setMode('vqa')}
              className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                mode === 'vqa'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              VQA
            </button>
          </div>

          <div className="flex items-center space-x-3">
            <span className="text-[11px] text-slate-400 hidden sm:inline-block font-mono">
              Phím tắt: <kbd className="bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">←/A</kbd> <kbd className="bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">→/D</kbd> <kbd className="bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">Enter</kbd> <kbd className="bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">Esc</kbd>
            </span>
            <button
              onClick={onClose}
              className="w-7 h-7 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded-full flex items-center justify-center transition-colors"
              title="Đóng (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="overflow-y-auto p-4 sm:p-5 space-y-4">
          
          {/* Large Image View (Selected Frame) */}
          <div className="aspect-video bg-slate-950 rounded-xl overflow-hidden relative border border-slate-800 shadow-inner flex items-center justify-center">
            {imgState === 'loaded' && (
              <img
                src={curSrc}
                alt={activeKeyframeId}
                className="w-full h-full object-contain select-none"
                onLoad={handleLoad}
                onError={handleError}
              />
            )}
            {imgState === 'loading' && (
              <>
                <div className="skeleton absolute inset-0 opacity-50" />
                <img src={curSrc} alt="" className="w-0 h-0 opacity-0" onLoad={handleLoad} onError={handleError} />
                <div className="absolute inset-0 flex items-center justify-center text-sky-400 gap-2 text-xs font-mono">
                  <div className="w-4 h-4 border-2 border-sky-400 border-t-transparent rounded-full animate-spin"></div>
                  <span>Đang tải frame {activeFrameId}...</span>
                </div>
              </>
            )}
            {imgState === 'error' && (
              <div className="w-full h-full flex flex-col items-center justify-center text-slate-300 bg-slate-900 p-4">
                <p className="text-xs font-semibold text-slate-300 mb-2">Không thể tải ảnh keyframe</p>
                <button onClick={manualRetry} className="px-3 py-1.5 bg-sky-600 hover:bg-sky-700 text-white text-xs font-bold rounded-lg shadow flex items-center gap-1">
                  <RefreshCw className="w-3.5 h-3.5" /> Tải lại
                </button>
              </div>
            )}
          </div>

          {/* Adjacent Navigation */}
          <div className="flex items-center justify-between p-2.5 bg-slate-100 rounded-xl border border-slate-200">
            <button
              onClick={goPrev}
              disabled={currentIndex <= 0 || loadingList}
              className={`px-4 py-2 rounded-lg font-semibold text-xs flex items-center gap-1.5 transition-all shadow-sm ${
                currentIndex <= 0 || loadingList
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 active:scale-95'
              }`}
              title="Frame trước đó (← hoặc A)"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous Frame</span>
              <kbd className="hidden sm:inline bg-slate-100 px-1 rounded text-[10px] text-slate-500 font-mono">← / A</kbd>
            </button>

            <div className="text-center font-mono text-xs text-slate-600 font-semibold px-2">
              {loadingList ? (
                <span className="text-slate-400 italic">Đang nạp danh sách keyframe...</span>
              ) : keyframes.length > 0 ? (
                <span>
                  Frame <strong className="text-slate-900">{currentIndex + 1}</strong> / {keyframes.length}
                </span>
              ) : (
                <span>Frame {activeFrameId}</span>
              )}
            </div>

            <button
              onClick={goNext}
              disabled={currentIndex >= keyframes.length - 1 || loadingList}
              className={`px-4 py-2 rounded-lg font-semibold text-xs flex items-center gap-1.5 transition-all shadow-sm ${
                currentIndex >= keyframes.length - 1 || loadingList
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 active:scale-95'
              }`}
              title="Frame kế tiếp (→ hoặc D)"
            >
              <span>Next Frame</span>
              <kbd className="hidden sm:inline bg-slate-100 px-1 rounded text-[10px] text-slate-500 font-mono">→ / D</kbd>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* MODE: SEMANTIC */}
          {mode === 'semantic' && (
            <>
              {/* Metadata Info Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="text-[11px] font-medium text-slate-500 mb-1 flex items-center gap-1">
                    <Film className="w-3.5 h-3.5 text-slate-400" />
                    <span>Video ID</span>
                  </div>
                  <div className="font-mono font-bold text-slate-900 text-sm truncate" title={videoId}>
                    {videoId}
                  </div>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="text-[11px] font-medium text-slate-500 mb-1 flex items-center gap-1">
                    <Hash className="w-3.5 h-3.5 text-slate-400" />
                    <span>Frame ID</span>
                  </div>
                  <div className="font-mono font-bold text-sky-700 text-sm">
                    {activeFrameId}
                  </div>
                </div>

                <div className="p-3 bg-sky-50/70 rounded-xl border border-sky-200/80">
                  <div className="text-[11px] font-semibold text-sky-800 mb-1 flex items-center gap-1">
                    <Clock className="w-3.5 h-3.5 text-sky-600" />
                    <span>Timestamp (ms)</span>
                  </div>
                  <div className="font-mono font-bold text-sky-950 text-sm">
                    {activeTimestamp} ms
                  </div>
                  <div className="text-[10px] font-mono text-sky-600 font-medium truncate">
                    {formatMilliseconds(activeTimestamp)}
                  </div>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="text-[11px] font-medium text-slate-500 mb-1 flex items-center gap-1">
                    <Tag className="w-3.5 h-3.5 text-slate-400" />
                    <span>Thứ hạng (Rank)</span>
                  </div>
                  <div className="font-mono font-bold text-slate-900 text-sm">
                    #{item.rank ?? 1}
                  </div>
                </div>
              </div>

              {/* Submit Action Button */}
              <div>
                <button
                  onClick={handleSubmit}
                  disabled={submitStatus === 'submitting'}
                  className={`w-full py-3.5 px-6 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2.5 shadow-lg transition-all active:scale-[0.99] ${
                    submitStatus === 'submitting'
                      ? 'bg-sky-700 cursor-wait'
                      : isAlreadySubmitted
                        ? 'bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-500 hover:to-teal-600 shadow-emerald-500/25'
                        : 'bg-gradient-to-r from-sky-600 to-blue-700 hover:from-sky-500 hover:to-blue-600 shadow-sky-500/30 ring-2 ring-sky-400/30'
                  }`}
                >
                  {submitStatus === 'submitting' ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Đang nộp bài lên DRES...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>{isAlreadySubmitted ? 'Nộp lại lên DRES (Resubmit to DRES)' : 'Submit to DRES'}</span>
                      <kbd className="bg-black/30 text-white/90 px-2 py-0.5 rounded text-xs font-mono font-semibold">
                        Enter
                      </kbd>
                    </>
                  )}
                </button>
              </div>

              {/* Submission Feedback Result */}
              {submitResult && (
                <div
                  className={`p-3 rounded-xl border flex items-center gap-2.5 transition-all animate-fade-in ${
                    submitResult.status
                      ? 'bg-emerald-50 border-emerald-300 text-emerald-900 font-bold text-xs'
                      : 'bg-rose-50 border-rose-300 text-rose-900 font-bold text-xs'
                  }`}
                >
                  {submitResult.status ? (
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                  )}
                  <span>{submitResult.status ? '✓ Submitted' : `Submission failed: ${submitResult.description}`}</span>
                </div>
              )}
            </>
          )}

          {/* MODE: VQA */}
          {mode === 'vqa' && (
            <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
              <label className="block text-xs font-bold text-slate-700">
                Answer:
              </label>
              
              <input
                type="text"
                value={vqaAnswer}
                onChange={(e) => setVqaAnswer(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleVqaSubmit();
                  }
                }}
                placeholder="Enter answer (e.g. TikTok, Nike, 7, Blue)..."
                className="w-full px-3.5 py-2.5 bg-white border border-slate-300 rounded-lg text-sm text-slate-900 font-medium placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500 shadow-sm"
                autoFocus
              />

              <button
                onClick={handleVqaSubmit}
                disabled={vqaSubmitStatus === 'submitting' || !vqaAnswer.trim()}
                className={`w-full py-3 px-6 rounded-xl font-bold text-sm text-white flex items-center justify-center gap-2 shadow-md transition-all active:scale-[0.99] ${
                  vqaSubmitStatus === 'submitting'
                    ? 'bg-purple-800 cursor-wait'
                    : !vqaAnswer.trim()
                      ? 'bg-slate-300 text-slate-500 cursor-not-allowed shadow-none'
                      : 'bg-gradient-to-r from-purple-600 to-indigo-700 hover:from-purple-500 hover:to-indigo-600 shadow-purple-500/25 ring-2 ring-purple-400/30'
                }`}
              >
                {vqaSubmitStatus === 'submitting' ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Đang nộp VQA...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Submit VQA</span>
                    <kbd className="bg-black/25 text-white/90 px-2 py-0.5 rounded text-xs font-mono font-semibold">
                      Enter
                    </kbd>
                  </>
                )}
              </button>

              {/* VQA Feedback Result */}
              {vqaSubmitResult && (
                <div
                  className={`p-3 rounded-xl border flex items-center gap-2.5 transition-all animate-fade-in ${
                    vqaSubmitResult.status
                      ? 'bg-emerald-50 border-emerald-300 text-emerald-900 font-bold text-xs'
                      : 'bg-rose-50 border-rose-300 text-rose-900 font-bold text-xs'
                  }`}
                >
                  {vqaSubmitResult.status ? (
                    <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                  )}
                  <span>{vqaSubmitResult.status ? '✓ Submitted' : `Submission failed: ${vqaSubmitResult.description}`}</span>
                </div>
              )}
            </div>
          )}

        </div>

        {/* Footer */}
        <div className="bg-slate-50 px-4 py-3 border-t border-slate-200 flex items-center justify-between">
          <div className="text-[11px] text-slate-500 font-mono">
            {mode === 'semantic' ? `${activeKeyframeId} • ${activeTimestamp} ms` : 'VQA Mode'}
          </div>
          <button
            onClick={onClose}
            className="py-1.5 px-4 bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs rounded-lg shadow transition-colors"
          >
            Đóng lại (Esc)
          </button>
        </div>

      </div>
    </div>
  );
}
