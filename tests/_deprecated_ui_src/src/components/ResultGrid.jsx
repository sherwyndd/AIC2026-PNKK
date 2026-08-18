import React from 'react';
import { Play, Plus, Check, AlertTriangle, Search } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8090';

const MATCH_TYPE_STYLES = {
  text: { label: 'Text', cls: 'bg-sky-500/90 text-white border-sky-600/40' },
  asr: { label: 'ASR', cls: 'bg-purple-500/90 text-white border-purple-600/40' },
  ocr: { label: 'OCR', cls: 'bg-amber-500/90 text-white border-amber-600/40' },
  object: { label: 'Object', cls: 'bg-emerald-500/90 text-white border-emerald-600/40' },
};

function resolveImageUrl(rawUrl) {
  if (!rawUrl) return '';
  if (rawUrl.startsWith('http')) return rawUrl;
  return `${API_BASE_URL}${rawUrl.startsWith('/') ? '' : '/'}${rawUrl}`;
}

function getMatchStyle(type) {
  const t = (type || 'text').toLowerCase();
  return MATCH_TYPE_STYLES[t] || MATCH_TYPE_STYLES.text;
}

export default function ResultGrid({
  results,
  error,
  onSelectResult,
  isSearching,
  onAddToRanking,
  rankingKeyframeIds = new Set(),
}) {
  const isEmpty = !results || results.length === 0;

  if (isSearching) {
    return (
      <div className="flex flex-col items-center justify-center py-24 px-6 text-slate-500">
        <div className="relative w-16 h-16 mb-5">
          <div className="absolute inset-0 rounded-full border-4 border-sky-200"></div>
          <div className="absolute inset-0 rounded-full border-4 border-sky-600 border-t-transparent animate-spin"></div>
        </div>
        <p className="text-sm font-semibold text-slate-700 mb-1">Đang tìm kiếm...</p>
        <p className="text-xs text-slate-400">Backend đang xử lý truy vấn của bạn</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-24 px-6 text-center">
        <div className="w-16 h-16 rounded-full bg-rose-50 flex items-center justify-center mb-4 text-rose-500">
          <AlertTriangle className="w-8 h-8" />
        </div>
        <p className="text-sm font-bold text-rose-700 mb-2">Tìm kiếm không thành công</p>
        <p className="text-xs text-slate-500 max-w-sm">{error}</p>
      </div>
    );
  }

  if (isEmpty) {
    return (
      <div className="flex flex-col items-center justify-center py-24 px-6 text-center">
        <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-4 text-slate-400">
          <Search className="w-8 h-8" />
        </div>
        <p className="text-sm font-semibold text-slate-600 mb-2">Chưa có kết quả</p>
        <p className="text-xs text-slate-400 max-w-sm">Nhập truy vấn vào Search Panel bên trái và nhấn &quot;Thực hiện tìm kiếm&quot; để xem kết quả.</p>
      </div>
    );
  }

  return (
    <div className="p-4 sm:p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="font-bold text-sm text-slate-700">Kết quả tìm kiếm</h3>
          <span className="text-xs font-mono font-bold text-slate-500 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-full">
            {results.length}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-7 gap-3 sm:gap-4">
        {results.map((item, idx) => {
          const imgUrl = resolveImageUrl(item.thumbnailUrl);
          const matchStyle = getMatchStyle(item.matchedType);
          const kfId = item.keyframe_id || item.id;
          const isRanked = rankingKeyframeIds.has(kfId);

          return (
            <div
              key={kfId || idx}
              className="group relative bg-white rounded-xl overflow-hidden border border-slate-200 shadow-sm hover:shadow-lg hover:border-sky-300 transition-all duration-200 flex flex-col"
            >
              <div
                className="relative aspect-video bg-slate-100 overflow-hidden cursor-pointer"
                onClick={() => onSelectResult(item)}
              >
                <img
                  src={imgUrl}
                  alt={item.keyframe_id || item.id}
                  loading="lazy"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  onError={(e) => {
                    e.currentTarget.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120"><rect width="200" height="120" fill="%23e2e8f0"/><text x="100" y="62" text-anchor="middle" fill="%2394a3b8" font-size="12" font-family="monospace">No image</text></svg>';
                  }}
                />

                <div className="absolute top-1.5 left-1.5 flex flex-col gap-1">
                  <span className="inline-flex items-center justify-center min-w-[28px] h-6 px-1.5 rounded-md bg-slate-900/80 backdrop-blur-sm text-white text-[10px] font-bold font-mono shadow-sm">
                    #{item.rank ?? idx + 1}
                  </span>
                  <span className={`inline-flex items-center justify-center h-5 px-1.5 rounded-md border text-[9px] font-bold shadow-sm ${matchStyle.cls}`}>
                    {matchStyle.label}
                  </span>
                </div>

                <div className="absolute inset-0 bg-sky-600/0 group-hover:bg-sky-600/10 transition-colors flex items-center justify-center">
                  <div className="w-10 h-10 rounded-full bg-white/0 group-hover:bg-white/90 text-white group-hover:text-sky-700 flex items-center justify-center shadow-sm scale-75 group-hover:scale-100 opacity-0 group-hover:opacity-100 transition-all duration-200">
                    <Play className="w-4 h-4 fill-current" />
                  </div>
                </div>
              </div>

              <div className="p-2 flex flex-col gap-1.5 flex-1">
                <div className="text-[10px] font-mono font-bold text-slate-800 truncate" title={item.keyframe_id || item.videoId}>
                  {item.keyframe_id || item.id}
                </div>
                <div className="text-[9px] font-mono text-slate-500 truncate" title={item.videoId}>
                  {item.videoId} • F{item.frameId}
                </div>

                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (onAddToRanking && !isRanked) onAddToRanking(item);
                  }}
                  disabled={isRanked}
                  title={isRanked ? 'Đã có trong danh sách Ranking' : 'Thêm Keyframe này vào tab Ranking'}
                  className={`mt-1 w-full py-1.5 px-2 rounded-lg text-[10px] font-bold flex items-center justify-center gap-1 transition-all ${
                    isRanked
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200 cursor-default'
                      : 'bg-sky-50 hover:bg-sky-600 text-sky-700 hover:text-white border border-sky-200 hover:border-sky-600 active:scale-95'
                  }`}
                >
                  {isRanked ? (
                    <>
                      <Check className="w-3 h-3" />
                      <span>Đã thêm vào Ranking</span>
                    </>
                  ) : (
                    <>
                      <Plus className="w-3 h-3" />
                      <span>Add To Ranking</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
