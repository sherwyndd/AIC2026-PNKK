import React from 'react';
import { StickyNote, Trophy, History, ChevronUp, ChevronDown, Trash2, FileText, Search, RotateCcw } from 'lucide-react';

const TAB_ORDER = [
  { key: 'note', label: 'Note', Icon: StickyNote },
  { key: 'ranking', label: 'Ranking', Icon: Trophy },
  { key: 'history', label: 'Query History', Icon: History },
];

function extractFileName(item) {
  if (item && item.keyframe_id) return item.keyframe_id;
  if (item && item.videoId && item.frameId) return `${item.videoId}_F${item.frameId}`;
  if (item && item.id) return item.id;
  return 'Unknown';
}

export default function ClipboardPanel({
  activeTab = 'note',
  onTabChange,
  noteText = '',
  onChangeNoteText,
  rankingList = [],
  onMoveRankingUp,
  onMoveRankingDown,
  onRemoveRankingItem,
  onSelectRankingItem,
  queryHistory = [],
  onReplayQuery,
  onClearHistory,
}) {
  return (
    <aside className="w-80 md:w-96 bg-slate-50 border-l border-slate-200 flex flex-col h-[calc(100vh-3.5rem)] sticky top-14">
      <div className="p-3 border-b border-slate-200 flex items-center justify-between bg-white">
        <h2 className="font-bold text-sm tracking-wide uppercase text-slate-700 flex items-center gap-1.5">
          <FileText className="w-4 h-4 text-slate-500" />
          Clipboard
        </h2>
        {activeTab === 'ranking' && (
          <span className="text-xs font-mono font-bold text-sky-700 bg-sky-50 border border-sky-200 px-2 py-0.5 rounded-full">
            {rankingList.length} item
          </span>
        )}
        {activeTab === 'history' && queryHistory.length > 0 && (
          <button
            type="button"
            onClick={onClearHistory}
            title="Xóa toàn bộ lịch sử"
            className="text-[10px] font-semibold text-slate-500 hover:text-rose-600 flex items-center gap-1 px-2 py-1 rounded hover:bg-rose-50 transition-colors"
          >
            <Trash2 className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>

      <div className="flex items-stretch bg-white border-b border-slate-200 px-1">
        {TAB_ORDER.map(({ key, label, Icon }) => {
          const isActive = activeTab === key;
          return (
            <button
              key={key}
              type="button"
              onClick={() => onTabChange && onTabChange(key)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 px-2 text-xs font-bold border-b-2 transition-all ${
                isActive
                  ? 'text-sky-700 border-sky-600 bg-sky-50/60'
                  : 'text-slate-500 border-transparent hover:text-slate-700 hover:bg-slate-50'
              }`}
              title={label}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span className="truncate">{label}</span>
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        {activeTab === 'note' && (
          <div className="flex-1 flex flex-col p-3 gap-2">
            <label className="text-[11px] font-bold text-slate-600 flex items-center gap-1">
              <StickyNote className="w-3 h-3 text-amber-500" />
              Ghi chú cá nhân
            </label>
            <textarea
              value={noteText}
              onChange={(e) => onChangeNoteText && onChangeNoteText(e.target.value)}
              placeholder="Ghi chú, suy nghĩ, tóm tắt kết quả tìm kiếm của bạn..."
              rows={20}
              className="flex-1 w-full resize-none bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-xs text-slate-800 leading-relaxed placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500/30 focus:border-sky-500 shadow-sm font-mono"
            />
            <div className="text-[10px] font-mono text-slate-400 text-right">
              {noteText.length} ký tự
            </div>
          </div>
        )}

        {activeTab === 'ranking' && (
          <div className="flex-1 flex flex-col min-h-0">
            {rankingList.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center px-5 text-center py-16">
                <div className="w-14 h-14 rounded-full bg-amber-50 border border-amber-100 flex items-center justify-center mb-3 text-amber-500">
                  <Trophy className="w-6 h-6" />
                </div>
                <p className="text-xs font-bold text-slate-600 mb-1">Danh sách Ranking đang trống</p>
                <p className="text-[10px] text-slate-400 max-w-xs">
                  Nhấn nút <span className="font-semibold text-sky-600">&quot;Add To Ranking&quot;</span> trên mỗi Keyframe trong Result Grid để thêm vào đây.
                </p>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto divide-y divide-slate-100 py-1">
                {rankingList.map((item, idx) => {
                  const fileName = extractFileName(item);
                  const isFirst = idx === 0;
                  const isLast = idx === rankingList.length - 1;
                  const rank = idx + 1;

                  let rankBadgeCls = 'bg-slate-100 text-slate-600 border-slate-200';
                  if (rank === 1) rankBadgeCls = 'bg-gradient-to-br from-amber-400 to-amber-500 text-white border-amber-500 shadow-amber-200 shadow-sm';
                  else if (rank === 2) rankBadgeCls = 'bg-gradient-to-br from-slate-300 to-slate-400 text-white border-slate-400';
                  else if (rank === 3) rankBadgeCls = 'bg-gradient-to-br from-orange-300 to-orange-400 text-white border-orange-400';

                  return (
                    <div
                      key={item.keyframe_id || item.id || idx}
                      className="group flex items-center gap-2 px-2 py-1.5 hover:bg-sky-50/70 transition-colors"
                    >
                      <div
                        className={`shrink-0 w-8 h-8 rounded-lg border flex items-center justify-center text-[11px] font-bold font-mono ${rankBadgeCls}`}
                        title={`Rank ${rank}`}
                      >
                        {rank}
                      </div>

                      <button
                        type="button"
                        onClick={() => onSelectRankingItem && onSelectRankingItem(item)}
                        className="flex-1 min-w-0 text-left px-1 py-1 rounded hover:bg-white/80 transition-colors"
                        title={`Click để xem Keyframe ${fileName}`}
                      >
                        <div className="text-xs font-mono font-bold text-slate-800 truncate">
                          {fileName}
                        </div>
                        <div className="text-[9px] font-mono text-slate-500 truncate">
                          {item.videoId || ''} {item.frameId ? `• F${item.frameId}` : ''}
                        </div>
                      </button>

                      <div className="shrink-0 flex items-center gap-0.5 opacity-80 group-hover:opacity-100 transition-opacity">
                        <button
                          type="button"
                          onClick={() => onMoveRankingUp && onMoveRankingUp(idx)}
                          disabled={isFirst}
                          title={isFirst ? 'Đã ở vị trí đầu tiên' : 'Lên 1 bậc'}
                          className={`w-6 h-6 rounded-md flex items-center justify-center transition-all ${
                            isFirst
                              ? 'bg-slate-100 text-slate-300 cursor-not-allowed'
                              : 'bg-sky-50 hover:bg-sky-600 text-sky-700 hover:text-white active:scale-95'
                          }`}
                        >
                          <ChevronUp className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onMoveRankingDown && onMoveRankingDown(idx)}
                          disabled={isLast}
                          title={isLast ? 'Đã ở vị trí cuối cùng' : 'Xuống 1 bậc'}
                          className={`w-6 h-6 rounded-md flex items-center justify-center transition-all ${
                            isLast
                              ? 'bg-slate-100 text-slate-300 cursor-not-allowed'
                              : 'bg-sky-50 hover:bg-sky-600 text-sky-700 hover:text-white active:scale-95'
                          }`}
                        >
                          <ChevronDown className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onRemoveRankingItem && onRemoveRankingItem(idx)}
                          title="Xóa khỏi danh sách Ranking"
                          className="w-6 h-6 rounded-md flex items-center justify-center bg-rose-50 hover:bg-rose-500 text-rose-500 hover:text-white transition-all active:scale-95 ml-0.5"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            {rankingList.length > 0 && (
              <div className="px-3 py-2 bg-white border-t border-slate-200 text-[10px] font-mono text-slate-500 flex items-center justify-between">
                <span>Tổng: <strong className="text-slate-700">{rankingList.length}</strong> keyframe</span>
                <span>Click dòng → xem ảnh</span>
              </div>
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="flex-1 flex flex-col min-h-0">
            {queryHistory.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center px-5 text-center py-16">
                <div className="w-14 h-14 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center mb-3 text-slate-400">
                  <History className="w-6 h-6" />
                </div>
                <p className="text-xs font-bold text-slate-600 mb-1">Chưa có lịch sử tìm kiếm</p>
                <p className="text-[10px] text-slate-400 max-w-xs">
                  Các truy vấn bạn đã thực hiện sẽ xuất hiện ở đây.
                </p>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto divide-y divide-slate-100 py-1">
                {queryHistory.map((entry, idx) => (
                  <div
                    key={entry.id || idx}
                    className="flex items-start gap-2 px-3 py-2 hover:bg-sky-50/60 transition-colors group"
                  >
                    <div className="shrink-0 mt-0.5 w-6 h-6 rounded-md bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-500">
                      <Search className="w-3 h-3" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold text-slate-800 leading-snug line-clamp-2" title={entry.query}>
                        {entry.query}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[9px] font-mono text-slate-400">
                          {entry.timestamp || ''}
                        </span>
                        {typeof entry.resultCount === 'number' && (
                          <span className="text-[9px] font-mono font-bold text-sky-600 bg-sky-50 px-1.5 py-0.5 rounded border border-sky-100">
                            {entry.resultCount} kết quả
                          </span>
                        )}
                      </div>
                    </div>
                    {onReplayQuery && (
                      <button
                        type="button"
                        onClick={() => onReplayQuery(entry)}
                        title="Thực hiện lại truy vấn này"
                        className="shrink-0 opacity-0 group-hover:opacity-100 transition-opacity w-7 h-7 rounded-md bg-sky-50 hover:bg-sky-600 text-sky-600 hover:text-white flex items-center justify-center active:scale-95"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
