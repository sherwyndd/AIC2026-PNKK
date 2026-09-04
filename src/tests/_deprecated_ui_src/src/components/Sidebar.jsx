import React from 'react';
import { Plus, Minus, Search } from 'lucide-react';
import SearchPanel from './SearchPanel';

export default function Sidebar({
  panels,
  onUpdatePanel,
  onAddPanel,
  onRemovePanel,
  onSearch,
  isSearching
}) {
  return (
    <aside className="w-80 md:w-96 bg-slate-50 border-r border-slate-200 flex flex-col h-[calc(100vh-3.5rem)] sticky top-14">
      {/* Sidebar Header */}
      <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-white">
        <h2 className="font-bold text-sm tracking-wide uppercase text-slate-700">
          Điều kiện truy vấn ({panels.length})
        </h2>
        <span className="text-xs text-slate-400 font-normal">
          {panels.filter(p => p.enabled).length} đang bật
        </span>
      </div>

      {/* Panels Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {panels.map((panel, idx) => (
          <SearchPanel
            key={panel.id}
            index={idx}
            panel={panel}
            onUpdate={onUpdatePanel}
          />
        ))}

        {/* Panel Controls (+ / - buttons) */}
        <div className="flex items-center space-x-2 pt-2">
          <button
            type="button"
            onClick={onAddPanel}
            title="Thêm một Search Panel mới vào danh sách"
            className="flex-1 py-2 px-3 bg-emerald-600 hover:bg-emerald-700 active:scale-98 text-white font-semibold text-xs rounded-xl flex items-center justify-center space-x-1.5 shadow-sm transition-all"
          >
            <Plus className="w-4 h-4" />
            <span>Thêm điều kiện (+)</span>
          </button>

          <button
            type="button"
            onClick={onRemovePanel}
            disabled={panels.length <= 1}
            title={panels.length <= 1 ? "Cần giữ tối thiểu 1 panel" : "Xóa panel cuối cùng"}
            className="py-2 px-3 bg-rose-600 hover:bg-rose-700 disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed active:scale-98 text-white font-semibold text-xs rounded-xl flex items-center justify-center space-x-1.5 shadow-sm transition-all"
          >
            <Minus className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Footer: Search Trigger Button */}
      <div className="p-4 bg-white border-t border-slate-200 shadow-lg">
        <button
          type="button"
          onClick={onSearch}
          disabled={isSearching}
          className="w-full py-3 px-4 bg-gradient-to-r from-sky-600 to-blue-600 hover:from-sky-500 hover:to-blue-500 text-white font-bold text-sm rounded-xl flex items-center justify-center space-x-2 shadow-md hover:shadow-lg active:scale-98 transition-all disabled:opacity-50"
        >
          <Search className="w-4 h-4" />
          <span>{isSearching ? 'Đang tìm kiếm...' : 'Thực hiện tìm kiếm'}</span>
        </button>
      </div>
    </aside>
  );
}
