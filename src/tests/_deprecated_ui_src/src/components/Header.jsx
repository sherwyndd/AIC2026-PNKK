import React, { useState } from 'react';
import { Settings, RotateCcw } from 'lucide-react';

export default function Header({ onReset, topK, onChangeTopK }) {
  const [showSettings, setShowSettings] = useState(false);

  const toggleSettings = () => setShowSettings(s => !s);

  return (
    <header className="relative h-14 bg-white border-b border-gray-200 px-4 flex items-center justify-between sticky top-0 z-20 shadow-sm">
      {/* Left: Logo K2PN */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-600 to-blue-500 flex items-center justify-center text-white font-bold text-lg shadow-sm">
          K
        </div>
        <div className="flex flex-col">
          <span className="font-extrabold text-xl tracking-tight text-slate-900 leading-none">
            K<span className="text-sky-600">2PN</span>
          </span>
          <span className="text-[10px] font-medium text-slate-400 tracking-wider">
            AI CHALLENGE 2026 SEARCH ENGINE
          </span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center space-x-2">
        {/* Settings (TopK) */}
        <div className="relative">
          <button
            onClick={toggleSettings}
            title="Cài đặt"
            className="p-2 rounded-lg text-slate-600 hover:text-sky-600 hover:bg-sky-50 active:scale-95 transition-all"
          >
            <Settings className="w-5 h-5" />
          </button>

          {showSettings && (
            <div className="absolute right-0 mt-2 w-44 bg-white border border-slate-200 rounded-md shadow-lg p-3 text-sm">
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-700">TopK</span>
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={topK}
                  onChange={(e) => onChangeTopK(Number(e.target.value) || 1)}
                  className="w-20 text-right border border-slate-100 rounded px-2 py-1 text-sm"
                />
              </label>
              <p className="text-xs text-slate-400 mt-2">Số kết quả hàng đầu hiển thị</p>
            </div>
          )}
        </div>

        {/* Refresh (Reset State) */}
        <button
          onClick={onReset}
          title="Làm mới (Reset lại tất cả khung tìm kiếm)"
          className="p-2 rounded-lg text-slate-600 hover:text-sky-600 hover:bg-sky-50 active:scale-95 transition-all"
        >
          <RotateCcw className="w-5 h-5" />
        </button>

        <div className="h-5 w-[1px] bg-slate-200 mx-1"></div>

        {/* User Badge "U" */}
        <div 
          title="Người dùng U"
          className="w-8 h-8 rounded-full bg-slate-100 border border-slate-300 flex items-center justify-center text-slate-700 font-semibold text-sm cursor-default"
        >
          U
        </div>
      </div>
    </header>
  );
}
