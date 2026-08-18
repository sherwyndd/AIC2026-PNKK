import React from 'react';
import { Volume2, FileText, Image as ImageIcon } from 'lucide-react';

const SEARCH_TYPES = [
  { id: 'text', label: 'Text', icon: 'Tr', placeholder: 'Mô tả nội dung video bằng văn bản...' },
  { id: 'asr', label: 'ASR', icon: Volume2, placeholder: 'Nhập lời thoại hoặc âm thanh cần tìm...' },
  { id: 'ocr', label: 'OCR', icon: FileText, placeholder: 'Nhập văn bản xuất hiện trên hình ảnh/khung hình...' },
  { id: 'object', label: 'Object', icon: ImageIcon, placeholder: 'Nhập object + số lượng (vd: 3 person 2 car)...' },
];

export default function SearchPanel({ panel, index, onUpdate }) {
  const { type, query, enabled } = panel;

  const handleTypeChange = (newType) => {
    onUpdate({ ...panel, type: newType });
  };

  const handleQueryChange = (e) => {
    onUpdate({ ...panel, query: e.target.value });
  };

  const handleToggle = () => {
    onUpdate({ ...panel, enabled: !enabled });
  };

  const activeTypeObj = SEARCH_TYPES.find(t => t.id === type) || SEARCH_TYPES[0];

  return (
    <div className={`rounded-xl border transition-all duration-200 shadow-sm ${
      enabled 
        ? 'bg-white border-slate-200 hover:border-sky-300' 
        : 'bg-slate-50 border-slate-200 opacity-60'
    }`}>
      {/* Panel Header: 4 Search Type Icons */}
      <div className="p-3 pb-2 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center space-x-1.5 bg-slate-100/80 p-1 rounded-lg">
          {SEARCH_TYPES.map((t) => {
            const isActive = type === t.id;
            const IconComp = t.icon;

            return (
              <button
                key={t.id}
                type="button"
                onClick={() => handleTypeChange(t.id)}
                title={`Tìm kiếm theo ${t.label}`}
                className={`flex items-center justify-center w-8 h-8 rounded-md text-xs font-bold transition-all ${
                  isActive
                    ? 'bg-sky-600 text-white shadow-sm scale-105'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                }`}
              >
                {typeof IconComp === 'string' ? (
                  <span>{IconComp}</span>
                ) : (
                  <IconComp className="w-4 h-4" />
                )}
              </button>
            );
          })}
        </div>

        <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
          Panel #{index + 1} • <span className="text-sky-600">{activeTypeObj.label}</span>
        </span>
      </div>

      {/* Query Textarea */}
      <div className="p-3">
        <textarea
          value={query}
          onChange={handleQueryChange}
          disabled={!enabled}
          placeholder={activeTypeObj.placeholder}
          rows={3}
          className="w-full text-sm bg-transparent border-0 focus:ring-0 p-0 resize-none placeholder-slate-400 text-slate-800 disabled:cursor-not-allowed"
        />
      </div>

      {/* Panel Footer: Toggle Switch */}
      <div className="px-3 py-2 bg-slate-50/60 border-t border-slate-100 rounded-b-xl flex items-center justify-end">
        <button
          type="button"
          onClick={handleToggle}
          role="switch"
          aria-checked={enabled}
          className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
            enabled ? 'bg-sky-600' : 'bg-slate-300'
          }`}
        >
          <span
            className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
              enabled ? 'translate-x-4' : 'translate-x-0'
            }`}
          />
        </button>
      </div>
    </div>
  );
}
