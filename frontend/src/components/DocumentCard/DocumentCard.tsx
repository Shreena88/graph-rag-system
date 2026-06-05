import React from 'react';
import { FileText, CheckCircle2 } from 'lucide-react';

interface DocumentCardProps {
  doc: {
    id: string;
    name: string;
    size?: number;
    status: string;
  };
  isActive: boolean;
  onClick: () => void;
}

export const DocumentCard: React.FC<DocumentCardProps> = ({ doc, isActive, onClick }) => {
  // Format bytes to MB
  const sizeStr = doc.size ? `${(doc.size / (1024 * 1024)).toFixed(1)} MB` : '1.3 MB';
  const entityCount = 34; // Mocked for now

  return (
    <div 
      onClick={onClick}
      className={`relative p-3 rounded-xl border cursor-pointer transition-all ${
        isActive 
          ? 'bg-violet-500/10 border-violet-500/50 shadow-[0_0_15px_rgba(139,92,246,0.1)]' 
          : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 overflow-hidden">
          <FileText className={`w-4 h-4 shrink-0 ${isActive ? 'text-violet-400' : 'text-slate-400'}`} />
          <span className={`text-sm font-medium truncate ${isActive ? 'text-violet-200' : 'text-slate-200'}`}>
            {doc.name}
          </span>
        </div>
      </div>
      
      <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-1 text-emerald-400">
          <CheckCircle2 className="w-3 h-3" />
          <span>{doc.status}</span>
        </div>
        <div className="flex items-center gap-2">
          <span>{sizeStr}</span>
          <span>•</span>
          <span>{entityCount} Entities</span>
        </div>
      </div>
      
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-violet-500 rounded-r-full" />
      )}
    </div>
  );
};
