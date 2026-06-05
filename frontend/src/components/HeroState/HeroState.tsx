import React from 'react';
import { FileText, Share2, MessageSquare } from 'lucide-react';

export const HeroState: React.FC = () => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-slate-950">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-3xl font-extrabold text-white tracking-tight">
            Welcome to Graph<span className="text-violet-400">Atlas</span>
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            A next-generation knowledge discovery engine.
          </p>
        </div>

        <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 text-left space-y-6 shadow-2xl">
          <div className="flex items-center gap-4">
            <div className="bg-blue-500/20 p-3 rounded-lg text-blue-400">
              <FileText className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-slate-200 font-medium">Upload your documents</h3>
              <p className="text-slate-400 text-sm">PDF, DOCX, or plain text</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="bg-violet-500/20 p-3 rounded-lg text-violet-400">
              <Share2 className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-slate-200 font-medium">Explore hidden relationships</h3>
              <p className="text-slate-400 text-sm">Automatic knowledge graph extraction</p>
            </div>
          </div>

        </div>

        <div className="pt-4">
          <p className="text-xs text-slate-500">
            Click the <span className="text-slate-300 font-semibold">⊕ Add Document</span> button in the sidebar to get started.
          </p>
        </div>
      </div>
    </div>
  );
};
