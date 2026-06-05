import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="py-2 text-center text-xs text-slate-500 bg-slate-950 border-t border-white/5 z-50">
      Powered by <span className="text-slate-400">Neo4j • FAISS • Groq • Ollama • React</span>
    </footer>
  );
};
