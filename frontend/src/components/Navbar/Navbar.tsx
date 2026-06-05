import React, { useState } from 'react';
import { Network, BookOpen, GitBranch, Moon, Sun } from 'lucide-react';

export const Navbar: React.FC = () => {
  const [isLight, setIsLight] = useState(false);

  const toggleTheme = () => {
    const nextIsLight = !isLight;
    setIsLight(nextIsLight);
    if (nextIsLight) {
      document.documentElement.classList.add('light-theme');
    } else {
      document.documentElement.classList.remove('light-theme');
    }
  };
  return (
    <nav className="flex items-center justify-between px-6 py-3 bg-slate-950/80 backdrop-blur-xl border-b border-white/10 z-50">
      <div className="flex items-center gap-2">
        <div className="bg-violet-500/20 p-2 rounded-lg border border-violet-500/30">
          <Network className="w-5 h-5 text-violet-400" />
        </div>
        <div className="text-xl font-bold tracking-tight text-slate-100">
          Graph<span className="text-violet-400">Atlas</span>
        </div>
      </div>

      <div className="flex items-center gap-4 text-sm font-medium text-slate-300">

        <a
          href="https://github.com/Shreena88/graph-rag-system"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-3 py-2 rounded-md hover:bg-white/5 hover:text-white transition-colors"
        >
          <GitBranch className="w-4 h-4" />
          Github
        </a>

        <div className="w-px h-6 bg-white/10 mx-2"></div>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-md hover:bg-white/5 text-slate-400 hover:text-amber-400 transition-colors"
          title="Toggle theme"
        >
          {isLight ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
        </button>
      </div>
    </nav>
  );
};
