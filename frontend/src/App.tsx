import React, { useState, Component } from "react";
import { DocumentUpload } from "./components/DocumentUpload/DocumentUpload";
import { ChatPanel } from "./components/ChatPanel/ChatPanel";
import { GraphViewer } from "./components/GraphViewer/GraphViewer";
import { Navbar } from "./components/Navbar/Navbar";
import { Footer } from "./components/Footer/Footer";
import { DocumentCard } from "./components/DocumentCard/DocumentCard";
import { deleteDocument } from "./api/client";
import { Sparkles } from "lucide-react";

class GraphErrorBoundary extends Component<{ children: React.ReactNode }, { error: boolean }> {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  render() {
    if (this.state.error) {
      return (
        <div className="p-4 text-slate-400 text-sm">
          Graph failed to render. Try uploading again.
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [docIds, setDocIds] = useState<string[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | undefined>();
  const [docs, setDocs] = useState<any[]>([]);
  const [nodesCount, setNodesCount] = useState(0);
  const [edgesCount, setEdgesCount] = useState(0);
  const [rightPanelWidth, setRightPanelWidth] = useState(450);
  const [summarizeSignal, setSummarizeSignal] = useState(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = rightPanelWidth;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX;
      const newWidth = Math.max(300, Math.min(1000, startWidth - deltaX));
      setRightPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "default";
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    document.body.style.cursor = "col-resize";
  };

  const handleUploaded = (docId: string, file: File) => {
    setDocIds((prev) => [...prev, docId]);
    setActiveDocId(docId);
    setDocs((prev) => [...prev, { id: docId, name: file.name, size: file.size, status: 'Indexed' }]);
  };

  const handleDeleteDocument = async (id: string) => {
    try {
      await deleteDocument(id);
    } catch (e) {
      console.warn("Error deleting document from backend:", e);
    }
    setDocs((prev) => prev.filter((d) => d.id !== id));
    setDocIds((prev) => prev.filter((docId) => docId !== id));
    if (activeDocId === id) {
      const remaining = docs.filter((d) => d.id !== id);
      setActiveDocId(remaining.length > 0 ? remaining[remaining.length - 1].id : undefined);
    }
  };

  const handleSummarizeAll = () => {
    setSummarizeSignal((prev) => prev + 1);
  };

  return (
    <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans">
      <Navbar />
      
      <main className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-72 min-w-72 bg-white/5 backdrop-blur-xl border-r border-white/10 flex flex-col p-4 gap-6">
          <div className="mb-2">
            <DocumentUpload onUploaded={handleUploaded} />
          </div>

          <div className="flex-1 overflow-y-auto flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Documents ({docs.length})</h3>
            </div>

            {docs.length > 0 && (
              <button
                onClick={handleSummarizeAll}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white text-xs font-medium rounded-xl shadow-lg shadow-violet-500/20 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <Sparkles className="w-3.5 h-3.5 text-amber-300" />
                Summarize All {docs.length > 1 ? `(${docs.length} PDFs)` : ''}
              </button>
            )}

            <div className="flex flex-col gap-3 mt-1">
              {docs.map(doc => (
                <DocumentCard 
                  key={doc.id} 
                  doc={doc} 
                  isActive={doc.id === activeDocId} 
                  onClick={() => setActiveDocId(doc.id)} 
                  onDelete={handleDeleteDocument}
                />
              ))}
              {docs.length === 0 && (
                <div className="text-sm text-slate-500 italic">No documents yet.</div>
              )}
            </div>
          </div>

        </aside>

        {/* Middle Panel */}
        <div className="flex-1 flex flex-col bg-slate-950 min-w-[300px]">
          <ChatPanel docIds={docIds} summarizeSignal={summarizeSignal} />
        </div>


        {/* Resizer */}
        <div 
          className="w-1 cursor-col-resize bg-white/10 hover:bg-violet-500 active:bg-violet-500 transition-colors z-10 shrink-0"
          onMouseDown={handleMouseDown}
          title="Drag to resize"
        />

        {/* Right Panel */}
        <div 
          style={{ width: rightPanelWidth }}
          className="bg-slate-900 flex flex-col shrink-0 min-w-[300px]"
        >
          {/* Top Stats */}
          <div className="grid grid-cols-3 gap-2 p-4 border-b border-white/10 bg-white/5 backdrop-blur-xl">
             <div className="bg-slate-950/50 p-3 rounded-lg border border-white/5">
                <div className="text-xs text-slate-400 mb-1">Docs</div>
                <div className="text-xl font-bold text-slate-200">{docs.length}</div>
             </div>
             <div className="bg-slate-950/50 p-3 rounded-lg border border-white/5">
                <div className="text-xs text-slate-400 mb-1">Nodes</div>
                <div className="text-xl font-bold text-slate-200">{nodesCount}</div>
             </div>
             <div className="bg-slate-950/50 p-3 rounded-lg border border-white/5">
                <div className="text-xs text-slate-400 mb-1">Edges</div>
                <div className="text-xl font-bold text-slate-200">{edgesCount}</div>
             </div>
          </div>
          
          <div className="flex-1 relative">
            <GraphErrorBoundary>
              <GraphViewer docId={activeDocId} onGraphStats={(n, e) => { setNodesCount(n); setEdgesCount(e); }} />
            </GraphErrorBoundary>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
