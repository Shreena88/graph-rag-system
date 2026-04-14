import React, { useState, Component } from "react";
import { DocumentUpload } from "./components/DocumentUpload/DocumentUpload";
import { ChatPanel } from "./components/ChatPanel/ChatPanel";
import { GraphViewer } from "./components/GraphViewer/GraphViewer";

class GraphErrorBoundary extends Component<{ children: React.ReactNode }, { error: boolean }> {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 20, color: "#64748b", fontSize: 13 }}>
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

  const handleUploaded = (docId: string) => {
    setDocIds((prev: string[]) => [...prev, docId]);
    setActiveDocId(docId);
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-title">Graph <span>RAG</span></div>
        <DocumentUpload onUploaded={handleUploaded} />
        <div className="doc-count">{docIds.length} document(s) indexed</div>
      </aside>
      <main className="main">
        <div className="chat-panel">
          <ChatPanel docIds={docIds} />
        </div>
        <div className="graph-panel">
          <GraphErrorBoundary>
            <GraphViewer docId={activeDocId} />
          </GraphErrorBoundary>
        </div>
      </main>
    </div>
  );
}
