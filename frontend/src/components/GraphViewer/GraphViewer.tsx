import React, { useEffect, useState, useCallback, useRef } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import cytoscape from "cytoscape";
import { getEdges, getEntities } from "../../api/client";
import { ZoomIn, ZoomOut, Maximize } from "lucide-react";

interface Props {
  docId?: string;
  onGraphStats?: (nodes: number, edges: number) => void;
}

const STYLESHEET: any = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "text-valign": "center",
      "text-halign": "center",
      "font-size": "10px",
      "background-color": "data(color)",
      color: "#f8fafc",
      "text-outline-width": 1,
      "text-outline-color": "#1e293b",
      width: "data(size)",
      height: "data(size)",
    },
  },
  {
    selector: "edge",
    style: {
      width: 1,
      "line-color": "#475569",
      "target-arrow-color": "#475569",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      opacity: 0.6,
    },
  },
];

const TYPE_COLORS: Record<string, string> = {
  Document: "#10b981",
  Chunk: "#f59e0b",
  PERSON: "#3b82f6",
  ORG: "#8b5cf6",
  GPE: "#ec4899",
  LOC: "#06b6d4",
  DATE: "#eab308",
  EVENT: "#f43f5e",
};
const FALLBACK_PALETTE = [
  "#14b8a6", // teal
  "#84cc16", // lime
  "#d946ef", // fuchsia
  "#ef4444", // red
  "#0ea5e9", // sky
  "#8b5cf6", // violet
  "#f97316", // orange
  "#64748b", // slate
];
const DEFAULT_COLOR = "#94a3b8";

export const GraphViewer: React.FC<Props> = ({ docId, onGraphStats }) => {
  const [elements, setElements] = useState<cytoscape.ElementDefinition[]>([]);
  const [activeTab, setActiveTab] = useState("Graph");
  const cyRef = useRef<cytoscape.Core | null>(null);

  const loadGraph = useCallback(async (id?: string) => {
    if (!id) {
      setElements([]);
      onGraphStats?.(0, 0);
      return;
    }
    try {
      const [entRes, edgeRes] = await Promise.all([
        getEntities(id),
        getEdges(id),
      ]);

      const nodeIds = new Set(entRes.data.map((e: any) => e.name));
      
      const validEdgesRaw = edgeRes.data.filter((r: any) => nodeIds.has(r.source) && nodeIds.has(r.target));
      const connectedNodeIds = new Set();
      validEdgesRaw.forEach((r: any) => {
        connectedNodeIds.add(r.source);
        connectedNodeIds.add(r.target);
      });

      const dynamicColors: Record<string, string> = {};

      const nodes: cytoscape.ElementDefinition[] = entRes.data
        .filter((e: any) => connectedNodeIds.has(e.name))
        .map((e: any) => {
        let typeColor = TYPE_COLORS[e.type];
        if (!typeColor) {
          if (!dynamicColors[e.type]) {
            dynamicColors[e.type] = FALLBACK_PALETTE[Object.keys(dynamicColors).length % FALLBACK_PALETTE.length];
          }
          typeColor = dynamicColors[e.type];
        }

        return {
          data: {
            id: e.name,
            label: e.name.length > 15 ? e.name.slice(0, 15) + "..." : e.name,
            type: e.type,
            size: Math.min(30 + e.mentions * 2, 60),
            color: typeColor,
          },
        };
      });

      const edges: cytoscape.ElementDefinition[] = validEdgesRaw.map((r: any, i: number) => ({
        data: {
          id: `e${i}`,
          source: r.source,
          target: r.target,
          weight: r.weight,
        },
      }));

      setElements([...nodes, ...edges]);
      onGraphStats?.(nodes.length, edges.length);
    } catch {
      setElements([]);
      onGraphStats?.(0, 0);
    }
  }, []);

  useEffect(() => {
    loadGraph(docId);
  }, [docId, loadGraph]);

  const fitGraph = () => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 30);
    }
  };

  const zoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.2);
  const zoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.8);

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-bl-xl border-l border-white/5">
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/5 backdrop-blur-xl">
        <div className="flex items-center gap-6">
          {['Graph', 'Raw Data'].map(tab => (
            <button 
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-sm font-medium pb-4 -mb-4 border-b-2 transition-colors ${
                activeTab === tab 
                  ? 'border-violet-500 text-violet-400' 
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
        
        <div className="flex items-center gap-2">
           <button onClick={zoomIn} className="p-1.5 rounded bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors">
              <ZoomIn className="w-4 h-4" />
           </button>
           <button onClick={zoomOut} className="p-1.5 rounded bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors">
              <ZoomOut className="w-4 h-4" />
           </button>
           <button onClick={fitGraph} className="p-1.5 rounded bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors">
              <Maximize className="w-4 h-4" />
           </button>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden bg-slate-950/50">
        {!docId ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-slate-500">
            <div className="text-5xl opacity-30">◎</div>
            <div className="text-sm">Graph will appear after upload</div>
          </div>
        ) : elements.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-slate-500">
            <div className="text-5xl opacity-30 animate-pulse">◎</div>
            <div className="text-sm">Graph unavailable or empty</div>
          </div>
        ) : activeTab === 'Raw Data' ? (
          <div className="absolute inset-0 overflow-auto p-6 text-xs text-slate-300 font-mono whitespace-pre-wrap bg-slate-950">
            {JSON.stringify(elements.map(e => e.data), null, 2)}
          </div>
        ) : (
          <>
            <CytoscapeComponent
              elements={elements}
              stylesheet={STYLESHEET}
              style={{ width: "100%", height: "100%", position: "absolute", top: 0, left: 0 }}
              layout={{ 
                name: "cose", 
                padding: 50, 
                animate: false,
                nodeRepulsion: () => 4000000,
                idealEdgeLength: () => 100,
                nodeOverlap: 50,
                gravity: 80
              } as any}
              cy={(cy) => {
                cyRef.current = cy;
                cy.on("layoutstop", () => cy.fit(undefined, 30));
              }}
            />
            
            <div className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-md border border-white/10 rounded-lg p-3 text-xs flex flex-col gap-2 shadow-xl max-h-64 overflow-y-auto">
              <div className="font-semibold text-slate-400 mb-1">Legend</div>
              {Array.from(new Set(elements.filter(e => e.data?.type).map(e => e.data.type as string))).map(type => (
                <div key={type} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: elements.find(e => e.data.type === type)?.data.color as string || DEFAULT_COLOR }}></div>
                  <span className="text-slate-300">{type}</span>
                </div>
              ))}
            </div>
            

          </>
        )}
      </div>
    </div>
  );
};

