import React, { useEffect, useState, useCallback } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import type { Core, EventObject, ElementDefinition } from "cytoscape";
import { getEntities, getEdges } from "../../api/client";

interface Props {
  docId?: string;
}

const TYPE_COLORS: Record<string, string> = {
  PERSON: "#4a90d9",
  ORG: "#e67e22",
  LOCATION: "#27ae60",
  DATE: "#8e44ad",
  CONCEPT: "#c0392b",
  LAW: "#16a085",
  PRODUCT: "#d35400",
  DEFAULT: "#4a90d9",
};

export const GraphViewer: React.FC<Props> = ({ docId }) => {
  const [elements, setElements] = useState<ElementDefinition[]>([]);
  const [loading, setLoading] = useState(false);
  const [cy, setCy] = useState<Core | null>(null);

  const loadGraph = useCallback(async (id?: string) => {
    if (!id) return;
    setLoading(true);
    try {
      const [entRes, edgeRes] = await Promise.all([
        getEntities(id),
        getEdges(id),
      ]);

      const maxMentions = Math.max(...entRes.data.map((e) => e.mentions), 1);

      const nodes: ElementDefinition[] = entRes.data.map((e) => ({
        data: {
          id: e.name,
          label: e.name,
          type: e.type,
          mentions: e.mentions,
          size: 20 + (e.mentions / maxMentions) * 60,
          color: TYPE_COLORS[e.type] ?? TYPE_COLORS.DEFAULT,
        },
      }));

      // Build a set of valid node IDs so we can filter dangling edges
      const nodeIds = new Set(nodes.map((n) => n.data.id as string));

      const edges: ElementDefinition[] = edgeRes.data
        .filter((e) => e.source !== e.target)
        .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
        .map((e, i) => ({
          data: {
            id: `e${i}`,
            source: e.source,
            target: e.target,
            weight: e.weight,
          },
        }));

      // Only keep nodes that appear in at least one edge
      const connectedIds = new Set(edges.flatMap((e) => [e.data.source as string, e.data.target as string]));
      const connectedNodes = nodes.filter((n) => connectedIds.has(n.data.id as string));

      setElements([...connectedNodes, ...edges]);
    } catch {
      // graph unavailable
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadGraph(docId);
  }, [docId, loadGraph]);

  // fit graph after layout completes
  useEffect(() => {
    if (cy && elements.length > 0) {
      setTimeout(() => {
        try { cy.fit(undefined, 40); } catch { /* cy may have been destroyed */ }
      }, 800);
    }
  }, [cy, elements]);

  return (
    <>
      <div className="graph-header">
        Knowledge Graph
        {loading && <span style={{ fontSize: 11, color: "#475569", marginLeft: 8 }}>loading...</span>}
        {elements.length > 0 && !loading && (
          <span style={{ fontSize: 11, color: "#475569", marginLeft: 8 }}>
            {elements.filter((e) => !e.data.source).length} nodes
          </span>
        )}
      </div>
      <div className="graph-container">
        {elements.length === 0 && !loading ? (
          <div className="graph-empty">
            <div className="graph-empty-icon">◎</div>
            <div>Graph will appear after upload</div>
          </div>
        ) : elements.filter((e) => e.data.source).length === 0 && !loading ? (
          <div className="graph-empty">
            <div className="graph-empty-icon">◎</div>
            <div>No relationships found in document</div>
          </div>
        ) : (
          <CytoscapeComponent
            elements={elements}
            style={{ width: "100%", height: "100%", background: "#0f172a" }}
            layout={{
              name: "cose",
              animate: true,
              animationDuration: 600,
              nodeRepulsion: () => 8000,
              idealEdgeLength: () => 80,
              edgeElasticity: () => 100,
              gravity: 0.25,
              numIter: 1000,
              fit: true,
              padding: 40,
            } as any}
            cy={(c: Core) => {
              setCy(c);
              c.on("tap", "node", (e: EventObject) => {
                const node = e.target;
                node.neighborhood().addClass("highlighted");
              });
              c.on("tap", (e: EventObject) => {
                if (e.target === c) c.elements().removeClass("highlighted");
              });
            }}
            stylesheet={[
              {
                selector: "node",
                style: {
                  width: "data(size)",
                  height: "data(size)",
                  "background-color": "data(color)",
                  label: "data(label)",
                  color: "#1e293b",
                  "font-size": 9,
                  "font-weight": "bold",
                  "text-valign": "center",
                  "text-halign": "center",
                  "text-wrap": "wrap",
                  "text-max-width": "80px",
                  "text-background-color": "white",
                  "text-background-opacity": 0.75,
                  "text-background-padding": "2px",
                  "text-background-shape": "roundrectangle",
                  "border-width": 1.5,
                  "border-color": "white",
                  "border-opacity": 0.4,
                },
              },
              {
                selector: "edge",
                style: {
                  "line-color": "#94a3b8",
                  width: 0.8,
                  opacity: 0.5,
                  "curve-style": "bezier",
                },
              },
              {
                selector: ".highlighted",
                style: {
                  "border-color": "#f59e0b",
                  "border-width": 3,
                  opacity: 1,
                },
              },
              {
                selector: "node:selected",
                style: {
                  "border-color": "#f59e0b",
                  "border-width": 3,
                },
              },
            ]}
          />
        )}
      </div>
    </>
  );
};
