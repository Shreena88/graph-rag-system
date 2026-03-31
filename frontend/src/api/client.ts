import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export const uploadDocument = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post<{ doc_id: string; status: string }>("/documents/upload", form);
};

export const getDocumentStatus = (docId: string) =>
  api.get<{ doc_id: string; status: string; error?: string }>(`/documents/${docId}/status`);

export const getEntities = (docId?: string) =>
  api.get<{ name: string; type: string; mentions: number }[]>("/graph/entities", {
    params: { doc_id: docId },
  });

export const getEdges = (docId?: string) =>
  api.get<{ source: string; target: string; weight: number }[]>("/graph/edges", {
    params: { doc_id: docId },
  });

export const getNeighbors = (entity: string, depth = 2) =>
  api.get<{ name: string; type: string }[]>("/graph/neighbors", {
    params: { entity, depth },
  });
