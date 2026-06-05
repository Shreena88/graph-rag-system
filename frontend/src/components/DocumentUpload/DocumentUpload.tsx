import React, { useRef, useState } from "react";
import { uploadDocument, getDocumentStatus } from "../../api/client";
import { PlusCircle, Loader2 } from "lucide-react";

interface Props {
  onUploaded: (docId: string, file: File) => void;
}

const pollStatus = async (docId: string, maxAttempts = 30): Promise<string> => {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 3000));
    try {
      const { data } = await getDocumentStatus(docId);
      if (data.status === "indexed" || data.status === "failed") return data.status;
    } catch {
      return "failed";
    }
  }
  return "timeout";
};

export const DocumentUpload: React.FC<Props> = ({ onUploaded }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string>("");
  const [statusType, setStatusType] = useState<"" | "success" | "error" | "loading">("");
  const [progress, setProgress] = useState(0);

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("Uploading...");
    setStatusType("loading");
    setProgress(10);
    try {
      const { data } = await uploadDocument(file);
      const docId = data.doc_id;
      
      setStatus("Parsing PDF...");
      setProgress(40);
      
      // Simulate progress for better UX
      setTimeout(() => { setStatus("Extracting entities..."); setProgress(70); }, 2000);
      setTimeout(() => { setStatus("Building graph..."); setProgress(90); }, 4000);

      const finalStatus = await pollStatus(docId);
      if (finalStatus === "indexed") {
        setStatus(`Indexed successfully`);
        setStatusType("success");
        setProgress(100);
        onUploaded(docId, file);
        setTimeout(() => { setStatus(""); setStatusType(""); }, 3000);
      } else {
        setStatus(`Indexing ${finalStatus === "failed" ? "failed" : "timed out"}`);
        setStatusType("error");
      }
    } catch {
      setStatus("Upload failed");
      setStatusType("error");
    }
    if (inputRef.current) inputRef.current.value = "";
  };

  const getProgressString = (pct: number) => {
    const full = Math.floor(pct / 10);
    const empty = 10 - full;
    return "█".repeat(full) + "░".repeat(empty);
  };

  return (
    <div className="flex flex-col gap-4">
      <button 
        className="flex items-center justify-center gap-2 w-full bg-violet-600 hover:bg-violet-500 text-white rounded-full py-3 px-4 font-semibold shadow-[0_0_20px_rgba(139,92,246,0.3)] hover:shadow-[0_0_25px_rgba(139,92,246,0.5)] transition-all transform hover:scale-105 active:scale-95"
        onClick={() => inputRef.current?.click()}
      >
        <PlusCircle className="w-5 h-5" />
        <span>Add Document</span>
      </button>
      
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.docx"
        style={{ display: "none" }}
        onChange={handleFile}
      />
      
      {status && (
        <div className={`p-4 rounded-xl text-sm ${
          statusType === "error" ? "bg-red-500/10 text-red-400 border border-red-500/20" : 
          statusType === "success" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : 
          "bg-white/5 border border-white/10 text-slate-300"
        }`}>
          <div className="flex items-center gap-2 mb-2 font-medium">
            {statusType === "loading" && <Loader2 className="w-4 h-4 animate-spin text-violet-400" />}
            {status}
          </div>
          {statusType === "loading" && (
            <div className="font-mono text-xs tracking-widest text-violet-400/70">
              {getProgressString(progress)}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
