import React, { useRef, useState } from "react";
import { uploadDocument, getDocumentStatus } from "../../api/client";

interface Props {
  onUploaded: (docId: string) => void;
}

const pollStatus = async (docId: string, maxAttempts = 30): Promise<string> => {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 3000));
    try {
      const { data } = await getDocumentStatus(docId);
      if (data.status === "indexed" || data.status === "failed") return data.status;
    } catch {
      // 404 means server reloaded and lost the doc — treat as failed
      return "failed";
    }
  }
  return "timeout";
};

export const DocumentUpload: React.FC<Props> = ({ onUploaded }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string>("");
  const [statusType, setStatusType] = useState<"" | "success" | "error">("");

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("Uploading...");
    setStatusType("");
    try {
      const { data } = await uploadDocument(file);
      const docId = data.doc_id;
      setStatus("Processing... building graph");
      const finalStatus = await pollStatus(docId);
      if (finalStatus === "indexed") {
        setStatus(`✓ ${file.name} indexed`);
        setStatusType("success");
        onUploaded(docId);
      } else {
        setStatus(`✗ Indexing ${finalStatus === "failed" ? "failed" : "timed out"}`);
        setStatusType("error");
      }
    } catch {
      setStatus("✗ Upload failed");
      setStatusType("error");
    }
    // reset input so same file can be re-uploaded
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="upload-box">
      <h3>Documents</h3>
      <button className="btn-upload" onClick={() => inputRef.current?.click()}>
        + Upload Document
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.docx"
        style={{ display: "none" }}
        onChange={handleFile}
      />
      {status && (
        <div className={`upload-status ${statusType}`}>{status}</div>
      )}
    </div>
  );
};
