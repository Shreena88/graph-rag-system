import React, { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  docIds: string[];
}

export const ChatPanel: React.FC<Props> = ({ docIds }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput("");
    setMessages((prev: Message[]) => [...prev, { role: "user", content: question }]);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/query/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, doc_ids: docIds, stream: true }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let answer = "";
      setMessages((prev: Message[]) => [...prev, { role: "assistant", content: "" }]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const token = line.slice(6);
          if (token === "[DONE]") break;
          answer += token;
          setMessages((prev: Message[]) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: "assistant", content: answer };
            return updated;
          });
        }
      }
    } catch {
      setMessages((prev: Message[]) => [...prev, { role: "assistant", content: "Error: could not reach the server." }]);
    }
    setLoading(false);
  };

  return (
    <>
      <div className="chat-header">Chat</div>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div style={{ color: "#334155", fontSize: 13, textAlign: "center", marginTop: 40 }}>
            Upload a document and ask a question
          </div>
        )}
        {messages.map((m: Message, i: number) => (
          <div key={i} className={`message ${m.role}`}>{m.content}</div>
        ))}
        {loading && <div className="thinking">● Thinking...</div>}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <input
          className="chat-input"
          value={input}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && sendMessage()}
          placeholder="Ask a question about your documents..."
        />
        <button className="btn-send" onClick={sendMessage} disabled={loading}>
          Send
        </button>
      </div>
    </>
  );
};
