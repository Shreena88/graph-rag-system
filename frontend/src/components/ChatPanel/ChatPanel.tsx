import React, { useState, useRef, useEffect } from "react";
import { Send, User, Bot, Loader2 } from "lucide-react";
import { HeroState } from "../HeroState/HeroState";

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
  const [activeTab, setActiveTab] = useState("Chat");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const question = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
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
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const token = line.slice(6);
          if (token === "[DONE]") break;
          answer += token;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: "assistant", content: answer };
            return updated;
          });
        }
      }
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error: could not reach the server." }]);
    }
    setLoading(false);
  };

  if (docIds.length === 0) {
    return <HeroState />;
  }

  return (
    <div className="flex flex-col h-full bg-slate-950">
      <div className="flex items-center gap-6 px-6 py-4 border-b border-white/10 bg-white/5 backdrop-blur-xl">
        {['Chat'].map(tab => (
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

      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
        {messages.map((m, i) => (
          <div key={i} className={`flex gap-4 max-w-[85%] ${m.role === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
              m.role === 'user' ? 'bg-violet-600' : 'bg-slate-800 border border-white/10'
            }`}>
              {m.role === 'user' ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-violet-400" />}
            </div>
            <div className={`p-4 rounded-2xl shadow-xl backdrop-blur-md ${
              m.role === 'user' 
                ? 'bg-violet-600 text-white rounded-tr-sm' 
                : 'bg-slate-900 border border-white/10 text-slate-200 rounded-tl-sm'
            }`}>
              <div className="text-sm whitespace-pre-wrap leading-relaxed">{m.content}</div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex gap-4 max-w-[85%]">
            <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-slate-800 border border-white/10">
              <Bot className="w-5 h-5 text-violet-400" />
            </div>
            <div className="p-4 rounded-2xl bg-slate-900 border border-white/10 text-slate-400 text-sm flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-violet-400" />
              GraphAtlas is thinking<span className="animate-pulse">...</span>
              <span className="w-1 h-4 bg-violet-400 animate-pulse ml-1"></span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="p-6 bg-slate-950 border-t border-white/10">
        <div className="relative flex items-center">
          <input
            className="w-full bg-slate-900 border border-white/10 rounded-xl py-4 pl-4 pr-14 text-sm text-slate-200 focus:outline-none focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/50 shadow-inner placeholder:text-slate-500"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask a question about your documents..."
          />
          <button 
            className="absolute right-2 p-2 bg-violet-600 hover:bg-violet-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg transition-colors"
            onClick={sendMessage} 
            disabled={loading || !input.trim()}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
