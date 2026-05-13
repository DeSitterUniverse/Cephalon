import { Send } from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message, RagSettings, SourceChunk } from "../../api";
import { queryModel } from "../../api";
import { useUiStore } from "../../store";

function parseThinking(content: string): { thinking: string; response: string } {
  const closeTag = "</think>";
  const closeIdx = content.indexOf(closeTag);
  if (closeIdx === -1) {
    const openTag = "<think>";
    const openIdx = content.indexOf(openTag);
    if (openIdx !== -1) return { thinking: content.substring(openIdx + openTag.length).trim(), response: "" };
    return { thinking: "", response: content };
  }
  const openTag = "<think>";
  const openIdx = content.indexOf(openTag);
  const thinkStart = openIdx !== -1 ? openIdx + openTag.length : 0;
  return {
    thinking: content.substring(thinkStart, closeIdx).trim(),
    response: content.substring(closeIdx + closeTag.length).trim(),
  };
}

type Props = {
  selectedModel: string;
  settings?: RagSettings;
};

export function ChatPanel({ selectedModel, settings }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const setSelectedSources = useUiStore(state => state.setSelectedSources);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || isTyping || !selectedModel || !settings) return;

    const userMsg = input.trim();
    const historyPayload = messages.slice(-6);
    setInput("");
    setIsTyping(true);
    setMessages(prev => [...prev, { role: "user", content: userMsg }, { role: "assistant", content: "" }]);

    try {
      const body = await queryModel(userMsg, selectedModel, historyPayload, settings);
      await consumeQueryStream(body, setSelectedSources, chunk => {
        setMessages(prev => {
          const next = [...prev];
          const last = next.length - 1;
          next[last] = { ...next[last], content: next[last].content + chunk };
          return next;
        });
      }, meta => {
        if (meta?.no_answer) {
          setMessages(prev => {
            const next = [...prev];
            const last = next.length - 1;
            next[last] = {
              ...next[last],
              content: `${next[last].content}\n\n_Confidence is low. Closest retrieved matches are shown in Sources._`,
            };
            return next;
          });
        }
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error connecting to local service.";
      setMessages(prev => {
        const next = [...prev];
        const last = next.length - 1;
        if (next[last]?.role === "assistant") next[last] = { role: "assistant", content: message };
        return next;
      });
    } finally {
      setIsTyping(false);
    }
  }

  return (
    <section className="chat-shell">
      <div className="message-feed">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>Search your documents</h2>
            <p>Select a model, import files, and review the cited sources for each response.</p>
          </div>
        )}
        {messages.map((message, index) => {
          const parsed = message.role === "assistant" ? parseThinking(message.content) : { thinking: "", response: message.content };
          const response = parsed.response || (!message.content.includes("</think>") ? message.content : "");
          return (
            <article key={index} className={`message ${message.role}`}>
              <div className="message-role">{message.role === "assistant" ? "response" : "query"}</div>
              <div className="message-body">
                {parsed.thinking && <details className="thinking"><summary>Internal trace</summary><ReactMarkdown remarkPlugins={[remarkGfm]}>{parsed.thinking}</ReactMarkdown></details>}
                {response ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown> : <span className="subtle">Working...</span>}
              </div>
            </article>
          );
        })}
        <div ref={endRef} />
      </div>
      <form className="composer" onSubmit={handleSend}>
        <input value={input} onChange={event => setInput(event.target.value)} disabled={isTyping || !selectedModel} placeholder={selectedModel ? "Search, compare, summarize..." : "Select a local model to start."} />
        <button disabled={isTyping || !input.trim() || !selectedModel || !settings}><Send size={16} />Run</button>
      </form>
    </section>
  );
}

async function consumeQueryStream(
  body: ReadableStream<Uint8Array>,
  onSources: (sources: SourceChunk[]) => void,
  onChunk: (chunk: string) => void,
  onMeta: (meta: Record<string, unknown>) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  const sources: SourceChunk[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const packet = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handleSsePacket(packet, sources, onSources, onChunk, onMeta);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function handleSsePacket(
  packet: string,
  sources: SourceChunk[],
  onSources: (sources: SourceChunk[]) => void,
  onChunk: (chunk: string) => void,
  onMeta: (meta: Record<string, unknown>) => void,
) {
  const eventType = packet.split("\n").find(line => line.startsWith("event: "))?.slice(7).trim() || "message";
  const data = packet.split("\n").filter(line => line.startsWith("data: ")).map(line => line.slice(6)).join("\n");
  let payload: Record<string, unknown> = {};
  if (data) {
    try {
      payload = JSON.parse(data);
    } catch {
      payload = { text: data };
    }
  }
  if (eventType === "source") {
    sources.push(payload as SourceChunk);
    onSources([...sources]);
  } else if (eventType === "token") {
    onChunk(typeof payload.text === "string" ? payload.text : "");
  } else if (eventType === "answer_meta") {
    onMeta(payload);
  } else if (eventType === "error") {
    throw new Error(typeof payload.message === "string" ? payload.message : "Query stream failed.");
  }
}
