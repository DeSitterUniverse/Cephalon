import { Send } from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AnswerSupport, Conversation, Message, RagSettings, SourceChunk } from "../../api";
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
  modelReady: boolean;
  settings?: RagSettings;
  conversation?: Conversation;
  selectedConversationId?: string | null;
  onConversationSelected?: (id: string) => void;
};

type ChatMessage = Message & {
  id?: string;
  sources?: SourceChunk[];
  support?: AnswerSupport | null;
};

export function ChatPanel({ selectedModel, modelReady, settings, conversation, selectedConversationId, onConversationSelected }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [reasoningMode, setReasoningMode] = useState("balanced");
  const setSelectedSources = useUiStore(state => state.setSelectedSources);
  const setSelectedSupport = useUiStore(state => state.setSelectedSupport);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    if (!conversation?.messages) return;
    setMessages(conversation.messages.map(message => ({
      id: message.id,
      role: message.role,
      content: message.content,
      sources: message.sources || [],
      support: (message.meta?.support as AnswerSupport | undefined) || null,
    })) as ChatMessage[]);
  }, [conversation?.id, conversation?.messages]);

  async function handleSend(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || isTyping || !selectedModel || !modelReady || !settings) return;

    const userMsg = input.trim();
    const historyPayload = messages.slice(-6).map(message => ({ role: message.role, content: message.content }));
    const assistantDraftId = `draft-${Date.now()}`;
    setInput("");
    setIsTyping(true);
    setMessages(prev => [...prev, { role: "user", content: userMsg }, { id: assistantDraftId, role: "assistant", content: "" }]);

    try {
      const body = await queryModel(userMsg, selectedModel, historyPayload, settings, selectedConversationId, reasoningMode);
      const completedConversationId = await consumeQueryStream(body, setSelectedSources, chunk => {
        setMessages(prev => {
          const next = [...prev];
          const target = next.findIndex(message => message.id === assistantDraftId);
          if (target === -1) return next;
          next[target] = { ...next[target], content: next[target].content + chunk };
          return next;
        });
      }, sources => {
        setMessages(prev => {
          const next = [...prev] as ChatMessage[];
          const target = next.findIndex(message => message.id === assistantDraftId);
          if (target === -1) return next;
          next[target] = { ...next[target], sources };
          return next;
        });
      }, meta => {
        if (meta?.support) {
          setMessages(prev => {
            const next = [...prev] as ChatMessage[];
            const target = next.findIndex(message => message.id === assistantDraftId);
            if (target === -1) return next;
            next[target] = { ...next[target], support: meta.support as AnswerSupport };
            return next;
          });
        }
      });
      if (completedConversationId) onConversationSelected?.(completedConversationId);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Error connecting to local service.";
      setMessages(prev => {
        const next = [...prev];
        const target = next.findIndex(item => item.id === assistantDraftId);
        if (target !== -1) next[target] = { id: assistantDraftId, role: "assistant", content: message };
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
            <p>Import files, load a local model, and review the cited sources for each response.</p>
          </div>
        )}
        {messages.map((message, index) => {
          const parsed = message.role === "assistant" ? parseThinking(message.content) : { thinking: "", response: message.content };
          const response = parsed.response || (!message.content.includes("</think>") ? message.content : "");
          return (
            <article key={index} className={`message ${message.role}`}>
              <div className="message-body">
                {response ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{renderSourceTags(response)}</ReactMarkdown> : <span className="subtle">Working...</span>}
                {message.role === "assistant" && (parsed.thinking || (message as ChatMessage).sources?.length || (message as ChatMessage).support) ? (
                  <details className="message-inspector">
                    <summary>Details</summary>
                    {parsed.thinking && <ReactMarkdown remarkPlugins={[remarkGfm]}>{parsed.thinking}</ReactMarkdown>}
                    {(message as ChatMessage).sources?.length ? (
                      <div className="message-actions">
                        <button className="source-link-row" type="button" onClick={() => setSelectedSources((message as ChatMessage).sources || [])}>
                          {(message as ChatMessage).sources?.length} sources
                        </button>
                        {(message as ChatMessage).support && (
                          <button className="source-link-row" type="button" onClick={() => setSelectedSupport((message as ChatMessage).support || null)}>
                            support: {(message as ChatMessage).support?.status}
                          </button>
                        )}
                      </div>
                    ) : null}
                  </details>
                ) : null}
              </div>
            </article>
          );
        })}
        <div ref={endRef} />
      </div>
      <form className="composer" onSubmit={handleSend}>
        <select
          className="reasoning-select"
          value={reasoningMode}
          onChange={event => setReasoningMode(event.target.value)}
          disabled={isTyping}
          title="Reasoning depth"
        >
          <option value="fast">Top 12 / Rerank 3 / 512</option>
          <option value="balanced">Current settings</option>
          <option value="deep">Top 28 / Rerank 6 / 1024</option>
        </select>
        <input
          value={input}
          onChange={event => setInput(event.target.value)}
          disabled={isTyping || !selectedModel || !modelReady}
          placeholder={!selectedModel ? "Select a local model." : !modelReady ? "Load the selected model first." : "Search, compare, summarize..."}
        />
        <button disabled={isTyping || !input.trim() || !selectedModel || !modelReady || !settings}><Send size={16} />Run</button>
      </form>
    </section>
  );
}

async function consumeQueryStream(
  body: ReadableStream<Uint8Array>,
  onSources: (sources: SourceChunk[]) => void,
  onChunk: (chunk: string) => void,
  onMessageSources: (sources: SourceChunk[]) => void,
  onMeta: (meta: Record<string, unknown>) => void,
): Promise<string | null> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  const sources: SourceChunk[] = [];
  let conversationId: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const packet = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const result = handleSsePacket(packet, sources, onSources, onChunk, onMessageSources, onMeta);
      if (result.conversationId) conversationId = result.conversationId;
      boundary = buffer.indexOf("\n\n");
    }
  }
  return conversationId;
}

function handleSsePacket(
  packet: string,
  sources: SourceChunk[],
  onSources: (sources: SourceChunk[]) => void,
  onChunk: (chunk: string) => void,
  onMessageSources: (sources: SourceChunk[]) => void,
  onMeta: (meta: Record<string, unknown>) => void,
): { conversationId?: string | null } {
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
    onMessageSources([...sources]);
  } else if (eventType === "token") {
    onChunk(typeof payload.text === "string" ? payload.text : "");
  } else if (eventType === "conversation") {
    return { conversationId: typeof payload.conversation_id === "string" ? payload.conversation_id : null };
  } else if (eventType === "answer_meta") {
    onMeta(payload);
  } else if (eventType === "error") {
    throw new Error(typeof payload.message === "string" ? payload.message : "Query stream failed.");
  }
  return {};
}

function renderSourceTags(content: string) {
  return content.replace(/\[\[src:(S\d+)\]\]/g, "`$1`");
}
