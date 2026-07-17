import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AnswerSupport, Conversation, Message, RagSettings, SourceChunk } from "../../api";
import { queryModel } from "../../api";
import { useUiStore } from "../../store";
import { Composer } from "./Composer";
import { MessageActions } from "./MessageActions";

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
  onLoadOlder?: () => void;
};

type ChatMessage = Message & {
  id?: string;
  sources?: SourceChunk[];
  support?: AnswerSupport | null;
  status?: "complete" | "streaming" | "error" | "stopped";
};

export function ChatPanel({ selectedModel, modelReady, settings, conversation, selectedConversationId, onConversationSelected, onLoadOlder }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [retrievalScope, setRetrievalScope] = useState("medium");
  const [responseEffort, setResponseEffort] = useState("balanced");
  const [responsePhase, setResponsePhase] = useState("");
  const setSelectedSources = useUiStore(state => state.setSelectedSources);
  const setSelectedSupport = useUiStore(state => state.setSelectedSupport);
  const setRightPanel = useUiStore(state => state.setRightPanel);
  const feedRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const followOutputRef = useRef(true);

  useEffect(() => {
    if (followOutputRef.current && typeof endRef.current?.scrollIntoView === "function") {
      endRef.current.scrollIntoView({ behavior: isTyping ? "auto" : "smooth" });
    }
  }, [isTyping, messages]);

  useEffect(() => {
    if (!conversation?.messages) return;
    setMessages(conversation.messages.map(message => ({
      id: message.id,
      role: message.role,
      content: message.content,
      sources: message.sources || [],
      support: (message.meta?.support as AnswerSupport | undefined) || null,
      status: "complete",
    })) as ChatMessage[]);
  }, [conversation?.id, conversation?.messages]);

  async function runRequest(userMsg: string, baseMessages: ChatMessage[]) {
    if (isTyping || !selectedModel || !modelReady || !settings) return;
    const historyPayload = baseMessages.map(message => ({ role: message.role, content: message.content }));
    const assistantDraftId = `draft-${Date.now()}`;
    const controller = new AbortController();
    abortRef.current = controller;
    followOutputRef.current = true;
    setIsTyping(true);
    setResponsePhase("Connecting to retrieval...");
    setMessages([...baseMessages, { role: "user", content: userMsg }, { id: assistantDraftId, role: "assistant", content: "", status: "streaming" }]);

    try {
      const body = await queryModel(
        userMsg,
        selectedModel,
        historyPayload,
        settings,
        selectedConversationId,
        retrievalScope,
        responseEffort,
        controller.signal,
      );
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
      }, setResponsePhase);
      setMessages(prev => prev.map(message => message.id === assistantDraftId ? { ...message, status: "complete" } : message));
      if (completedConversationId) onConversationSelected?.(completedConversationId);
    } catch (error) {
      const stopped = controller.signal.aborted || (error instanceof DOMException && error.name === "AbortError");
      if (stopped) {
        setMessages(prev => prev.map(message => message.id === assistantDraftId ? { ...message, status: "stopped" } : message));
        return;
      }
      const message = error instanceof Error ? error.message : "Error connecting to local service.";
      setMessages(prev => {
        const next = [...prev];
        const target = next.findIndex(item => item.id === assistantDraftId);
        if (target !== -1) next[target] = { id: assistantDraftId, role: "assistant", content: message, status: "error" };
        return next;
      });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setIsTyping(false);
      setResponsePhase("");
    }
  }

  function handleSend() {
    const userMsg = input.trim();
    if (!userMsg) return;
    setInput("");
    void runRequest(userMsg, messages);
  }

  function regenerate(messageIndex: number) {
    const userIndex = findPreviousUserMessage(messages, messageIndex);
    if (userIndex === -1) return;
    const prompt = messages[userIndex].content;
    void runRequest(prompt, messages.slice(0, userIndex));
  }

  function openSources(sources: SourceChunk[]) {
    setSelectedSources(sources);
    setRightPanel("sources");
  }

  const handleFeedScroll = () => {
    const feed = feedRef.current;
    if (!feed) return;
    followOutputRef.current = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 96;
  };

  return (
    <section className="chat-shell">
      <div className="message-feed" ref={feedRef} onScroll={handleFeedScroll}>
        {conversation?.has_more && onLoadOlder && (
          <button type="button" className="load-older" onClick={onLoadOlder}>Load older messages</button>
        )}
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>Search your documents</h2>
            <p>Import files, connect to your external llama.cpp server, and review the cited sources for each response.</p>
          </div>
        )}
        {messages.map((message, index) => {
          const parsed = message.role === "assistant" ? parseThinking(message.content) : { thinking: "", response: message.content };
          const response = parsed.response || (!message.content.includes("</think>") ? message.content : "");
          return (
            <article key={message.id || `${message.role}-${index}`} className={`message ${message.role}`}>
              <div className="message-body">
                {response ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ href, children }) => href?.startsWith("#source-")
                        ? (
                          <button
                            type="button"
                            className="inline-citation"
                            onClick={() => {
                              const sourceId = href.slice("#source-".length);
                              const source = message.sources?.find(item => item.source_id === sourceId);
                              if (source) openSources([source]);
                            }}
                          >
                            {children}
                          </button>
                        )
                        : <a href={href}>{children}</a>,
                    }}
                  >
                    {renderSourceTags(response)}
                  </ReactMarkdown>
                ) : <span className="subtle">{phaseLabel(responsePhase)}</span>}
                {message.role === "assistant" && (parsed.thinking || message.sources?.length || message.support) ? (
                  <details className="message-inspector">
                    <summary>Answer details</summary>
                    <p>{message.sources?.length ? `Generated from ${message.sources.length} retrieved source${message.sources.length === 1 ? "" : "s"}.` : "Generated without retrieved sources."}</p>
                  </details>
                ) : null}
                {message.role === "assistant" && parsed.thinking ? (
                  <details className="thinking-trace">
                    <summary>Thinking trace</summary>
                    <pre>{parsed.thinking}</pre>
                  </details>
                ) : null}
                {message.role === "assistant" && message.content && message.status !== "streaming" && (
                  <MessageActions
                    content={message.content}
                    sources={message.sources}
                    support={message.support}
                    isError={message.status === "error"}
                    onOpenSources={() => openSources(message.sources || [])}
                    onOpenSupport={() => {
                      setSelectedSupport(message.support || null);
                      setRightPanel("support");
                    }}
                    onRegenerate={() => regenerate(index)}
                  />
                )}
              </div>
            </article>
          );
        })}
        <div ref={endRef} />
      </div>
      <Composer
        value={input}
        onChange={setInput}
        onSubmit={handleSend}
        onStop={() => abortRef.current?.abort()}
        isRunning={isTyping}
        disabled={isTyping || !selectedModel || !modelReady || !settings}
        placeholder={!selectedModel ? "Select the configured external server." : !modelReady ? "Connect to the external llama.cpp server first." : "Search, compare, summarize..."}
        retrievalScope={retrievalScope}
        responseEffort={responseEffort}
        onRetrievalScopeChange={setRetrievalScope}
        onResponseEffortChange={setResponseEffort}
      />
    </section>
  );
}

async function consumeQueryStream(
  body: ReadableStream<Uint8Array>,
  onSources: (sources: SourceChunk[]) => void,
  onChunk: (chunk: string) => void,
  onMessageSources: (sources: SourceChunk[]) => void,
  onMeta: (meta: Record<string, unknown>) => void,
  onPhase: (phase: string) => void,
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
      const result = handleSsePacket(packet, sources, onSources, onChunk, onMessageSources, onMeta, onPhase);
      if (result.conversationId) conversationId = result.conversationId;
      boundary = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    const result = handleSsePacket(buffer, sources, onSources, onChunk, onMessageSources, onMeta, onPhase);
    if (result.conversationId) conversationId = result.conversationId;
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
  onPhase: (phase: string) => void,
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
  } else if (eventType === "phase") {
    onPhase(typeof payload.phase === "string" ? payload.phase : "answering");
  } else if (eventType === "error") {
    throw new Error(typeof payload.message === "string" ? payload.message : "Query stream failed.");
  }
  return {};
}

function renderSourceTags(content: string) {
  return content.replace(/\[\[src:(S\d+)\]\]/g, "[$1](#source-$1)");
}

function phaseLabel(phase: string) {
  if (phase === "Connecting to retrieval...") return phase;
  if (phase === "drafting") return "Drafting an answer...";
  if (phase === "refining") return "Refining the answer...";
  if (phase === "answering") return "Writing the answer...";
  return "Retrieving relevant context...";
}

function findPreviousUserMessage(messages: ChatMessage[], startIndex: number) {
  for (let index = startIndex - 1; index >= 0; index -= 1) {
    if (messages[index].role === "user") return index;
  }
  return -1;
}
