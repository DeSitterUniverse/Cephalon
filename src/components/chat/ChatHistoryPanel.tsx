import { FormEvent, useState } from "react";
import { Check, Pencil, Search, Trash2, X } from "lucide-react";
import type { Conversation } from "../../api";

type Props = {
  conversations: Conversation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => Promise<unknown> | void;
};

export function ChatHistoryPanel({ conversations, selectedId, onSelect, onDelete, onRename }: Props) {
  const [search, setSearch] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const visibleConversations = conversations.filter(conversation =>
    conversation.title.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()),
  );

  const beginRename = (id: string, title: string) => {
    setEditingId(id);
    setDraftTitle(title);
  };

  const submitRename = async (event: FormEvent, id: string) => {
    event.preventDefault();
    const title = draftTitle.trim();
    if (!title) return;
    await onRename(id, title);
    setEditingId(null);
  };

  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Chat history</h2>
          <span>{conversations.length} sessions</span>
        </div>
      </div>
      <div className="search-box">
        <Search size={15} />
        <input
          type="search"
          aria-label="Search saved chats"
          placeholder="Search saved chats"
          value={search}
          onChange={event => setSearch(event.target.value)}
        />
      </div>
      <div className="document-list">
        {visibleConversations.length === 0 && <div className="empty-state">{conversations.length ? "No matching chats." : "No saved chats yet."}</div>}
        {visibleConversations.map(conversation => (
          <div
            key={conversation.id}
            className={conversation.id === selectedId ? "document-row history-row active" : "document-row history-row"}
          >
            {editingId === conversation.id ? (
              <form className="history-rename" onSubmit={event => void submitRename(event, conversation.id)}>
                <input aria-label="Chat title" value={draftTitle} onChange={event => setDraftTitle(event.target.value)} autoFocus />
                <button type="submit" title="Save chat title"><Check size={13} /></button>
                <button type="button" title="Cancel rename" onClick={() => setEditingId(null)}><X size={13} /></button>
              </form>
            ) : (
              <button type="button" className="history-select" onClick={() => onSelect(conversation.id)}>
                <span className="document-main">
                  <strong>{conversation.title}</strong>
                  <span>{new Date(conversation.updated_at * 1000).toLocaleString()}</span>
                </span>
              </button>
            )}
            <div className="row-actions">
              <button type="button" title="Rename chat" onClick={() => beginRename(conversation.id, conversation.title)}>
                <Pencil size={13} />
              </button>
              <button
                type="button"
                title="Delete chat"
                onClick={() => onDelete(conversation.id)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
