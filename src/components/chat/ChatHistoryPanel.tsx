import { MessageSquarePlus, Search, Trash2 } from "lucide-react";
import type { Conversation } from "../../api";

type Props = {
  conversations: Conversation[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
};

export function ChatHistoryPanel({ conversations, selectedId, onSelect, onNew, onDelete }: Props) {
  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Chat history</h2>
          <span>{conversations.length} sessions</span>
        </div>
        <button className="icon-button" type="button" onClick={onNew} title="Start new chat">
          <MessageSquarePlus size={15} />
        </button>
      </div>
      <div className="search-box">
        <Search size={15} />
        <input placeholder="Search saved chats" disabled />
      </div>
      <div className="document-list">
        {conversations.length === 0 && <div className="empty-state">No saved chats yet.</div>}
        {conversations.map(conversation => (
          <button
            key={conversation.id}
            type="button"
            className={conversation.id === selectedId ? "document-row active" : "document-row"}
            onClick={() => onSelect(conversation.id)}
          >
            <div className="document-main">
              <strong>{conversation.title}</strong>
              <span>{new Date(conversation.updated_at * 1000).toLocaleString()}</span>
            </div>
            <div className="row-actions">
              <button
                type="button"
                title="Delete chat"
                onClick={(event) => {
                  event.stopPropagation();
                  onDelete(conversation.id);
                }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
