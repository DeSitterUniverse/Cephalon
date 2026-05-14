import { FolderOpen, Plus, RefreshCw, Save, Trash2, X } from "lucide-react";
import { useState } from "react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import type { Document } from "../../api";

type Props = {
  document?: Document;
  onRename: (name: string) => void;
  onAddTag: (tag: string) => void;
  onDeleteTag: (tag: string) => void;
  onReindex: () => void;
  onDelete: () => void;
};

export function DocumentDetails({ document, onRename, onAddTag, onDeleteTag, onReindex, onDelete }: Props) {
  const [name, setName] = useState(document?.name || "");
  const [tag, setTag] = useState("");

  if (!document) {
    return <div className="empty-state">Select a document to view details.</div>;
  }

  const saveName = () => onRename(name.trim() || document.name);
  const addTag = () => {
    if (!tag.trim()) return;
    onAddTag(tag);
    setTag("");
  };

  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Document</h2>
          <span>{document.status} / {document.chunks} chunks</span>
        </div>
      </div>

      <label className="field">
        <span>Display name</span>
        <div className="inline-field">
          <input value={name} onChange={event => setName(event.target.value)} />
          <button className="icon-button" onClick={saveName} title="Save name"><Save size={15} /></button>
        </div>
      </label>

      <div className="meta-grid">
        <span>Size</span><strong>{Math.round((document.size_bytes || 0) / 1024)} KB</strong>
        <span>Indexed</span><strong>{document.last_indexed_at ? new Date(document.last_indexed_at * 1000).toLocaleString() : "Never"}</strong>
      </div>

      <div className="tag-editor">
        <div className="tag-list">
          {(document.tags || []).map(existing => (
            <button key={existing} onClick={() => onDeleteTag(existing)}>{existing}<X size={11} /></button>
          ))}
        </div>
        <div className="inline-field">
          <input value={tag} onChange={event => setTag(event.target.value)} placeholder="Add tag" />
          <button className="icon-button" onClick={addTag} title="Add tag"><Plus size={15} /></button>
        </div>
      </div>

      <div className="action-row">
        <button onClick={() => revealItemInDir(document.path)}><FolderOpen size={14} />Open</button>
        <button onClick={onReindex}><RefreshCw size={14} />Reindex</button>
        <button className="danger" onClick={onDelete}><Trash2 size={14} />Delete</button>
      </div>

      <div className="chunk-preview">
        <h3>Chunk preview</h3>
        {(document.chunk_preview || []).map(chunk => (
          <article key={chunk.id}>
            <strong>Chunk {chunk.index}</strong>
            <p>{chunk.text}</p>
          </article>
        ))}
        {(document.chunk_preview || []).length === 0 && <div className="empty-state">No chunk preview loaded.</div>}
      </div>
    </section>
  );
}
