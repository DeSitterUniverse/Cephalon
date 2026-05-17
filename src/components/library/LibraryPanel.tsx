import { BookOpen, FileText, FolderPlus, RefreshCw, Search, Tag, Trash2 } from "lucide-react";
import type { Document } from "../../api";
import { useUiStore } from "../../store";

type Props = {
  documents: Document[];
  search: string;
  setSearch: (value: string) => void;
  statusFilter: string;
  setStatusFilter: (value: string) => void;
  onImportFolder: () => void;
  onImportText: () => void;
  onImportVault: () => void;
  onDelete: (doc: Document) => void;
  onReindex: (doc: Document) => void;
};

export function LibraryPanel({ documents, search, setSearch, statusFilter, setStatusFilter, onImportFolder, onImportText, onImportVault, onDelete, onReindex }: Props) {
  const selectedDocumentId = useUiStore(state => state.selectedDocumentId);
  const setSelectedDocumentId = useUiStore(state => state.setSelectedDocumentId);

  const filtered = documents.filter(doc => {
    const query = search.toLowerCase();
    const matchesSearch = doc.name.toLowerCase().includes(query)
      || doc.path.toLowerCase().includes(query)
      || (doc.tags || []).some(tag => tag.toLowerCase().includes(query));
    const normalizedStatus = doc.status.startsWith("failed") ? "failed" : doc.status === "queued" ? "ingesting" : doc.status;
    const matchesStatus = statusFilter === "all" || normalizedStatus === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="library">
      <div className="panel-header">
        <div>
          <h2>Library</h2>
          <span>{documents.length} documents</span>
        </div>
        <div className="action-row compact-actions">
          <button className="icon-button" onClick={onImportText} title="Import file as text"><FileText size={16} /></button>
          <button className="icon-button" onClick={onImportFolder} title="Import folder"><FolderPlus size={16} /></button>
          <button className="icon-button" onClick={onImportVault} title="Import Obsidian vault"><BookOpen size={16} /></button>
        </div>
      </div>

      <label className="search-box">
        <Search size={15} />
        <input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search path, name, tag" />
      </label>

      <div className="segmented">
        {["all", "ready", "ingesting", "failed"].map(status => (
          <button key={status} className={statusFilter === status ? "active" : ""} onClick={() => setStatusFilter(status)}>
            {status}
          </button>
        ))}
      </div>

      <div className="document-list">
        {filtered.map(doc => (
          <button key={doc.id} className={selectedDocumentId === doc.id ? "document-row active" : "document-row"} onClick={() => setSelectedDocumentId(doc.id)}>
            <div className="document-main">
              <strong>{doc.name}</strong>
              <span>{doc.status} / {doc.chunks} chunks</span>
            </div>
            <div className="document-tags">
              {(doc.tags || []).slice(0, 3).map(tag => <span key={tag}><Tag size={11} />{tag}</span>)}
            </div>
            <div className="row-actions" onClick={event => event.stopPropagation()}>
              <button title="Reindex" onClick={() => onReindex(doc)}><RefreshCw size={13} /></button>
              <button title="Delete" onClick={() => onDelete(doc)}><Trash2 size={13} /></button>
            </div>
          </button>
        ))}
        {filtered.length === 0 && <div className="empty-state">No matching documents.</div>}
      </div>
    </div>
  );
}
