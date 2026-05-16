import type { IndexHealth } from "../../api";

type Props = {
  health?: IndexHealth;
  isLoading?: boolean;
};

function bytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function IndexHealthPanel({ health, isLoading }: Props) {
  if (isLoading) return <div className="empty-state">Loading index health.</div>;
  if (!health) return <div className="empty-state">Index health is unavailable.</div>;

  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Index Health</h2>
          <span>{health.document_count} documents</span>
        </div>
      </div>
      <div className="meta-grid health-grid">
        <span>Chunks</span><strong>{health.chunk_count}</strong>
        <span>Embedded</span><strong>{health.embedded_chunk_count}</strong>
        <span>Stale docs</span><strong>{health.stale_document_count}</strong>
        <span>Failed</span><strong>{health.failed_ingestion_count}</strong>
        <span>Warnings</span><strong>{health.parse_warning_count}</strong>
        <span>Duplicate chunks</span><strong>{health.duplicate_chunk_count} ({(health.duplicate_chunk_rate * 100).toFixed(1)}%)</strong>
        <span>Avg length</span><strong>{health.average_chunk_length}</strong>
        <span>Median length</span><strong>{health.median_chunk_length}</strong>
        <span>Never retrieved</span><strong>{health.documents_never_retrieved}</strong>
        <span>Index size</span><strong>{bytes(health.index_size_bytes)}</strong>
      </div>
      <div className="diagnostic-block">
        <h3>Embedding models</h3>
        {Object.entries(health.embedding_model_counts).map(([model, count]) => <p key={model}><span>{model}</span><strong>{count}</strong></p>)}
      </div>
      <div className="diagnostic-block">
        <h3>Most retrieved</h3>
        {health.top_retrieved_documents.map(doc => <p key={doc.id}><span>{doc.name}</span><strong>{doc.retrieval_count}</strong></p>)}
      </div>
    </section>
  );
}
