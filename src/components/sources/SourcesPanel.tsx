import type { SourceChunk } from "../../api";

type Props = {
  sources: SourceChunk[];
  onOpenDocument?: (id: string) => void;
};

export function SourcesPanel({ sources, onOpenDocument }: Props) {
  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Sources</h2>
          <span>{sources.length} matches</span>
        </div>
      </div>
      <div className="source-list">
        {sources.map(source => (
          <article key={source.chunk_id} className="source-card">
            <div className="source-head">
              <strong>{source.source_id || `#${source.rank}`} {source.doc_name}</strong>
              <span>{source.score.toFixed(3)}</span>
            </div>
            <div className="source-metrics">
              {source.page_number != null && (
                <span>
                  page {source.page_number}
                  {source.page_end != null && source.page_end !== source.page_number ? `-${source.page_end}` : ""}
                </span>
              )}
              {source.section_heading && <span>{source.section_heading}</span>}
              {source.block_type && source.block_type !== "paragraph" && <span>{source.block_type}</span>}
              {source.vector_score != null && <span>dense {source.vector_score.toFixed(3)}</span>}
              {source.lexical_score != null && <span>bm25 {source.lexical_score.toFixed(3)}</span>}
              {source.fusion_score != null && <span>rrf {source.fusion_score.toFixed(3)}</span>}
              {source.rerank_score != null && <span>rerank {source.rerank_score.toFixed(3)}</span>}
            </div>
            <p>{source.snippet}</p>
            {onOpenDocument && (
              <button className="source-document-link" type="button" onClick={() => onOpenDocument(source.doc_id)}>
                Open document
              </button>
            )}
          </article>
        ))}
        {sources.length === 0 && <div className="empty-state">Run a query to inspect matched sources.</div>}
      </div>
    </section>
  );
}
