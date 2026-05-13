import type { SourceChunk } from "../../api";

type Props = {
  sources: SourceChunk[];
};

export function SourcesPanel({ sources }: Props) {
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
              <strong>#{source.rank} {source.doc_name}</strong>
              <span>{source.score.toFixed(3)}</span>
            </div>
            <div className="source-metrics">
              {source.vector_score != null && <span>dense {source.vector_score.toFixed(3)}</span>}
              {source.lexical_score != null && <span>bm25 {source.lexical_score.toFixed(3)}</span>}
              {source.fusion_score != null && <span>rrf {source.fusion_score.toFixed(3)}</span>}
              {source.rerank_score != null && <span>rerank {source.rerank_score.toFixed(3)}</span>}
            </div>
            <p>{source.snippet}</p>
          </article>
        ))}
        {sources.length === 0 && <div className="empty-state">Run a query to inspect matched sources.</div>}
      </div>
    </section>
  );
}
