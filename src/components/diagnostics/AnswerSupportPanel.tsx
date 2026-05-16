import type { AnswerSupport } from "../../api";

type Props = {
  support: AnswerSupport | null;
};

export function AnswerSupportPanel({ support }: Props) {
  if (!support) return <div className="empty-state">Select answer support from a response.</div>;
  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Answer Support</h2>
          <span>{support.status}</span>
        </div>
      </div>
      <div className="diagnostic-list">
        {support.citations.map(citation => (
          <article key={citation.chunk_id} className="source-card">
            <div className="source-head">
              <strong>{citation.source_id || citation.chunk_id}</strong>
              <span>{citation.status}</span>
            </div>
            <p>{citation.reason}</p>
            <div className="source-metrics">
              {citation.score != null && <span>score {citation.score.toFixed(3)}</span>}
              {citation.rerank_score != null && <span>rerank {citation.rerank_score.toFixed(3)}</span>}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
