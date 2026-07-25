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
        {support.accounting && (
          <article className="source-card">
            <div className="source-head">
              <strong>Citation accounting</strong>
              <span>{support.accounting.valid_source_ids.length}/{support.accounting.unique_citation_count} valid</span>
            </div>
            <div className="source-metrics">
              <span>{support.accounting.citation_count} uses</span>
              <span>{support.accounting.uncited_source_count} retrieved but uncited</span>
              {support.accounting.invalid_source_ids.length > 0 && (
                <span>{support.accounting.invalid_source_ids.length} invalid</span>
              )}
            </div>
          </article>
        )}
        {support.claim_validation && (
          <article className="source-card">
            <div className="source-head">
              <strong>Claim validation</strong>
              <span>{support.claim_validation.supported_claim_count}/{support.claim_validation.claim_count} supported</span>
            </div>
            <div className="source-metrics">
              <span>{support.claim_validation.weak_claim_count} weak</span>
              <span>{support.claim_validation.unsupported_claim_count} unsupported</span>
              <span>{support.claim_validation.uncited_claim_count} uncited</span>
            </div>
          </article>
        )}
        {support.citations.map(citation => (
          <article key={`${citation.source_id || "source"}:${citation.chunk_id}`} className="source-card">
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
