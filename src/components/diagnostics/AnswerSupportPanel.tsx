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
              {(support.accounting.duplicate_source_ids?.length || 0) > 0 && (
                <span>{support.accounting.duplicate_source_ids!.length} duplicated</span>
              )}
              {(support.accounting.malformed_citations?.length || 0) > 0 && (
                <span>{support.accounting.malformed_citations!.length} malformed</span>
              )}
              {(support.accounting.unused_citation_source_ids?.length || 0) > 0 && (
                <span>{support.accounting.unused_citation_source_ids!.length} unattached</span>
              )}
            </div>
            {(support.accounting.uncited_source_ids?.length || 0) > 0 && (
              <p>Unused evidence: {support.accounting.uncited_source_ids!.join(", ")}</p>
            )}
            <div className="claim-list">
              {support.claim_validation!.claims.map(claim => (
                <div className="claim-row" key={claim.claim_id}>
                  <div className="source-head">
                    <strong>{claim.claim_id}</strong>
                    <span>{claim.status}</span>
                  </div>
                  <p>{claim.text}</p>
                  <small>{claim.reason}</small>
                </div>
              ))}
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
            {citation.claims?.map((claim, index) => (
              <div className="citation-claim" key={`${citation.chunk_id}:claim:${index}`}>
                <span>Cited claim</span>
                <p>{claim}</p>
              </div>
            ))}
            {citation.evidence && (
              <div className="source-evidence">
                <span>Evidence</span>
                <p>{citation.evidence}</p>
              </div>
            )}
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
