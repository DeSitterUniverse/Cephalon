import type { SourceChunk } from "../../api";
import { apiUrl } from "../../api/client";

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
              <strong>
                <span className={`source-kind-badge source-kind-${source.source_kind || "text"}`}>
                  {source.source_kind || "text"}
                </span>{" "}
                {source.source_id || `#${source.rank}`} {source.doc_name}
              </strong>
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
              {source.table_title && <span>table {source.table_title}</span>}
              {source.sheet_name && <span>sheet {source.sheet_name}</span>}
              {source.table_operation && <span>operation {source.table_operation}</span>}
              {(source.cell_refs?.length || 0) > 0 && <span>{source.cell_refs!.length} cited cells</span>}
              {source.vector_score != null && <span>dense {source.vector_score.toFixed(3)}</span>}
              {source.lexical_score != null && <span>bm25 {source.lexical_score.toFixed(3)}</span>}
              {source.fusion_score != null && <span>rrf {source.fusion_score.toFixed(3)}</span>}
              {source.reranker_raw_score != null && <span>v3.5 raw {source.reranker_raw_score.toFixed(3)}</span>}
              {source.listwise_rank != null && <span>listwise #{source.listwise_rank}</span>}
              {source.final_score != null && <span>final {source.final_score.toFixed(3)}</span>}
              {source.context_assembly?.context_kind && source.context_assembly.context_kind !== "child" && (
                <span>
                  {source.context_assembly.context_kind === "parent" ? "parent context" : "sibling span"}
                  {source.context_assembly.context_tokens != null ? ` · ${source.context_assembly.context_tokens} tokens` : ""}
                </span>
              )}
              {(source.context_assembly?.structural_relationships?.length || 0) > 0 && (
                <span>{source.context_assembly?.structural_relationships?.length} layout links</span>
              )}
              {source.context_selection?.requirement_coverage && (
                <span>
                  {Object.values(source.context_selection.requirement_coverage).filter(value => value >= 0.34).length} requirements
                </span>
              )}
              {(source.retrieval_round || 0) > 0 && (
                <span>retrieval round {source.retrieval_round} · gap {source.triggering_gap || "unknown"}</span>
              )}
            </div>
            {(source.cell_refs?.length || 0) > 0 && (
              <div className="source-evidence table-citation-details">
                <span>Exact cell evidence</span>
                <p>{source.cell_refs!.join(", ")}</p>
                {(source.header_refs?.length || 0) > 0 && <p>Headers: {source.header_refs!.join(", ")}</p>}
                {source.cells?.filter(cell => source.cell_refs!.includes(cell.cell_ref)).map(cell => (
                  <p key={cell.cell_ref}>
                    <strong>{cell.cell_ref}</strong>
                    {cell.header ? ` · ${cell.header}` : ""}
                    {` = ${cell.raw_value}`}
                    {cell.sheet_name ? ` · sheet ${cell.sheet_name}` : ""}
                    {cell.page_number != null ? ` · page ${cell.page_number}` : ""}
                    {cell.bounding_box ? ` · box ${cell.bounding_box.join(", ")}` : ""}
                  </p>
                ))}
              </div>
            )}
            {source.evidence_text && (
              <div className="source-evidence">
                <span>Evidence sent to model</span>
                <p>{source.evidence_text}</p>
              </div>
            )}
            {(!source.evidence_text || source.evidence_text !== source.snippet) && (
              <details className="source-raw">
                <summary>Retrieved chunk</summary>
                <p>{source.snippet}</p>
              </details>
            )}
            {source.assets?.map(asset => (
              <figure className="source-asset" key={asset.asset_id}>
                <img
                  src={apiUrl(asset.url)}
                  alt={asset.caption || `Extracted figure on page ${asset.page_number}`}
                  loading="lazy"
                />
                <figcaption>
                  {asset.caption || `Embedded image · page ${asset.page_number}`}
                </figcaption>
              </figure>
            ))}
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
