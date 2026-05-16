import type { RetrievalTrace, RetrievalTraceSummary } from "../../api";

type Props = {
  traces: RetrievalTraceSummary[];
  selected?: RetrievalTrace;
  selectedId?: string | null;
  onSelect: (id: string) => void;
};

function value(item: Record<string, unknown>, key: string) {
  const raw = item[key];
  return typeof raw === "number" ? raw.toFixed(3) : raw == null ? "-" : String(raw);
}

export function RetrievalTracePanel({ traces, selected, selectedId, onSelect }: Props) {
  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Retrieval Trace</h2>
          <span>{traces.length} recent</span>
        </div>
      </div>
      <div className="diagnostic-list">
        {traces.map(trace => (
          <button key={trace.query_id} className={trace.query_id === selectedId ? "trace-row active" : "trace-row"} onClick={() => onSelect(trace.query_id)}>
            <strong>{trace.raw_query}</strong>
            <span>{trace.retrieval_mode || "unknown"} · {trace.total_ms ? `${Math.round(trace.total_ms)} ms` : "pending"}</span>
          </button>
        ))}
        {traces.length === 0 && <div className="empty-state">Run a query to capture retrieval stages.</div>}
      </div>
      {selected && (
        <div className="trace-detail">
          <div className="meta-grid compact">
            <span>Mode</span><strong>{selected.retrieval_mode || "unknown"}</strong>
            <span>Confidence</span><strong>{value(selected.no_answer || {}, "confidence")}</strong>
            <span>Total</span><strong>{value(selected.latency, "total_ms")} ms</strong>
          </div>
          {(["vector", "bm25", "fused", "reranked", "unused"] as const).map(stage => (
            <details key={stage} open={stage === "reranked" || stage === "fused"}>
              <summary>{stage} ({selected.candidates[stage]?.length || 0})</summary>
              <table className="compact-table">
                <tbody>
                  {(selected.candidates[stage] || []).slice(0, 8).map((candidate, index) => (
                    <tr key={`${stage}-${index}`}>
                      <td>#{value(candidate, "rank")}</td>
                      <td>{value(candidate, "chunk_id")}</td>
                      <td>{value(candidate, "score")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          ))}
          <details open>
            <summary>Final context ({selected.final_context.length})</summary>
            <div className="context-list">
              {selected.final_context.map((item, index) => (
                <p key={index}><strong>{value(item, "source_id")}</strong> {value(item, "doc_name")} · {value(item, "chunk_id")}</p>
              ))}
            </div>
          </details>
        </div>
      )}
    </section>
  );
}
