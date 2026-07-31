import { useState } from "react";
import type { EvalRun } from "../../api";

type Props = {
  runs: EvalRun[];
  onRun: (question: string, expectedDoc: string) => void;
  isRunning?: boolean;
};

export function EvaluationPanel({ runs, onRun, isRunning }: Props) {
  const [question, setQuestion] = useState("");
  const [expectedDoc, setExpectedDoc] = useState("");

  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Evaluation</h2>
          <span>{runs.length} runs</span>
        </div>
      </div>
      <div className="eval-form">
        <input value={question} onChange={event => setQuestion(event.target.value)} placeholder="Question" />
        <input value={expectedDoc} onChange={event => setExpectedDoc(event.target.value)} placeholder="Expected document id or name" />
        <button disabled={!question.trim() || !expectedDoc.trim() || isRunning} onClick={() => onRun(question, expectedDoc)}>
          {isRunning ? "Running" : "Run eval"}
        </button>
      </div>
      <div className="diagnostic-list">
        {runs.map(run => (
          <article key={run.id} className="source-card">
            <div className="source-head">
              <strong>{run.pipeline}</strong>
              <span>{run.top_k}</span>
            </div>
            <div className="source-metrics">
              <span>recall {metric(run, "recall_at_k")}</span>
              <span>precision {metric(run, "precision_at_k")}</span>
              <span>nDCG {metric(run, "ndcg_at_k")}</span>
              <span>MRR {metric(run, "mrr")}</span>
              {Number(run.aggregate.case_count || 0) > 0 && <span>{Number(run.aggregate.case_count)} cases</span>}
            </div>
          </article>
        ))}
        {runs.length === 0 && <div className="empty-state">Run a small eval to compare retrieval changes.</div>}
      </div>
    </section>
  );
}

/**
 * Aggregate values can also contain domain/category trees. Keeping this guard
 * here prevents a malformed imported report from rendering as ``NaN``.
 */
function metric(run: EvalRun, name: string): string {
  const value = run.aggregate[name];
  return (typeof value === "number" ? value : 0).toFixed(3);
}
