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
              <span>recall {Number(run.aggregate.recall_at_k || 0).toFixed(3)}</span>
              <span>mrr {Number(run.aggregate.mrr || 0).toFixed(3)}</span>
            </div>
          </article>
        ))}
        {runs.length === 0 && <div className="empty-state">Run a small eval to compare retrieval changes.</div>}
      </div>
    </section>
  );
}
