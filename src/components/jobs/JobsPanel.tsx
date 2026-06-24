import { CheckCircle2, Loader2, RefreshCw, Square, XCircle } from "lucide-react";
import type { Job } from "../../api";

type Props = {
  jobs: Job[];
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
};

function iconFor(status: string) {
  if (status === "succeeded") return <CheckCircle2 size={15} />;
  if (status === "failed") return <XCircle size={15} />;
  return <Loader2 size={15} />;
}

export function JobsPanel({ jobs, onCancel, onRetry }: Props) {
  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Jobs</h2>
          <span>Import queue</span>
        </div>
      </div>
      <div className="job-list">
        {jobs.map(job => {
          const filePct = job.total_files ? Math.round((job.processed_files / job.total_files) * 100) : 0;
          const pct = job.status === "running" && job.stage_progress != null
            ? Math.max(filePct, Math.round(((job.processed_files + job.stage_progress / 100) / Math.max(job.total_files, 1)) * 100))
            : filePct;
          return (
            <div key={job.id} className={`job-card ${job.status}`}>
              <div className="job-title">
                {iconFor(job.status)}
                <strong>{job.kind}</strong>
                <span>{job.status}</span>
              </div>
              <div className="progress"><span style={{ width: `${pct}%` }} /></div>
              <div className="job-meta">
                {job.processed_files}/{job.total_files} processed
                {job.skipped_files ? ` / ${job.skipped_files} skipped` : ""}
              </div>
              {job.current_file && <div className="subtle truncate">{job.current_file}</div>}
              {job.stage && job.status === "running" && (
                <div className="job-stage">{formatStage(job.stage)} · {job.stage_progress || 0}%</div>
              )}
              {job.error && <div className="error-text">{job.error}</div>}
              <div className="job-actions">
                {["queued", "running"].includes(job.status) && (
                  <button type="button" onClick={() => onCancel(job.id)}><Square size={12} />Cancel</button>
                )}
                {["failed", "cancelled"].includes(job.status) && (
                  <button type="button" onClick={() => onRetry(job.id)}><RefreshCw size={12} />Retry</button>
                )}
              </div>
            </div>
          );
        })}
        {jobs.length === 0 && <div className="empty-state">No jobs yet.</div>}
      </div>
    </section>
  );
}

function formatStage(stage: string) {
  return stage.charAt(0).toUpperCase() + stage.slice(1);
}
