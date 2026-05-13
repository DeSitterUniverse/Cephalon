import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import type { Job } from "../../api";

type Props = {
  jobs: Job[];
};

function iconFor(status: string) {
  if (status === "succeeded") return <CheckCircle2 size={15} />;
  if (status === "failed") return <XCircle size={15} />;
  return <Loader2 size={15} />;
}

export function JobsPanel({ jobs }: Props) {
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
          const pct = job.total_files ? Math.round((job.processed_files / job.total_files) * 100) : 0;
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
              {job.error && <div className="error-text">{job.error}</div>}
            </div>
          );
        })}
        {jobs.length === 0 && <div className="empty-state">No jobs yet.</div>}
      </div>
    </section>
  );
}
