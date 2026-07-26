import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AnswerSupport, SourceChunk } from "../../api";
import { AnswerSupportPanel } from "./AnswerSupportPanel";
import { SourcesPanel } from "../sources/SourcesPanel";


const source: SourceChunk = {
  rank: 1,
  source_id: "S1",
  doc_id: "doc-1",
  doc_name: "paper.pdf",
  chunk_id: "chunk-1",
  score: 0.91,
  snippet: "The full retrieved parent contains additional context.",
  evidence_text: "RATE improved retrieval recall to 81.7 percent.",
  page_number: 3,
  block_type: "caption",
  assets: [{
    asset_id: "p3-img-a",
    page_number: 3,
    mime_type: "image/png",
    caption: "Figure 1: Retrieval pipeline",
    url: "/documents/doc-1/assets/p3-img-a",
  }],
};


const answerSupport: AnswerSupport = {
  status: "supported",
  accounting: {
    citation_count: 2,
    unique_citation_count: 1,
    cited_source_ids: ["S1"],
    valid_source_ids: ["S1"],
    invalid_source_ids: [],
    duplicate_source_ids: ["S1"],
    malformed_citations: [],
    unused_citation_source_ids: [],
    uncited_source_ids: ["S2"],
    available_source_count: 2,
    uncited_source_count: 1,
    citation_precision: 1,
  },
  claim_validation: {
    method: "deterministic_claim_coverage_v1",
    claim_count: 1,
    supported_claim_count: 1,
    weak_claim_count: 0,
    unsupported_claim_count: 0,
    uncited_claim_count: 0,
    claims: [{
      claim_id: "C1",
      text: "RATE improved retrieval recall to 81.7 percent.",
      source_ids: ["S1"],
      status: "supported",
      reason: "The evidence supports the claim.",
      coverage: 1,
      coverage_by_source: { S1: 1 },
    }],
  },
  citations: [{
    chunk_id: "chunk-1",
    source_id: "S1",
    status: "supported",
    reason: "Strong model-visible evidence.",
    claim_ids: ["C1"],
    claims: ["RATE improved retrieval recall to 81.7 percent."],
    evidence: "RATE improved retrieval recall to 81.7 percent.",
  }],
};


describe("evidence panels", () => {
  it("distinguishes model-visible evidence from the raw retrieved chunk and renders assets accessibly", () => {
    render(<SourcesPanel sources={[source]} />);

    expect(screen.getByText("Evidence sent to model")).toBeInTheDocument();
    expect(screen.getByText("Retrieved chunk")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Figure 1: Retrieval pipeline" })).toHaveAttribute(
      "src",
      "http://127.0.0.1:8765/documents/doc-1/assets/p3-img-a",
    );
  });

  it("shows per-claim status, cited evidence, duplicates, and unused sources", () => {
    render(<AnswerSupportPanel support={answerSupport} />);

    expect(screen.getByText("1 duplicated")).toBeInTheDocument();
    expect(screen.getByText("Unused evidence: S2")).toBeInTheDocument();
    expect(screen.getByText("C1")).toBeInTheDocument();
    expect(screen.getByText("Cited claim")).toBeInTheDocument();
    expect(screen.getAllByText("RATE improved retrieval recall to 81.7 percent.")).toHaveLength(4);
  });
});
