//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_sources(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        if self.selected_sources.is_empty() {
            return div()
                .p_3()
                .text_color(muted())
                .child("Sources will appear here after an answer uses the library.");
        }
        let mut list = div().flex().flex_col().gap_2();
        for (index, source) in self.selected_sources.iter().enumerate() {
            let doc_id = source.doc_id.clone();
            let key = source_key(source);
            let expanded = self.expanded_sources.contains(&key);
            let toggle_key = key.clone();
            let mut card = div()
                .id(SharedString::from(format!("source-{index}")))
                .p_2()
                .flex()
                .flex_col()
                .gap_1()
                .bg(panel_2())
                .border_1()
                .border_color(line())
                .rounded_sm()
                .child(div().text_color(orange_light()).child(format!(
                    "{} · {}",
                    source.source_id.as_deref().unwrap_or("source"),
                    source.doc_name
                )))
                .child(div().text_size(px(11.)).text_color(muted()).child(format!(
                    "rank {} · score {:.3} · chunk {}",
                    source.rank,
                    source.final_score.unwrap_or(source.score),
                    source.chunk_id
                )))
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(text())
                        .child(clamp_text(&source.snippet, 280)),
                );
            if let Some(page) = source.page_number {
                card = card.child(div().text_size(px(11.)).text_color(faint()).child(format!(
                            "page {}{}",
                            page,
                            source
                                .page_end
                                .map(|end| format!("–{end}"))
                                .unwrap_or_default()
                        )));
            }
            if expanded {
                if let Some(section) = &source.section_heading {
                    card = card.child(detail_line("Section", section, muted()));
                }
                if !source.heading_path.is_empty() {
                    let heading_path = source.heading_path.join(" › ");
                    card = card.child(detail_line("Heading path", &heading_path, muted()));
                }
                if let Some(kind) = source
                    .source_kind
                    .as_deref()
                    .or(source.block_type.as_deref())
                {
                    card = card.child(detail_line("Type", kind, muted()));
                }
                card = card.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_1()
                        .child(score_badge("dense", source.vector_score))
                        .child(score_badge("BM25", source.lexical_score))
                        .child(score_badge("fusion", source.fusion_score))
                        .child(score_badge("rerank", source.rerank_score))
                        .child(score_badge("final", source.final_score)),
                );
                if let Some(evidence) = &source.evidence_text {
                    card = card.child(disclosure_text("Evidence sent to model", evidence));
                }
                if let Some(raw_chunk) = &source.raw_chunk {
                    card = card.child(disclosure_text("Retrieved raw chunk", raw_chunk));
                }
                if !source.context_assembly.is_null() {
                    card = card.child(disclosure_value(
                        "Context assembly",
                        &source.context_assembly,
                    ));
                }
                if !source.context_selection.is_null() {
                    card = card.child(disclosure_value(
                        "Context selection",
                        &source.context_selection,
                    ));
                }
                if !source.provenance.is_null() {
                    card = card.child(disclosure_value("Provenance", &source.provenance));
                }
                if source.retrieval_round > 0 || source.triggering_gap.is_some() {
                    let round = format!(
                        "{}{}",
                        source.retrieval_round,
                        source
                            .triggering_gap
                            .as_deref()
                            .map(|gap| format!(" · gap {gap}"))
                            .unwrap_or_default()
                    );
                    card = card.child(detail_line("Retrieval round", &round, muted()));
                }
                if source.listwise_rank.is_some() || source.reranker_raw_score.is_some() {
                    let reranker = format!(
                        "rank {} · raw {}",
                        source
                            .listwise_rank
                            .map(|rank| rank.to_string())
                            .unwrap_or_else(|| "–".into()),
                        source
                            .reranker_raw_score
                            .map(|score| format!("{score:.4}"))
                            .unwrap_or_else(|| "–".into())
                    );
                    card = card.child(detail_line("Listwise reranker", &reranker, muted()));
                }
                if !source.table_result.is_empty() || !source.cells.is_empty() {
                    let table = json!({
                        "table_id": source.table_id,
                        "table_title": source.table_title,
                        "sheet_name": source.sheet_name,
                        "cell_refs": source.cell_refs,
                        "header_refs": source.header_refs,
                        "table_result": source.table_result,
                        "cells": source.cells,
                    });
                    card = card.child(disclosure_value("Table / exact cell evidence", &table));
                }
                if !source.assets.is_empty() {
                    card = card.child(disclosure_value(
                        "Extracted assets",
                        &Value::Array(source.assets.clone()),
                    ));
                }
            }
            list = list.child(
                card.child(
                    div()
                        .flex()
                        .gap_1()
                        .child(ui_button(
                            format!("source-details-{index}"),
                            if expanded {
                                "Hide provenance"
                            } else {
                                "Show provenance"
                            },
                            expanded,
                            cx.listener(move |this, _, _, cx| {
                                if !this.expanded_sources.remove(&toggle_key) {
                                    this.expanded_sources.insert(toggle_key.clone());
                                }
                                cx.notify();
                            }),
                        ))
                        .child(ui_button(
                            format!("open-source-document-{index}"),
                            "Open document",
                            false,
                            cx.listener(move |this, _, _, cx| {
                                this.open_document_by_id(doc_id.clone(), cx)
                            }),
                        )),
                ),
            );
        }
        div().flex().flex_col().gap_2().flex_1().child(list)
    }
}
