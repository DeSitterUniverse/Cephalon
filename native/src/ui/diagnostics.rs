//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_trace(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut list = div().flex().flex_col().gap_1();
        for trace in &self.data.traces {
            let id = trace.query_id.clone();
            list = list.child(ui_button(
                format!("trace-{}", trace.query_id),
                format!(
                    "{} · {} ms · {}",
                    trace.raw_query,
                    trace
                        .total_ms
                        .map(|ms| format!("{ms:.1}"))
                        .unwrap_or_else(|| "?".into()),
                    trace.created_at
                ),
                self.data.selected_trace.is_some()
                    && self.selected_sources.is_empty()
                    && self.panel == Panel::Trace,
                cx.listener(move |this, _, _, cx| this.load_trace(id.clone(), cx)),
            ));
        }
        if let Some(trace) = &self.data.selected_trace {
            list = list.child(
                div()
                    .p_2()
                    .bg(panel_3())
                    .border_1()
                    .border_color(line())
                    .text_size(px(11.))
                    .text_color(muted())
                    .child(diagnostic_value("Trace details", trace)),
            );
        }
        div()
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .child(ui_button(
                "refresh-traces",
                "Refresh traces",
                false,
                cx.listener(|this, _, _, cx| this.refresh_traces(cx)),
            ))
            .child(list)
    }

    pub(super) fn render_health(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut panel = div().flex().flex_col().gap_2().flex_1();
        if let Some(health) = &self.data.health {
            panel = panel
                .child(detail_line(
                    "Backend",
                    &health.status,
                    status_color(&health.status),
                ))
                .child(detail_line(
                    "Active model",
                    health.active_model.as_deref().unwrap_or("none"),
                    text(),
                ))
                .child(detail_line(
                    "Engines",
                    if health.engines_ready {
                        "ready"
                    } else {
                        "starting"
                    },
                    if health.engines_ready {
                        green()
                    } else {
                        yellow()
                    },
                ));
            if let Some(context_tokens) = health.active_context_tokens {
                panel = panel.child(detail_line(
                    "Context",
                    &format!("{context_tokens} tokens"),
                    muted(),
                ));
            }
            if let Some(error) = &health.last_model_load_error {
                panel = panel.child(
                    div()
                        .p_2()
                        .bg(panel_3())
                        .text_color(red())
                        .child(format!("Model load: {error}")),
                );
            }
            if let Some(error) = &health.last_model_error {
                panel = panel.child(
                    div()
                        .p_2()
                        .bg(panel_3())
                        .text_color(red())
                        .child(format!("Model request: {error}")),
                );
            }
            if let Some(error) = &health.retrieval_error {
                panel = panel.child(
                    div()
                        .p_2()
                        .bg(panel_3())
                        .text_color(red())
                        .child(error.clone()),
                );
            }
            if let Some(llama) = &health.llama_backend {
                panel = panel
                    .child(detail_line(
                        "llama.cpp",
                        llama.backend_label.as_deref().unwrap_or("external"),
                        muted(),
                    ))
                    .child(detail_line(
                        "Server",
                        if llama.server_available == Some(true) {
                            "available"
                        } else {
                            "unavailable"
                        },
                        if llama.server_available == Some(true) {
                            green()
                        } else {
                            yellow()
                        },
                    ))
                    .child(detail_line(
                        "Connection",
                        llama.connection_status.as_deref().unwrap_or("unknown"),
                        status_color(llama.connection_status.as_deref().unwrap_or("unknown")),
                    ))
                    .child(detail_line(
                        "Server URL",
                        llama.server_url.as_deref().unwrap_or("not configured"),
                        muted(),
                    ))
                    .child(detail_line(
                        "Configured model",
                        llama.model_name.as_deref().unwrap_or("not selected"),
                        muted(),
                    ));
                if let Some(error) = &llama.server_error {
                    panel = panel.child(
                        div()
                            .p_2()
                            .bg(panel_3())
                            .text_color(red())
                            .child(format!("llama.cpp: {error}")),
                    );
                }
            }
            if let Some(retrieval) = &health.retrieval_stack {
                panel = panel.child(detail_line(
                    "Retrieval stack",
                    if retrieval.reindex_required {
                        "reindex required"
                    } else {
                        "ready"
                    },
                    if retrieval.reindex_required {
                        yellow()
                    } else {
                        green()
                    },
                ));
            }
        }
        if let Some(index) = &self.data.index_health {
            panel = panel
                .child(div().text_size(px(12.)).text_color(muted()).child("Index"))
                .child(detail_line(
                    "Documents",
                    &index.document_count.to_string(),
                    text(),
                ))
                .child(detail_line(
                    "Chunks",
                    &index.chunk_count.to_string(),
                    text(),
                ))
                .child(detail_line(
                    "Embedded",
                    &index.embedded_chunk_count.to_string(),
                    text(),
                ))
                .child(detail_line(
                    "Stale",
                    &index.stale_document_count.to_string(),
                    if index.stale_document_count == 0 {
                        green()
                    } else {
                        yellow()
                    },
                ))
                .child(detail_line(
                    "Failed",
                    &index.failed_ingestion_count.to_string(),
                    if index.failed_ingestion_count == 0 {
                        green()
                    } else {
                        red()
                    },
                ))
                .child(detail_line(
                    "Index size",
                    &format_bytes(index.index_size_bytes),
                    muted(),
                ))
                .child(detail_line(
                    "Average chunk",
                    &format!("{:.1} chars", index.average_chunk_length),
                    muted(),
                ))
                .child(detail_line(
                    "Duplicate rate",
                    &format!("{:.1}%", index.duplicate_chunk_rate * 100.0),
                    muted(),
                ));
        }
        panel
            .child(ui_button(
                "refresh-health",
                "Refresh health",
                false,
                cx.listener(|this, _, _, cx| this.refresh_health(cx)),
            ))
            .child(ui_button(
                "export-metrics",
                "Export metrics",
                false,
                cx.listener(|this, _, _, cx| this.export_metrics(cx)),
            ))
    }

    pub(super) fn render_evaluation(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut runs = div().flex().flex_col().gap_1();
        for run in &self.data.eval_runs {
            runs = runs.child(
                div()
                    .p_2()
                    .bg(panel_2())
                    .border_1()
                    .border_color(line())
                    .rounded_sm()
                    .child(format!(
                        "{} · top_k {} · {} · created {}",
                        run.pipeline,
                        run.top_k,
                        if run.id.is_empty() { "manual" } else { &run.id },
                        run.created_at
                    ))
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(muted())
                            .child(diagnostic_value("Aggregate", &run.aggregate)),
                    ),
            );
        }
        div()
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .child(input_field(
                "eval-question",
                self.inputs.eval_question.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::EvalQuestion, window, cx)
                }),
            ))
            .child(input_field(
                "eval-document",
                self.inputs.eval_document.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::EvalDocument, window, cx)
                }),
            ))
            .child(ui_button(
                "run-eval",
                "Run evaluation",
                true,
                cx.listener(|this, _, _, cx| this.run_eval(cx)),
            ))
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(muted())
                    .child("Saved runs"),
            )
            .child(runs)
    }

    pub(super) fn render_support(&mut self, _cx: &mut Context<Self>) -> gpui::Div {
        match &self.selected_support {
            Some(support) => div()
                .flex()
                .flex_col()
                .gap_2()
                .flex_1()
                .child(
                    div()
                        .text_color(green())
                        .child("Answer-support diagnostics"),
                )
                .child(
                    div()
                        .p_2()
                        .bg(panel_3())
                        .border_1()
                        .border_color(line())
                        .text_size(px(11.))
                        .text_color(muted())
                        .child(diagnostic_value("Support details", support)),
                ),
            None => div()
                .p_3()
                .text_color(muted())
                .child("Answer-support details appear after a response."),
        }
    }
}
