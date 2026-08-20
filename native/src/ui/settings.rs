//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_fixed_model(
        &mut self,
        kind: &str,
        label: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let info = self.data.retrieval.as_ref().and_then(|status| {
            if kind == "embedder" {
                status.embedder.clone()
            } else {
                status.reranker.clone()
            }
        });
        let state = match info.as_ref() {
            Some(model) if model.installed && model.verified => "installed · verified",
            Some(model) if model.installed => "installed · needs verification",
            _ => "not installed",
        };
        let state_color = if state.contains("verified") {
            green()
        } else {
            yellow()
        };
        let kind_download = kind.to_string();
        let kind_verify = kind.to_string();
        let kind_open = kind.to_string();
        let kind_delete = kind.to_string();
        div()
            .p_2()
            .flex()
            .flex_col()
            .gap_1()
            .bg(panel_2())
            .border_1()
            .border_color(line())
            .rounded_sm()
            .child(div().text_color(text()).child(label.to_string()))
            .child(
                div()
                    .text_size(px(11.))
                    .text_color(state_color)
                    .child(state),
            )
            .child(if let Some(model) = info {
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .text_size(px(11.))
                    .text_color(muted())
                    .child(format!(
                        "{}{}",
                        model.name,
                        if model.kind.is_empty() {
                            String::new()
                        } else {
                            format!(" · {}", model.kind)
                        }
                    ))
                    .child(format!(
                        "backend: {}",
                        model.selected_backend.as_deref().unwrap_or("not selected")
                    ))
            } else {
                div()
                    .text_size(px(11.))
                    .text_color(muted())
                    .child("No model metadata available")
            })
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        format!("download-{kind}"),
                        "Download",
                        false,
                        cx.listener(move |this, _, _, cx| {
                            this.download_fixed_model(kind_download.clone(), cx)
                        }),
                    ))
                    .child(ui_button(
                        format!("verify-{kind}"),
                        "Verify",
                        false,
                        cx.listener(move |this, _, _, cx| {
                            this.verify_fixed_model(kind_verify.clone(), cx)
                        }),
                    ))
                    .child(ui_button(
                        format!("open-{kind}"),
                        "Open folder",
                        false,
                        cx.listener(move |this, _, _, cx| {
                            this.open_fixed_model(kind_open.clone(), cx)
                        }),
                    ))
                    .child(ui_button(
                        format!("delete-model-{kind}"),
                        "Delete",
                        false,
                        cx.listener(move |this, _, window, cx| {
                            this.ask_delete_model(kind_delete.clone(), window, cx)
                        }),
                    )),
            )
    }

    pub(super) fn render_settings(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let settings = self.data.settings.clone().unwrap_or_default();
        let toggles = [
            (
                "evidence_required",
                "Evidence required",
                settings.evidence_required,
            ),
            (
                "conversation_memory",
                "Conversation memory",
                settings.conversation_memory,
            ),
            (
                "trace_persistence",
                "Persist retrieval traces",
                settings.trace_persistence,
            ),
            (
                "hierarchical_context",
                "Hierarchical context",
                settings.hierarchical_context,
            ),
            (
                "layout_evidence",
                "Layout evidence",
                settings.layout_evidence,
            ),
            (
                "evidence_ledger",
                "Evidence ledger",
                settings.evidence_ledger,
            ),
            (
                "coverage_selection",
                "Coverage selection",
                settings.coverage_selection,
            ),
            ("gap_retrieval", "Gap retrieval", settings.gap_retrieval),
            (
                "verified_answer_repair",
                "Verified answer repair",
                settings.verified_answer_repair,
            ),
        ];
        let mut toggle_list = div().flex().flex_col().gap_1();
        for (name, label, enabled) in toggles {
            let name_owned = name.to_string();
            toggle_list = toggle_list.child(ui_toggle(
                format!("toggle-{name}"),
                label,
                enabled,
                cx.listener(move |this, _, _, cx| this.toggle_rag_setting(name_owned.clone(), cx)),
            ));
        }
        div()
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(muted())
                    .child("External llama.cpp"),
            )
            .child(detail_line("HTTP API", self.api.base_url(), faint()))
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(muted())
                    .child("Appearance"),
            )
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        "theme-black",
                        "Black",
                        !self.theme_graphite,
                        cx.listener(|this, _, _, cx| this.set_theme(false, cx)),
                    ))
                    .child(ui_button(
                        "theme-graphite",
                        "Graphite",
                        self.theme_graphite,
                        cx.listener(|this, _, _, cx| this.set_theme(true, cx)),
                    )),
            )
            .child(input_field(
                "server-url",
                self.inputs.server_url.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::ServerUrl, window, cx)
                }),
            ))
            .child(input_field(
                "model-name",
                self.inputs.model_name.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::ModelName, window, cx)
                }),
            ))
            .child(input_field(
                "server-context",
                self.inputs.context_tokens.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::ContextTokens, window, cx)
                }),
            ))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        "save-server",
                        "Save endpoint",
                        false,
                        cx.listener(|this, _, _, cx| this.save_server_settings(cx)),
                    ))
                    .child(ui_button(
                        "load-model",
                        "Connect model",
                        true,
                        cx.listener(|this, _, _, cx| this.connect_model(cx)),
                    )),
            )
            .child(if self.data.models.models.is_empty() {
                div()
                    .text_size(px(11.))
                    .text_color(faint())
                    .child("No models reported by llama.cpp.")
            } else {
                div().text_size(px(11.)).text_color(muted()).child(format!(
                    "Available models: {}",
                    self.data.models.models.join(", ")
                ))
            })
            .child(detail_line(
                "Active context",
                &self
                    .data
                    .models
                    .active_context_tokens
                    .map(|tokens| format!("{tokens} tokens"))
                    .unwrap_or_else(|| "unknown".into()),
                muted(),
            ))
            .child(if let Some(llama) = &self.data.models.llama_backend {
                detail_line(
                    "llama.cpp",
                    llama.backend_label.as_deref().unwrap_or("external"),
                    muted(),
                )
            } else {
                div()
            })
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(muted())
                    .child("Retrieval models"),
            )
            .child(self.render_fixed_model("embedder", "Embedder", cx))
            .child(self.render_fixed_model("reranker", "Reranker", cx))
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(muted())
                    .child("Pipeline switches"),
            )
            .child(toggle_list)
            .child(
                div()
                    .text_size(px(12.))
                    .text_color(muted())
                    .child("Retrieval and chunking"),
            )
            .child(self.rag_input(InputTarget::RagTopK, "rag-top-k", "Top K", cx))
            .child(self.rag_input(
                InputTarget::RagRerankTopN,
                "rag-rerank-top-n",
                "Rerank top N",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagMaxTokens,
                "rag-max-tokens",
                "Answer max tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagTemperature,
                "rag-temperature",
                "Temperature",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagParentTargetTokens,
                "rag-parent-target",
                "Parent target tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagParentMaxTokens,
                "rag-parent-max",
                "Parent max tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagChildTargetTokens,
                "rag-child-target",
                "Child target tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagChildMaxTokens,
                "rag-child-max",
                "Child max tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagChildOverlapTokens,
                "rag-child-overlap",
                "Child overlap tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagContextTokens,
                "rag-context-tokens",
                "Retrieval context tokens",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagMinConfidence,
                "rag-min-confidence",
                "No-answer min confidence",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagMinRerankScore,
                "rag-min-rerank",
                "No-answer min rerank",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagMinVectorScore,
                "rag-min-vector",
                "No-answer min vector",
                cx,
            ))
            .child(self.rag_input(
                InputTarget::RagMinSourceCount,
                "rag-min-source-count",
                "No-answer min sources",
                cx,
            ))
            .child(ui_button(
                "save-retrieval-settings",
                "Save retrieval settings",
                true,
                cx.listener(|this, _, _, cx| this.save_rag_settings(cx)),
            ))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        "reindex-stale",
                        "Reindex stale",
                        false,
                        cx.listener(|this, _, _, cx| this.run_reindex(true, cx)),
                    ))
                    .child(ui_button(
                        "reindex-all",
                        "Reindex all",
                        false,
                        cx.listener(|this, _, _, cx| this.run_reindex(false, cx)),
                    )),
            )
            .child(
                if self
                    .data
                    .retrieval
                    .as_ref()
                    .is_some_and(|status| status.reindex_required)
                {
                    div()
                        .text_color(yellow())
                        .child("Retrieval stack requires reindexing.")
                } else {
                    div().text_color(green()).child("Retrieval stack is ready.")
                },
            )
            .child(if let Some(progress) = &self.data.reindex_progress {
                let total = progress.total.max(1);
                let run = progress
                    .run_id
                    .as_deref()
                    .map(|id| format!(" · run {id}"))
                    .unwrap_or_default();
                div()
                    .text_size(px(11.))
                    .text_color(if progress.reindex_required {
                        yellow()
                    } else {
                        muted()
                    })
                    .child(format!(
                        "Reindex: {} · {}/{} processed · {} succeeded · {} failed · {} stale{}",
                        progress.status,
                        progress.processed,
                        total,
                        progress.succeeded,
                        progress.failed,
                        progress.stale_document_count,
                        run
                    ))
            } else {
                div()
                    .text_size(px(11.))
                    .text_color(faint())
                    .child("Reindex progress unavailable")
            })
    }
}
