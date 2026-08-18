use crate::api::{
    ApiClient, ApiError, Conversation, Document, EventStreamEvent, FixedRetrievalStatus,
    HealthResponse, IndexHealth, LlamaServerSettings, Message, ModelsResponse, QueryEvent,
    QueryRequest, RagSettings, RetrievalTraceSummary, SourceChunk,
};
use crate::backend::BackendService;
use async_channel::{Receiver, Sender};
use gpui::prelude::*;
use gpui::{
    div, px, App, ClickEvent, Context, ExternalPaths, FocusHandle, Focusable, KeyDownEvent,
    ParentElement, PathPromptOptions, SharedString, Window,
};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn bg() -> gpui::Rgba {
    gpui::rgb(0x000000)
}

impl NativeApp {
    fn render_right_panel(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let content = match self.panel {
            Panel::History => self.render_history(cx),
            Panel::Document => self.render_document(cx),
            Panel::Sources => self.render_sources(cx),
            Panel::Settings => self.render_settings(cx),
            Panel::Trace => self.render_trace(cx),
            Panel::Health => self.render_health(cx),
            Panel::Evaluation => self.render_evaluation(cx),
            Panel::Support => self.render_support(cx),
        };
        div()
            .w(px(390.))
            .h_full()
            .p_3()
            .flex()
            .flex_col()
            .gap_2()
            .bg(panel())
            .border_l_1()
            .border_color(line())
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_size(px(15.))
                            .text_color(text())
                            .child(self.panel.title()),
                    )
                    .child(ui_button(
                        "close-details",
                        "×",
                        false,
                        cx.listener(|this, _, _, cx| {
                            this.right_open = false;
                            cx.notify();
                        }),
                    )),
            )
            .child(content)
    }

    fn render_history(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut list = div().flex().flex_col().gap_1().flex_1();
        for conversation in &self.data.conversations {
            let id = conversation.id.clone();
            let title = if conversation.title.is_empty() {
                "Untitled chat".to_string()
            } else {
                conversation.title.clone()
            };
            let selected = self.selected_conversation.as_deref() == Some(conversation.id.as_str());
            let delete_id = conversation.id.clone();
            let delete_title = title.clone();
            let mut row = div()
                .id(SharedString::from(format!("history-{}", conversation.id)))
                .w_full()
                .p_2()
                .flex()
                .items_center()
                .gap_1()
                .rounded_sm()
                .border_1()
                .border_color(if selected { orange() } else { line() })
                .bg(if selected { panel_3() } else { panel_2() })
                .cursor_pointer()
                .on_click(
                    cx.listener(move |this, _, _, cx| this.select_conversation(id.clone(), cx)),
                )
                .child(div().flex_1().text_color(text()).child(title));
            row = row.child(ui_button(
                format!("delete-chat-{}", delete_id),
                "×",
                false,
                cx.listener(move |this, _, _, cx| {
                    this.ask_delete_conversation(delete_id.clone(), delete_title.clone(), cx)
                }),
            ));
            list = list.child(row);
        }
        let title = self
            .selected_conversation
            .as_ref()
            .and_then(|id| self.data.conversations.iter().find(|item| &item.id == id))
            .map(|item| item.title.clone())
            .unwrap_or_default();
        let rename_value = if self.rename_draft.is_empty() {
            title
        } else {
            self.rename_draft.clone()
        };
        div()
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .child(ui_button(
                "history-new",
                "+ New chat",
                true,
                cx.listener(|this, _, _, cx| this.new_conversation(cx)),
            ))
            .child(list)
            .child(input_field(
                "rename-chat",
                &rename_value,
                "Rename selected chat",
                self.active_input == InputTarget::RenameConversation,
                cx.listener(|this, _, window, _| {
                    this.active_input = InputTarget::RenameConversation;
                    window.focus(&this.focus);
                }),
            ))
            .child(ui_button(
                "save-chat-name",
                "Save chat name",
                false,
                cx.listener(|this, _, _, cx| this.rename_conversation(cx)),
            ))
    }

    fn render_document(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let document = self
            .selected_document
            .as_ref()
            .and_then(|id| self.data.documents.iter().find(|item| &item.id == id))
            .cloned();
        let Some(document) = document else {
            return div()
                .p_3()
                .text_color(muted())
                .child("Select a document from the library to inspect it.");
        };
        let rename_value = if self.rename_draft.is_empty() {
            document.name.clone()
        } else {
            self.rename_draft.clone()
        };
        let mut tags = div().flex().flex_wrap().gap_1();
        for tag in &document.tags {
            let tag_value = tag.clone();
            tags = tags.child(ui_button(
                format!("tag-{}", tag),
                format!("{} ×", tag),
                false,
                cx.listener(move |this, _, _, cx| this.remove_tag(tag_value.clone(), cx)),
            ));
        }
        div()
            .flex()
            .flex_col()
            .gap_2()
            .flex_1()
            .child(
                div()
                    .text_size(px(18.))
                    .text_color(text())
                    .child(document.name.clone()),
            )
            .child(detail_line(
                "Status",
                &document.status,
                status_color(&document.status),
            ))
            .child(detail_line("Path", &document.path, muted()))
            .child(detail_line("Chunks", &document.chunks.to_string(), text()))
            .child(detail_line(
                "Size",
                &document
                    .size_bytes
                    .map(|size| format_bytes(size))
                    .unwrap_or_else(|| "Unknown".into()),
                muted(),
            ))
            .child(if document.stale_embedding {
                div()
                    .text_color(yellow())
                    .child("Embedding is stale; reindex this document.")
            } else {
                div().text_color(green()).child("Index is current.")
            })
            .child(if let Some(error) = document.last_error.clone() {
                div().p_2().bg(panel_3()).text_color(red()).child(error)
            } else {
                div()
            })
            .child(div().text_size(px(12.)).text_color(muted()).child("Tags"))
            .child(tags)
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(input_field(
                        "new-tag",
                        &self.tag_draft,
                        "Add a tag",
                        self.active_input == InputTarget::Tag,
                        cx.listener(|this, _, window, _| {
                            this.focus_input(InputTarget::Tag, window)
                        }),
                    ))
                    .child(ui_button(
                        "add-tag",
                        "Add",
                        false,
                        cx.listener(|this, _, _, cx| this.add_tag(cx)),
                    )),
            )
            .child(input_field(
                "rename-document",
                &rename_value,
                "Rename document",
                self.active_input == InputTarget::RenameDocument,
                cx.listener(|this, _, window, _| {
                    this.active_input = InputTarget::RenameDocument;
                    window.focus(&this.focus);
                }),
            ))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        "save-document-name",
                        "Save name",
                        false,
                        cx.listener(|this, _, _, cx| this.rename_document(cx)),
                    ))
                    .child(ui_button("reindex-document", "Reindex", false, {
                        let id = document.id.clone();
                        cx.listener(move |this, _, _, cx| this.reindex_document(id.clone(), cx))
                    }))
                    .child(ui_button("delete-document", "Delete", false, {
                        let id = document.id.clone();
                        let name = document.name.clone();
                        cx.listener(move |this, _, _, cx| {
                            this.ask_delete_document(id.clone(), name.clone(), cx)
                        })
                    })),
            )
    }

    fn render_sources(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        if self.selected_sources.is_empty() {
            return div()
                .p_3()
                .text_color(muted())
                .child("Sources will appear here after an answer uses the library.");
        }
        let mut list = div().flex().flex_col().gap_2().flex_1();
        for (index, source) in self.selected_sources.iter().enumerate() {
            let doc_id = source.doc_id.clone();
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
                .child(
                    div()
                        .text_color(orange_light())
                        .child(format!("#{} {}", source.rank, source.doc_name)),
                )
                .child(div().text_size(px(11.)).text_color(muted()).child(format!(
                    "score {:.3} · chunk {}",
                    source.final_score.unwrap_or(source.score),
                    source.chunk_id
                )))
                .child(div().text_color(text()).child(source.snippet.clone()));
            if let Some(page) = source.page_number {
                card = card.child(
                    div()
                        .text_size(px(11.))
                        .text_color(faint())
                        .child(format!("page {page}")),
                );
            }
            list = list.child(card.child(ui_button(
                format!("open-source-document-{index}"),
                "Open document",
                false,
                cx.listener(move |this, _, _, cx| this.select_document(doc_id.clone(), cx)),
            )));
        }
        div().flex().flex_col().gap_2().flex_1().child(list)
    }

    fn render_fixed_model(&mut self, kind: &str, label: &str, cx: &mut Context<Self>) -> gpui::Div {
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
                    .text_size(px(11.))
                    .text_color(muted())
                    .child(model.name)
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
                        cx.listener(move |this, _, _, cx| {
                            this.confirmation = Some(Confirmation {
                                title: "Delete cached model?".into(),
                                message: format!("Remove the cached {kind_delete} model?"),
                                action: ConfirmationAction::DeleteModel(kind_delete.clone()),
                            });
                            cx.notify();
                        }),
                    )),
            )
    }

    fn render_settings(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let server_url = self.server_url_draft.clone();
        let model_name = self.model_name_draft.clone();
        let context_tokens = self.context_tokens_draft.clone();
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
            toggle_list = toggle_list.child(ui_button(
                format!("toggle-{name}"),
                format!("{} {}", if enabled { "✓" } else { "○" }, label),
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
            .child(input_field(
                "server-url",
                &server_url,
                "http://127.0.0.1:8080",
                self.active_input == InputTarget::ServerUrl,
                cx.listener(|this, _, window, _| this.focus_input(InputTarget::ServerUrl, window)),
            ))
            .child(input_field(
                "model-name",
                &model_name,
                "Model name",
                self.active_input == InputTarget::ModelName,
                cx.listener(|this, _, window, _| this.focus_input(InputTarget::ModelName, window)),
            ))
            .child(input_field(
                "server-context",
                &context_tokens,
                "Context tokens",
                self.active_input == InputTarget::ContextTokens,
                cx.listener(|this, _, window, _| {
                    this.focus_input(InputTarget::ContextTokens, window)
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
    }

    fn render_trace(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut list = div().flex().flex_col().gap_1().flex_1();
        for trace in &self.data.traces {
            let id = trace.query_id.clone();
            list = list.child(ui_button(
                format!("trace-{}", trace.query_id),
                format!(
                    "{} · {} ms",
                    trace.raw_query,
                    trace
                        .total_ms
                        .map(|ms| format!("{ms:.1}"))
                        .unwrap_or_else(|| "?".into())
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
                    .child(pretty_value(trace)),
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
                cx.listener(|this, _, _, cx| this.refresh_snapshot(cx)),
            ))
            .child(list)
    }

    fn render_health(&mut self, cx: &mut Context<Self>) -> gpui::Div {
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
                cx.listener(|this, _, _, cx| this.refresh_snapshot(cx)),
            ))
            .child(ui_button(
                "export-metrics",
                "Export metrics",
                false,
                cx.listener(|this, _, _, cx| this.export_metrics(cx)),
            ))
    }

    fn render_evaluation(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut runs = div().flex().flex_col().gap_1().flex_1();
        for run in &self.data.eval_runs {
            runs = runs.child(
                div()
                    .p_2()
                    .bg(panel_2())
                    .border_1()
                    .border_color(line())
                    .rounded_sm()
                    .child(format!("{} · top_k {}", run.pipeline, run.top_k))
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(muted())
                            .child(pretty_value(&run.aggregate)),
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
                &self.eval_question,
                "Evaluation question",
                self.active_input == InputTarget::EvalQuestion,
                cx.listener(|this, _, window, _| {
                    this.focus_input(InputTarget::EvalQuestion, window)
                }),
            ))
            .child(input_field(
                "eval-document",
                &self.eval_document,
                "Expected document id",
                self.active_input == InputTarget::EvalDocument,
                cx.listener(|this, _, window, _| {
                    this.focus_input(InputTarget::EvalDocument, window)
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

    fn render_support(&mut self, _cx: &mut Context<Self>) -> gpui::Div {
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
                        .child(pretty_value(support)),
                ),
            None => div()
                .p_3()
                .text_color(muted())
                .child("Answer-support details appear after a response."),
        }
    }

    fn render_overlays(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut overlay = div().flex().flex_col().gap_1();
        for notice in &self.notices {
            overlay = overlay.child(
                div()
                    .p_2()
                    .bg(panel_3())
                    .border_1()
                    .border_color(notice.color)
                    .text_size(px(12.))
                    .text_color(notice.color)
                    .child(notice.message.clone()),
            );
        }
        if let Some(confirmation) = self.confirmation.clone() {
            overlay = overlay.child(
                div()
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .bg(panel_3())
                    .border_1()
                    .border_color(orange())
                    .child(div().text_color(text()).child(confirmation.title))
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(muted())
                            .child(confirmation.message),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child(ui_button(
                                "confirm-action",
                                "Confirm",
                                true,
                                cx.listener(|this, _, _, cx| this.confirm_action(cx)),
                            ))
                            .child(ui_button(
                                "cancel-action",
                                "Cancel",
                                false,
                                cx.listener(|this, _, _, cx| this.close_confirmation(cx)),
                            )),
                    ),
            );
        }
        div().w_full().p_3().child(overlay)
    }
}

fn panel() -> gpui::Rgba {
    gpui::rgb(0x020202)
}

fn panel_2() -> gpui::Rgba {
    gpui::rgb(0x080808)
}

fn panel_3() -> gpui::Rgba {
    gpui::rgb(0x121212)
}

fn line() -> gpui::Rgba {
    gpui::rgb(0x252525)
}

fn line_strong() -> gpui::Rgba {
    gpui::rgb(0x3c3c3c)
}

fn text() -> gpui::Rgba {
    gpui::rgb(0xffe5cc)
}

fn muted() -> gpui::Rgba {
    gpui::rgb(0x9b9b9b)
}

fn faint() -> gpui::Rgba {
    gpui::rgb(0x666666)
}

fn orange() -> gpui::Rgba {
    gpui::rgb(0xff9a2e)
}

fn orange_light() -> gpui::Rgba {
    gpui::rgb(0xffbd6b)
}

fn green() -> gpui::Rgba {
    gpui::rgb(0x6ee7a8)
}

fn yellow() -> gpui::Rgba {
    gpui::rgb(0xe9b949)
}

fn red() -> gpui::Rgba {
    gpui::rgb(0xf87171)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    History,
    Document,
    Sources,
    Settings,
    Trace,
    Health,
    Evaluation,
    Support,
}

impl Panel {
    fn title(self) -> &'static str {
        match self {
            Self::History => "Chats",
            Self::Document => "Document",
            Self::Sources => "Sources",
            Self::Settings => "Settings",
            Self::Trace => "Retrieval trace",
            Self::Health => "Index health",
            Self::Evaluation => "Evaluation",
            Self::Support => "Answer support",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputTarget {
    Composer,
    Search,
    ServerUrl,
    ModelName,
    ContextTokens,
    EvalQuestion,
    EvalDocument,
    RenameDocument,
    RenameConversation,
    Tag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BootState {
    Starting,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventStatus {
    Connecting,
    Connected,
    Reconnecting,
    Offline,
}

impl EventStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Connecting => "connecting",
            Self::Connected => "connected",
            Self::Reconnecting => "reconnecting",
            Self::Offline => "offline",
        }
    }

    fn color(self) -> gpui::Rgba {
        match self {
            Self::Connected => green(),
            Self::Connecting | Self::Reconnecting => yellow(),
            Self::Offline => red(),
        }
    }
}

#[derive(Debug, Clone)]
struct ChatMessage {
    id: Option<String>,
    role: String,
    content: String,
    sources: Vec<SourceChunk>,
    support: Option<Value>,
    streaming: bool,
    error: bool,
}

impl From<&crate::api::StoredMessage> for ChatMessage {
    fn from(message: &crate::api::StoredMessage) -> Self {
        Self {
            id: Some(message.id.clone()),
            role: message.role.clone(),
            content: message.content.clone(),
            sources: message.sources.clone(),
            support: message
                .meta
                .as_ref()
                .and_then(|meta| meta.get("support").cloned()),
            streaming: false,
            error: false,
        }
    }
}

#[derive(Debug, Clone)]
struct Notice {
    id: u64,
    message: String,
    color: gpui::Rgba,
}

#[derive(Debug, Clone)]
struct Confirmation {
    title: String,
    message: String,
    action: ConfirmationAction,
}

#[derive(Debug, Clone)]
enum ConfirmationAction {
    DeleteDocument(String),
    DeleteConversation(String),
    DeleteModel(String),
}

#[derive(Debug, Default)]
struct WorkspaceData {
    health: Option<HealthResponse>,
    models: ModelsResponse,
    documents: Vec<Document>,
    conversations: Vec<Conversation>,
    conversation: Option<Conversation>,
    settings: Option<RagSettings>,
    server: Option<LlamaServerSettings>,
    retrieval: Option<FixedRetrievalStatus>,
    index_health: Option<IndexHealth>,
    traces: Vec<RetrievalTraceSummary>,
    selected_trace: Option<Value>,
    eval_runs: Vec<crate::api::EvalRun>,
}

pub struct NativeApp {
    api: ApiClient,
    backend: Arc<BackendService>,
    stop: Arc<AtomicBool>,
    focus: FocusHandle,
    boot: BootState,
    boot_status: String,
    boot_error: Option<String>,
    data: WorkspaceData,
    panel: Panel,
    left_open: bool,
    right_open: bool,
    theme_graphite: bool,
    event_status: EventStatus,
    search: String,
    status_filter: String,
    selected_document: Option<String>,
    selected_conversation: Option<String>,
    selected_sources: Vec<SourceChunk>,
    selected_support: Option<Value>,
    composer: String,
    retrieval_scope: String,
    response_effort: String,
    response_phase: String,
    messages: Vec<ChatMessage>,
    is_typing: bool,
    query_stop: Option<Arc<AtomicBool>>,
    active_input: InputTarget,
    server_url_draft: String,
    model_name_draft: String,
    context_tokens_draft: String,
    eval_question: String,
    eval_document: String,
    rename_draft: String,
    tag_draft: String,
    notice_counter: u64,
    notices: Vec<Notice>,
    confirmation: Option<Confirmation>,
}

#[derive(Debug)]
struct Snapshot {
    health: HealthResponse,
    models: ModelsResponse,
    documents: Vec<Document>,
    conversations: Vec<Conversation>,
    settings: Option<RagSettings>,
    server: Option<LlamaServerSettings>,
    retrieval: Option<FixedRetrievalStatus>,
    index_health: Option<IndexHealth>,
    traces: Vec<RetrievalTraceSummary>,
    eval_runs: Vec<crate::api::EvalRun>,
    conversation: Option<Conversation>,
}

impl NativeApp {
    pub fn new(
        api: ApiClient,
        backend: Arc<BackendService>,
        stop: Arc<AtomicBool>,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus = cx.focus_handle();
        let mut app = Self {
            api,
            backend,
            stop,
            focus,
            boot: BootState::Starting,
            boot_status: "Starting local service…".into(),
            boot_error: None,
            data: WorkspaceData::default(),
            panel: Panel::History,
            left_open: true,
            right_open: true,
            theme_graphite: false,
            event_status: EventStatus::Connecting,
            search: String::new(),
            status_filter: "all".into(),
            selected_document: None,
            selected_conversation: None,
            selected_sources: Vec::new(),
            selected_support: None,
            composer: String::new(),
            retrieval_scope: "medium".into(),
            response_effort: "balanced".into(),
            response_phase: String::new(),
            messages: Vec::new(),
            is_typing: false,
            query_stop: None,
            active_input: InputTarget::Composer,
            server_url_draft: String::new(),
            model_name_draft: String::new(),
            context_tokens_draft: String::new(),
            eval_question: String::new(),
            eval_document: String::new(),
            rename_draft: String::new(),
            tag_draft: String::new(),
            notice_counter: 0,
            notices: Vec::new(),
            confirmation: None,
        };
        app.start_boot(cx);
        app.start_event_stream(cx);
        app
    }

    fn start_boot(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        let backend = self.backend.clone();
        let selected_conversation = self.selected_conversation.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || {
                    let _ = backend.start();
                    let started = Instant::now();
                    loop {
                        let status = match api.health() {
                            Ok(health) if health.status == "ok" || health.status == "ready" => {
                                let snapshot =
                                    load_snapshot(&api, selected_conversation.as_deref())?;
                                return Ok(snapshot);
                            }
                            Ok(health) => health
                                .startup_error
                                .unwrap_or_else(|| "The local service is still starting.".into()),
                            Err(error) => error.to_string(),
                        };
                        if started.elapsed() > Duration::from_secs(30) {
                            return Err(ApiError {
                                status: None,
                                message: status,
                            });
                        }
                        std::thread::sleep(Duration::from_millis(650));
                    }
                })
                .await;
                let _ = this.update(&mut *cx, |this, cx| {
                    match result {
                        Ok(snapshot) => {
                            this.boot = BootState::Ready;
                            this.boot_status = "Local service ready".into();
                            this.boot_error = None;
                            this.apply_snapshot(snapshot, cx);
                        }
                        Err(error) => {
                            this.boot = BootState::Failed;
                            this.boot_status = "Local backend is unavailable.".into();
                            this.boot_error = Some(error.to_string());
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn start_event_stream(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        let stop = self.stop.clone();
        let (tx, rx) = async_channel::unbounded();
        smol::spawn(async move {
            smol::unblock(move || {
                api.event_loop(&stop, |event| {
                    let _ = tx.try_send(event);
                });
            })
            .await;
        })
        .detach();

        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                while let Ok(event) = rx.recv().await {
                    let _ = this.update(&mut *cx, |this, cx| {
                        match event {
                            EventStreamEvent::Connected | EventStreamEvent::Heartbeat => {
                                this.event_status = EventStatus::Connected;
                            }
                            EventStreamEvent::DataChanged => {
                                this.event_status = EventStatus::Connected;
                                this.refresh_snapshot(cx);
                            }
                            EventStreamEvent::Error(message) => {
                                this.event_status = EventStatus::Reconnecting;
                                this.boot_error = Some(message);
                            }
                        }
                        cx.notify();
                    });
                }
            },
        )
        .detach();
    }

    fn refresh_snapshot(&mut self, cx: &mut Context<Self>) {
        if self.boot != BootState::Ready {
            return;
        }
        let api = self.api.clone();
        let selected_conversation = self.selected_conversation.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result =
                    smol::unblock(move || load_snapshot(&api, selected_conversation.as_deref()))
                        .await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if let Ok(snapshot) = result {
                        this.apply_snapshot(snapshot, cx);
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn apply_snapshot(&mut self, snapshot: Snapshot, cx: &mut Context<Self>) {
        let first_conversation = snapshot.conversations.first().map(|item| item.id.clone());
        self.data.health = Some(snapshot.health);
        self.data.models = snapshot.models;
        self.data.documents = snapshot.documents;
        self.data.conversations = snapshot.conversations;
        self.data.settings = snapshot.settings;
        self.data.server = snapshot.server;
        self.data.retrieval = snapshot.retrieval;
        self.data.index_health = snapshot.index_health;
        self.data.traces = snapshot.traces;
        self.data.eval_runs = snapshot.eval_runs;
        self.data.conversation = snapshot.conversation;
        if let Some(server) = &self.data.server {
            self.server_url_draft = server.server_url.clone();
            self.model_name_draft = server.model_name.clone();
            self.context_tokens_draft = server
                .context_tokens
                .map(|value| value.to_string())
                .unwrap_or_default();
        }
        if self.selected_conversation.is_none() {
            self.selected_conversation = first_conversation;
        }
        if let Some(conversation) = &self.data.conversation {
            self.selected_conversation = Some(conversation.id.clone());
            self.messages = conversation
                .messages
                .iter()
                .map(ChatMessage::from)
                .collect();
        } else if self.selected_conversation.is_some() {
            self.load_selected_conversation(cx);
        }
    }

    fn load_selected_conversation(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.selected_conversation.clone() else {
            self.messages.clear();
            return;
        };
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.conversation(&id)).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if let Ok(conversation) = result {
                        this.data.conversation = Some(conversation.clone());
                        this.messages = conversation
                            .messages
                            .iter()
                            .map(ChatMessage::from)
                            .collect();
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn notify(&mut self, message: impl Into<String>, color: gpui::Rgba, cx: &mut Context<Self>) {
        self.notice_counter += 1;
        self.notices.push(Notice {
            id: self.notice_counter,
            message: message.into(),
            color,
        });
        if self.notices.len() > 4 {
            self.notices.remove(0);
        }
        cx.notify();
    }

    fn focus_input(&mut self, target: InputTarget, window: &mut Window) {
        self.active_input = target;
        window.focus(&self.focus);
    }

    fn handle_key(&mut self, event: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let key = event.keystroke.key.to_ascii_lowercase();
        if key == "escape" {
            if self.confirmation.is_some() {
                self.confirmation = None;
            } else {
                self.right_open = false;
            }
            cx.notify();
            return;
        }
        if key == "enter" || key == "return" {
            if self.active_input == InputTarget::Composer {
                self.send_message(cx);
            }
            return;
        }
        if key == "backspace" {
            self.active_value_mut().pop();
            cx.notify();
            return;
        }
        if event.keystroke.modifiers.control
            || event.keystroke.modifiers.platform
            || event.keystroke.modifiers.alt
        {
            return;
        }
        if let Some(value) = &event.keystroke.key_char {
            if !value.is_empty() {
                self.active_value_mut().push_str(value);
                window.focus(&self.focus);
                cx.notify();
            }
        }
    }

    fn active_value_mut(&mut self) -> &mut String {
        match self.active_input {
            InputTarget::Composer => &mut self.composer,
            InputTarget::Search => &mut self.search,
            InputTarget::ServerUrl => &mut self.server_url_draft,
            InputTarget::ModelName => &mut self.model_name_draft,
            InputTarget::ContextTokens => &mut self.context_tokens_draft,
            InputTarget::EvalQuestion => &mut self.eval_question,
            InputTarget::EvalDocument => &mut self.eval_document,
            InputTarget::RenameDocument | InputTarget::RenameConversation => &mut self.rename_draft,
            InputTarget::Tag => &mut self.tag_draft,
        }
    }

    fn retry_backend(&mut self, cx: &mut Context<Self>) {
        self.boot = BootState::Starting;
        self.boot_status = "Retrying local backend…".into();
        self.boot_error = None;
        self.start_boot(cx);
    }

    fn select_conversation(&mut self, id: String, cx: &mut Context<Self>) {
        self.rename_draft = self
            .data
            .conversations
            .iter()
            .find(|conversation| conversation.id == id)
            .map(|conversation| conversation.title.clone())
            .unwrap_or_default();
        self.selected_conversation = Some(id);
        self.panel = Panel::History;
        self.right_open = true;
        self.load_selected_conversation(cx);
        cx.notify();
    }

    fn select_document(&mut self, id: String, cx: &mut Context<Self>) {
        self.rename_draft = self
            .data
            .documents
            .iter()
            .find(|document| document.id == id)
            .map(|document| document.name.clone())
            .unwrap_or_default();
        self.selected_document = Some(id);
        self.panel = Panel::Document;
        self.right_open = true;
        cx.notify();
    }

    fn choose_panel(&mut self, panel: Panel, cx: &mut Context<Self>) {
        self.panel = panel;
        self.right_open = true;
        cx.notify();
    }

    fn new_conversation(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.create_conversation()).await;
                let _ = this.update(&mut *cx, |this, cx| match result {
                    Ok(conversation) => {
                        this.selected_conversation = Some(conversation.id.clone());
                        this.messages.clear();
                        this.data.conversation = Some(conversation);
                        this.notify("New chat created.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn send_message(&mut self, cx: &mut Context<Self>) {
        let prompt = self.composer.trim().to_string();
        let Some(model) = self.data.models.active_model.clone() else {
            self.notify(
                "Connect to the configured external llama.cpp server first.",
                yellow(),
                cx,
            );
            return;
        };
        if self.is_typing || prompt.is_empty() || self.data.settings.is_none() {
            return;
        }
        self.selected_sources.clear();
        self.selected_support = None;
        let history: Vec<Message> = self
            .messages
            .iter()
            .filter(|message| !message.streaming)
            .map(|message| Message {
                role: message.role.clone(),
                content: message.content.clone(),
            })
            .collect();
        let assistant_id = format!(
            "draft-{}",
            self.notice_counter + self.messages.len() as u64 + 1
        );
        self.messages.push(ChatMessage {
            id: None,
            role: "user".into(),
            content: prompt.clone(),
            sources: Vec::new(),
            support: None,
            streaming: false,
            error: false,
        });
        self.messages.push(ChatMessage {
            id: Some(assistant_id),
            role: "assistant".into(),
            content: String::new(),
            sources: Vec::new(),
            support: None,
            streaming: true,
            error: false,
        });
        self.composer.clear();
        self.is_typing = true;
        self.response_phase = "Connecting…".into();
        let stop = Arc::new(AtomicBool::new(false));
        self.query_stop = Some(stop.clone());
        let (tx, rx): (Sender<QueryEvent>, Receiver<QueryEvent>) = async_channel::unbounded();
        let api = self.api.clone();
        let request = QueryRequest {
            prompt,
            model,
            history,
            settings: self.data.settings.clone(),
            conversation_id: self.selected_conversation.clone(),
            retrieval_scope: self.retrieval_scope.clone(),
            response_effort: self.response_effort.clone(),
        };
        smol::spawn(async move {
            let result = smol::unblock(move || {
                let result = api.query_stream(request, &stop, |event| {
                    let _ = tx.try_send(event);
                });
                if let Err(error) = result {
                    let _ = tx.try_send(QueryEvent::Error(error.to_string()));
                }
            })
            .await;
            let _ = result;
        })
        .detach();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                while let Ok(event) = rx.recv().await {
                    let terminal = matches!(event, QueryEvent::Done | QueryEvent::Error(_));
                    let _ = this.update(&mut *cx, |this, cx| {
                        this.apply_query_event(event, cx);
                    });
                    if terminal {
                        break;
                    }
                }
                let _ = this.update(&mut *cx, |this, cx| {
                    this.is_typing = false;
                    this.query_stop = None;
                    this.response_phase.clear();
                    this.refresh_snapshot(cx);
                    cx.notify();
                });
            },
        )
        .detach();
        cx.notify();
    }

    fn apply_query_event(&mut self, event: QueryEvent, cx: &mut Context<Self>) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        match event {
            QueryEvent::Phase(phase) => self.response_phase = phase_label(&phase).into(),
            QueryEvent::Token(text) => last.content.push_str(&text),
            QueryEvent::Source(source) => {
                last.sources.push(source.clone());
                self.selected_sources.push(source);
            }
            QueryEvent::Conversation(id) => {
                self.selected_conversation = Some(id);
            }
            QueryEvent::AnswerMeta(meta) => {
                last.support = meta.get("support").cloned();
                if let Some(support) = &last.support {
                    self.selected_support = Some(support.clone());
                }
            }
            QueryEvent::Error(message) => {
                last.content = message;
                last.error = true;
                last.streaming = false;
            }
            QueryEvent::Done => last.streaming = false,
        }
        cx.notify();
    }

    fn stop_query(&mut self, cx: &mut Context<Self>) {
        if let Some(stop) = &self.query_stop {
            stop.store(true, Ordering::Relaxed);
        }
        self.is_typing = false;
        if let Some(last) = self.messages.last_mut() {
            last.streaming = false;
        }
        self.response_phase.clear();
        cx.notify();
    }

    fn select_and_ingest(&mut self, directories: bool, force_text: bool, cx: &mut Context<Self>) {
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: !directories,
            directories,
            multiple: false,
            prompt: Some(if directories {
                "Import folder".into()
            } else {
                "Import text file".into()
            }),
        });
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = receiver.await;
                let Some(path) = result
                    .ok()
                    .and_then(|result| result.ok())
                    .flatten()
                    .and_then(|paths| paths.into_iter().next())
                else {
                    return;
                };
                let path = path.to_string_lossy().to_string();
                let result = smol::unblock(move || api.ingest_path(&path, force_text)).await;
                let _ = this.update(&mut *cx, |this, cx| match result {
                    Ok(response) => {
                        this.notify(
                            response
                                .message
                                .unwrap_or_else(|| "Ingestion queued.".into()),
                            green(),
                            cx,
                        );
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn ingest_dropped(&mut self, paths: &ExternalPaths, cx: &mut Context<Self>) {
        let paths = paths
            .paths()
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        if paths.is_empty() {
            return;
        }
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let results = smol::unblock(move || {
                    paths
                        .into_iter()
                        .map(|path| {
                            let result = api.ingest_path(&path, false);
                            (path, result)
                        })
                        .collect::<Vec<_>>()
                })
                .await;
                let _ = this.update(&mut *cx, |this, cx| {
                    let mut queued = 0;
                    let mut errors = Vec::new();
                    for (path, result) in results {
                        match result {
                            Ok(_) => queued += 1,
                            Err(error) => errors.push(format!("{}: {}", path, error)),
                        }
                    }
                    if queued > 0 {
                        this.notify(
                            format!("Queued {queued} dropped path(s) for ingestion."),
                            green(),
                            cx,
                        );
                    }
                    for error in errors {
                        this.notify(error, red(), cx);
                    }
                    this.refresh_snapshot(cx);
                });
            },
        )
        .detach();
    }

    fn ask_delete_document(&mut self, id: String, name: String, cx: &mut Context<Self>) {
        self.confirmation = Some(Confirmation {
            title: "Delete document?".into(),
            message: format!("Delete {name} from the library and remove its indexed content?"),
            action: ConfirmationAction::DeleteDocument(id),
        });
        cx.notify();
    }

    fn ask_delete_conversation(&mut self, id: String, title: String, cx: &mut Context<Self>) {
        self.confirmation = Some(Confirmation {
            title: "Delete chat?".into(),
            message: format!("Delete {title} and its saved messages?"),
            action: ConfirmationAction::DeleteConversation(id),
        });
        cx.notify();
    }

    fn confirm_action(&mut self, cx: &mut Context<Self>) {
        let Some(confirmation) = self.confirmation.take() else {
            return;
        };
        let api = self.api.clone();
        match confirmation.action {
            ConfirmationAction::DeleteDocument(id) => {
                cx.spawn(
                    async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                        let result = smol::unblock(move || api.delete_document(&id)).await;
                        let _ = this.update(&mut *cx, |this, cx| match result {
                            Ok(_) => {
                                this.notify("Document deleted.", green(), cx);
                                this.selected_document = None;
                                this.refresh_snapshot(cx);
                            }
                            Err(error) => this.notify(error.to_string(), red(), cx),
                        });
                    },
                )
                .detach();
            }
            ConfirmationAction::DeleteConversation(id) => {
                let selected_id = id.clone();
                cx.spawn(
                    async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                        let result = smol::unblock(move || api.delete_conversation(&id)).await;
                        let _ = this.update(&mut *cx, |this, cx| match result {
                            Ok(_) => {
                                this.notify("Chat deleted.", green(), cx);
                                if this.selected_conversation.as_deref() == Some(&selected_id) {
                                    this.selected_conversation = None;
                                    this.messages.clear();
                                }
                                this.refresh_snapshot(cx);
                            }
                            Err(error) => this.notify(error.to_string(), red(), cx),
                        });
                    },
                )
                .detach();
            }
            ConfirmationAction::DeleteModel(kind) => {
                cx.spawn(
                    async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                        let result = smol::unblock(move || api.delete_fixed_model(&kind)).await;
                        let _ = this.update(&mut *cx, |this, cx| match result {
                            Ok(_) => {
                                this.notify("Model cache removed.", green(), cx);
                                this.refresh_snapshot(cx);
                            }
                            Err(error) => this.notify(error.to_string(), red(), cx),
                        });
                    },
                )
                .detach();
            }
        }
    }

    fn close_confirmation(&mut self, cx: &mut Context<Self>) {
        self.confirmation = None;
        cx.notify();
    }

    fn save_server_settings(&mut self, cx: &mut Context<Self>) {
        let settings = LlamaServerSettings {
            server_url: self.server_url_draft.trim().to_string(),
            model_name: self.model_name_draft.trim().to_string(),
            context_tokens: self.context_tokens_draft.trim().parse().ok(),
        };
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.update_server_settings(&settings)).await;
                let _ = this.update(&mut *cx, |this, cx| match result {
                    Ok(settings) => {
                        this.data.server = Some(settings);
                        this.notify("Saved llama.cpp endpoint.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn connect_model(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        self.notify("Connecting to external llama.cpp server…", yellow(), cx);
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.load_model()).await;
                let _ = this.update(&mut *cx, |this, cx| match result {
                    Ok(response) => {
                        this.notify(
                            format!(
                                "Connected to {}.",
                                response
                                    .active_model
                                    .unwrap_or_else(|| "llama.cpp server".into())
                            ),
                            green(),
                            cx,
                        );
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn reindex_document(&mut self, id: String, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.reindex_document(&id)).await;
                let _ = this.update(&mut *cx, |this, cx| match result {
                    Ok(response) => {
                        this.notify(
                            response.message.unwrap_or_else(|| "Reindex queued.".into()),
                            green(),
                            cx,
                        );
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn run_eval(&mut self, cx: &mut Context<Self>) {
        let question = self.eval_question.trim().to_string();
        let expected_doc = self.eval_document.trim().to_string();
        if question.is_empty() || expected_doc.is_empty() {
            self.notify(
                "Enter an evaluation question and expected document id.",
                yellow(),
                cx,
            );
            return;
        }
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result =
                    smol::unblock(move || api.run_manual_eval(&question, &expected_doc)).await;
                let _ = this.update(&mut *cx, |this, cx| match result {
                    Ok(_) => {
                        this.notify("Eval run saved.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn load_trace(&mut self, id: String, cx: &mut Context<Self>) {
        self.data.selected_trace = None;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.retrieval_trace(&id)).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if let Ok(trace) = result {
                        this.data.selected_trace = Some(trace);
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }
}

impl Focusable for NativeApp {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Drop for NativeApp {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.backend.shutdown();
    }
}

impl NativeApp {
    fn cycle_retrieval_scope(&mut self, cx: &mut Context<Self>) {
        self.retrieval_scope = match self.retrieval_scope.as_str() {
            "small" => "medium",
            "medium" => "large",
            _ => "small",
        }
        .into();
        cx.notify();
    }

    fn cycle_response_effort(&mut self, cx: &mut Context<Self>) {
        self.response_effort = match self.response_effort.as_str() {
            "fast" => "balanced",
            "balanced" => "deep",
            _ => "fast",
        }
        .into();
        cx.notify();
    }

    fn toggle_rag_setting(&mut self, name: String, cx: &mut Context<Self>) {
        let Some(settings) = &mut self.data.settings else {
            return;
        };
        match name.as_str() {
            "evidence_required" => settings.evidence_required = !settings.evidence_required,
            "conversation_memory" => settings.conversation_memory = !settings.conversation_memory,
            "trace_persistence" => settings.trace_persistence = !settings.trace_persistence,
            "hierarchical_context" => {
                settings.hierarchical_context = !settings.hierarchical_context
            }
            "layout_evidence" => settings.layout_evidence = !settings.layout_evidence,
            "evidence_ledger" => settings.evidence_ledger = !settings.evidence_ledger,
            "coverage_selection" => settings.coverage_selection = !settings.coverage_selection,
            "gap_retrieval" => settings.gap_retrieval = !settings.gap_retrieval,
            "verified_answer_repair" => {
                settings.verified_answer_repair = !settings.verified_answer_repair
            }
            _ => return,
        }
        self.save_rag_settings(cx);
    }

    fn save_rag_settings(&mut self, cx: &mut Context<Self>) {
        let Some(settings) = self.data.settings.clone() else {
            return;
        };
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.update_settings(&settings)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(settings) => {
                        this.data.settings = Some(settings);
                        this.notify(
                            "Retrieval settings saved. Reindex after changing chunk boundaries.",
                            green(),
                            cx,
                        );
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn run_reindex(&mut self, stale_only: bool, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || {
                    if stale_only {
                        api.reindex_stale()
                    } else {
                        api.reindex_all()
                    }
                })
                .await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(value) => {
                        let total = value.get("total").and_then(Value::as_i64).unwrap_or(0);
                        this.notify(
                            format!("Queued {total} document(s) for reindexing."),
                            green(),
                            cx,
                        );
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn rename_document(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.selected_document.clone() else {
            return;
        };
        let name = self.rename_draft.trim().to_string();
        if name.is_empty() {
            return;
        }
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.rename_document(&id, &name)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(document) => {
                        this.rename_draft = document.name.clone();
                        this.notify("Document renamed.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn rename_conversation(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.selected_conversation.clone() else {
            return;
        };
        let title = self.rename_draft.trim().to_string();
        if title.is_empty() {
            return;
        }
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.rename_conversation(&id, &title)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(conversation) => {
                        this.rename_draft = conversation.title.clone();
                        this.data.conversation = Some(conversation);
                        this.notify("Chat renamed.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn add_tag(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.selected_document.clone() else {
            return;
        };
        let tag = self.tag_draft.trim().to_string();
        if tag.is_empty() {
            return;
        }
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.add_document_tag(&id, &tag)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(_) => {
                        this.tag_draft.clear();
                        this.notify("Tag added.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn remove_tag(&mut self, tag: String, cx: &mut Context<Self>) {
        let Some(id) = self.selected_document.clone() else {
            return;
        };
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.delete_document_tag(&id, &tag)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(_) => this.refresh_snapshot(cx),
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn verify_fixed_model(&mut self, kind: String, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.verify_fixed_model(&kind)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(model) if model.verified => {
                        this.notify("Model integrity verified.", green(), cx)
                    }
                    Ok(_) => this.notify("Model integrity check failed.", red(), cx),
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn download_fixed_model(&mut self, kind: String, cx: &mut Context<Self>) {
        let api = self.api.clone();
        self.notify(format!("Downloading {kind}…"), yellow(), cx);
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.download_fixed_model(&kind)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(_) => {
                        this.notify("Model downloaded and verified.", green(), cx);
                        this.refresh_snapshot(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn open_fixed_model(&mut self, kind: String, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.open_fixed_model_directory(&kind)).await;
                let _ = this.update(cx, |this, cx| {
                    if let Err(error) = result {
                        this.notify(error.to_string(), red(), cx);
                    }
                });
            },
        )
        .detach();
    }

    fn export_metrics(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.export_metrics()).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(export) if export.status == "success" => this.notify(
                        format!("Metrics exported: {}", export.path.unwrap_or_default()),
                        green(),
                        cx,
                    ),
                    Ok(export) => this.notify(
                        format!(
                            "Metrics export failed: {}",
                            export.error.unwrap_or_default()
                        ),
                        red(),
                        cx,
                    ),
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn render_boot(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let retry = if self.boot == BootState::Failed {
            ui_button(
                "boot-retry",
                "Retry backend",
                true,
                cx.listener(|this, _, _, cx| this.retry_backend(cx)),
            )
        } else {
            ui_button("boot-spacer", "", false, |_, _, _| {})
        };
        let mut card = div()
            .w(px(540.))
            .p_6()
            .bg(panel_2())
            .border_1()
            .border_color(line())
            .rounded_sm()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .text_size(px(24.))
                    .text_color(text())
                    .child("Cephalon"),
            )
            .child(
                div()
                    .text_size(px(13.))
                    .text_color(muted())
                    .child("Native GPUI workbench"),
            )
            .child(
                div()
                    .text_color(if self.boot == BootState::Failed {
                        red()
                    } else {
                        yellow()
                    })
                    .child(self.boot_status.clone()),
            );
        if let Some(error) = &self.boot_error {
            card = card.child(
                div()
                    .p_3()
                    .bg(panel_3())
                    .border_1()
                    .border_color(line())
                    .text_color(red())
                    .child(error.clone()),
            );
        }
        card = card.child(retry);
        div()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(bg())
            .child(card)
    }

    fn render_shell(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let model_name = self
            .data
            .models
            .active_model
            .clone()
            .or_else(|| {
                self.data
                    .server
                    .as_ref()
                    .map(|server| server.model_name.clone())
            })
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "No model connected".into());
        let model_color = if self.data.models.active_model.is_some() {
            green()
        } else {
            yellow()
        };
        let mut body = div().flex().flex_1();
        if self.left_open {
            body = body.child(self.render_library(cx));
        }
        body = body.child(self.render_nav(cx)).child(self.render_chat(cx));
        if self.right_open {
            body = body.child(self.render_right_panel(cx));
        }
        let topbar = div()
            .h(px(58.))
            .w_full()
            .px_4()
            .flex()
            .items_center()
            .justify_between()
            .bg(panel_2())
            .border_b_1()
            .border_color(line())
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .text_size(px(18.))
                            .text_color(orange_light())
                            .child("CEPHALON"),
                    )
                    .child(ui_button(
                        "new-chat",
                        "+ New chat",
                        false,
                        cx.listener(|this, _, _, cx| this.new_conversation(cx)),
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().text_color(model_color).child(model_name))
                    .child(ui_button(
                        "connect-model",
                        "Connect",
                        self.data.models.active_model.is_some(),
                        cx.listener(|this, _, _, cx| this.connect_model(cx)),
                    ))
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(self.event_status.color())
                            .child(format!("● {}", self.event_status.label())),
                    )
                    .child(ui_button(
                        "toggle-library",
                        if self.left_open {
                            "Hide library"
                        } else {
                            "Show library"
                        },
                        false,
                        cx.listener(|this, _, _, cx| {
                            this.left_open = !this.left_open;
                            cx.notify();
                        }),
                    ))
                    .child(ui_button(
                        "toggle-details",
                        if self.right_open {
                            "Hide details"
                        } else {
                            "Show details"
                        },
                        false,
                        cx.listener(|this, _, _, cx| {
                            this.right_open = !this.right_open;
                            cx.notify();
                        }),
                    )),
            );
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(bg())
            .text_color(text())
            .child(topbar)
            .child(body)
            .child(self.render_overlays(cx))
    }

    fn render_library(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut list = div().flex().flex_col().gap_1().flex_1();
        let mut visible_count = 0;
        let query = self.search.to_ascii_lowercase();
        for document in self.data.documents.iter().filter(|document| {
            let status_match = self.status_filter == "all" || document.status == self.status_filter;
            let text_match = query.is_empty()
                || document.name.to_ascii_lowercase().contains(&query)
                || document.path.to_ascii_lowercase().contains(&query);
            status_match && text_match
        }) {
            visible_count += 1;
            let id = document.id.clone();
            let selected = self.selected_document.as_deref() == Some(document.id.as_str());
            let row = div()
                .id(SharedString::from(format!("library-{}", document.id)))
                .w_full()
                .p_2()
                .flex()
                .flex_col()
                .gap_1()
                .rounded_sm()
                .border_1()
                .border_color(if selected { orange() } else { line() })
                .bg(if selected { panel_3() } else { panel_2() })
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, cx| this.select_document(id.clone(), cx)))
                .child(div().text_color(text()).child(document.name.clone()))
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(status_color(&document.status))
                        .child(format!("{} · {} chunks", document.status, document.chunks)),
                );
            list = list.child(row);
        }
        let empty = if self.data.documents.is_empty() {
            "No documents yet. Import a folder or text file."
        } else {
            "No documents match this filter."
        };
        if visible_count == 0 {
            list = list.child(div().p_2().text_color(muted()).child(empty));
        }
        div()
            .w(px(300.))
            .h_full()
            .p_3()
            .flex()
            .flex_col()
            .gap_2()
            .bg(panel())
            .border_r_1()
            .border_color(line())
            .child(div().text_size(px(15.)).text_color(text()).child("Library"))
            .child(input_field(
                "library-search",
                &self.search,
                "Search documents",
                self.active_input == InputTarget::Search,
                cx.listener(|this, _, window, _| this.focus_input(InputTarget::Search, window)),
            ))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(status_filter_button("all", &self.status_filter, cx))
                    .child(status_filter_button("ready", &self.status_filter, cx))
                    .child(status_filter_button("error", &self.status_filter, cx)),
            )
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        "import-folder",
                        "Import folder",
                        false,
                        cx.listener(|this, _, _, cx| this.select_and_ingest(true, false, cx)),
                    ))
                    .child(ui_button(
                        "import-text",
                        "Import text",
                        false,
                        cx.listener(|this, _, _, cx| this.select_and_ingest(false, true, cx)),
                    )),
            )
            .child(list)
    }

    fn render_nav(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut nav = div()
            .w(px(112.))
            .h_full()
            .p_2()
            .flex()
            .flex_col()
            .gap_1()
            .bg(panel_2())
            .border_r_1()
            .border_color(line());
        nav = nav.child(
            div()
                .p_1()
                .text_size(px(10.))
                .text_color(faint())
                .child("WORK"),
        );
        for (panel, label) in [
            (Panel::History, "Chats"),
            (Panel::Document, "Document"),
            (Panel::Sources, "Sources"),
            (Panel::Settings, "Settings"),
        ] {
            nav = nav.child(ui_button(
                format!("nav-{}", label.to_ascii_lowercase()),
                label,
                self.panel == panel,
                cx.listener(move |this, _, _, cx| this.choose_panel(panel, cx)),
            ));
        }
        nav = nav.child(
            div()
                .p_1()
                .text_size(px(10.))
                .text_color(faint())
                .child("DIAGNOSTICS"),
        );
        for (panel, label) in [
            (Panel::Trace, "Trace"),
            (Panel::Health, "Health"),
            (Panel::Evaluation, "Eval"),
            (Panel::Support, "Support"),
        ] {
            nav = nav.child(ui_button(
                format!("nav-{}", label.to_ascii_lowercase()),
                label,
                self.panel == panel,
                cx.listener(move |this, _, _, cx| this.choose_panel(panel, cx)),
            ));
        }
        nav.child(
            div().flex_1().child(
                div()
                    .text_size(px(10.))
                    .text_color(faint())
                    .child("DROP FILES HERE"),
            ),
        )
    }

    fn render_chat(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let title = self
            .data
            .conversations
            .iter()
            .find(|conversation| {
                self.selected_conversation.as_deref() == Some(conversation.id.as_str())
            })
            .map(|conversation| conversation.title.clone())
            .unwrap_or_else(|| "New conversation".into());
        let mut messages = div().flex().flex_col().gap_3().p_5().flex_1();
        if self.messages.is_empty() {
            messages = messages.child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap_2()
                    .p_6()
                    .text_color(muted())
                    .child("Ask a question about your library")
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(faint())
                            .child("Sources and retrieval traces will appear here."),
                    ),
            );
        }
        for (index, message) in self.messages.iter().enumerate() {
            let user = message.role == "user";
            let content = if message.content.is_empty() && message.streaming {
                "…".to_string()
            } else {
                message.content.clone()
            };
            let mut card = div()
                .id(SharedString::from(format!("message-{index}")))
                .w_full()
                .p_3()
                .rounded_sm()
                .border_1()
                .border_color(if message.error { red() } else { line() })
                .bg(if user { panel_3() } else { panel_2() })
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(if user { orange_light() } else { muted() })
                        .child(if user { "YOU" } else { "CEPHALON" }),
                )
                .child(
                    div()
                        .mt_2()
                        .text_color(if message.error { red() } else { text() })
                        .child(content),
                );
            if message.streaming {
                card = card.child(
                    div()
                        .text_size(px(11.))
                        .text_color(yellow())
                        .child(self.response_phase.clone()),
                );
            }
            if !message.sources.is_empty() {
                let sources = message.sources.clone();
                card = card.child(ui_button(
                    format!("message-sources-{index}"),
                    format!("{} sources", sources.len()),
                    false,
                    cx.listener(move |this, _, _, cx| {
                        this.selected_sources = sources.clone();
                        this.choose_panel(Panel::Sources, cx);
                    }),
                ));
            }
            if let Some(support) = &message.support {
                let support = support.clone();
                card = card.child(ui_button(
                    format!("message-support-{index}"),
                    "Answer support",
                    false,
                    cx.listener(move |this, _, _, cx| {
                        this.selected_support = Some(support.clone());
                        this.choose_panel(Panel::Support, cx);
                    }),
                ));
            }
            messages = messages.child(card);
        }
        let composer = input_field(
            "composer",
            &self.composer,
            "Ask Cephalon about your documents…",
            self.active_input == InputTarget::Composer,
            cx.listener(|this, _, window, _| this.focus_input(InputTarget::Composer, window)),
        );
        div()
            .flex()
            .flex_col()
            .flex_1()
            .h_full()
            .bg(bg())
            .child(
                div()
                    .h(px(58.))
                    .px_5()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(line())
                    .child(div().text_size(px(16.)).text_color(text()).child(title))
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(muted())
                            .child(if self.is_typing {
                                self.response_phase.clone()
                            } else {
                                "Ready".into()
                            }),
                    ),
            )
            .child(messages)
            .child(
                div()
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .border_t_1()
                    .border_color(line())
                    .child(composer)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(ui_button(
                                "retrieval-scope",
                                format!("Scope: {}", self.retrieval_scope),
                                false,
                                cx.listener(|this, _, _, cx| this.cycle_retrieval_scope(cx)),
                            ))
                            .child(ui_button(
                                "response-effort",
                                format!("Effort: {}", self.response_effort),
                                false,
                                cx.listener(|this, _, _, cx| this.cycle_response_effort(cx)),
                            ))
                            .child(div().flex_1())
                            .child(if self.is_typing {
                                ui_button(
                                    "stop-query",
                                    "Stop",
                                    false,
                                    cx.listener(|this, _, _, cx| this.stop_query(cx)),
                                )
                            } else {
                                ui_button(
                                    "send-query",
                                    "Send",
                                    true,
                                    cx.listener(|this, _, _, cx| this.send_message(cx)),
                                )
                            }),
                    ),
            )
    }
}

fn ui_button(
    id: impl Into<SharedString>,
    label: impl Into<SharedString>,
    active: bool,
    listener: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id.into())
        .flex_none()
        .px_2()
        .py_1()
        .bg(if active { panel_3() } else { panel_2() })
        .border_1()
        .border_color(if active { orange() } else { line() })
        .rounded_sm()
        .cursor_pointer()
        .text_size(px(12.))
        .text_color(if active { orange_light() } else { text() })
        .child(label.into())
        .on_click(listener)
}

fn input_field(
    id: impl Into<SharedString>,
    value: &str,
    placeholder: &str,
    active: bool,
    listener: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    let display = if value.is_empty() { placeholder } else { value };
    div()
        .id(id.into())
        .w_full()
        .h(px(36.))
        .p_2()
        .border_1()
        .border_color(if active { orange() } else { line() })
        .rounded_sm()
        .cursor_pointer()
        .text_size(px(12.))
        .text_color(if value.is_empty() { muted() } else { text() })
        .child(display.to_string())
        .on_click(listener)
}

fn status_filter_button(
    status: &'static str,
    selected: &str,
    cx: &mut Context<NativeApp>,
) -> gpui::Stateful<gpui::Div> {
    let value = status.to_string();
    ui_button(
        format!("status-filter-{status}"),
        status,
        selected == status,
        cx.listener(move |this, _, _, cx| {
            this.status_filter = value.clone();
            cx.notify();
        }),
    )
}

fn detail_line(label: &str, value: &str, color: gpui::Rgba) -> gpui::Div {
    div()
        .flex()
        .justify_between()
        .gap_2()
        .child(
            div()
                .text_size(px(11.))
                .text_color(muted())
                .child(label.to_string()),
        )
        .child(
            div()
                .text_size(px(12.))
                .text_color(color)
                .child(value.to_string()),
        )
}

fn status_color(status: &str) -> gpui::Rgba {
    match status {
        "ready" | "completed" | "connected" | "ok" => green(),
        "error" | "failed" | "offline" => red(),
        _ => yellow(),
    }
}

fn format_bytes(bytes: i64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let units = ["KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = "B";
    for next in units {
        value /= 1024.0;
        unit = next;
        if value < 1024.0 {
            break;
        }
    }
    format!("{value:.1} {unit}")
}

fn pretty_value(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn load_snapshot(
    api: &ApiClient,
    selected_conversation: Option<&str>,
) -> Result<Snapshot, ApiError> {
    let health = api.health()?;
    let models = api.models().unwrap_or_default();
    let documents = api.documents().unwrap_or_default();
    let conversations = api.conversations().unwrap_or_default();
    let settings = api.settings().ok();
    let server = api.server_settings().ok();
    let retrieval = api.fixed_retrieval_status().ok();
    let index_health = api.index_health().ok();
    let traces = api.retrieval_traces().unwrap_or_default();
    let eval_runs = api.eval_runs().unwrap_or_default();
    let selected = selected_conversation
        .and_then(|id| api.conversation(id).ok())
        .or_else(|| {
            conversations
                .first()
                .and_then(|item| api.conversation(&item.id).ok())
        });
    Ok(Snapshot {
        health,
        models,
        documents,
        conversations,
        settings,
        server,
        retrieval,
        index_health,
        traces,
        eval_runs,
        conversation: selected,
    })
}

fn response_phase_label(phase: &str) -> &'static str {
    match phase {
        "routing" => "Choosing whether to search documents…",
        "retrieving" => "Retrieving relevant context…",
        "drafting" => "Drafting an answer…",
        "refining" => "Refining the answer…",
        "answering" => "Writing the answer…",
        "evidence_required" => "Evidence check complete",
        _ => "Retrieving relevant context…",
    }
}

fn phase_label(phase: &str) -> &'static str {
    if phase == "Connecting…" {
        "Connecting…"
    } else {
        response_phase_label(phase)
    }
}

impl Render for NativeApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut root = if self.boot == BootState::Ready {
            self.render_shell(cx)
        } else {
            self.render_boot(cx)
        };
        root = root
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                this.handle_key(event, window, cx)
            }))
            .on_drop::<ExternalPaths>(
                cx.listener(|this, paths, _, cx| this.ingest_dropped(paths, cx)),
            );
        root
    }
}
