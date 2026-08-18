use crate::api::{
    merge_conversation_messages, ApiClient, ApiError, Conversation, Document, EventStreamEvent,
    EventStreamRefresh, FixedRetrievalStatus, HealthResponse, IndexHealth, IngestResponse,
    LlamaServerSettings, Message, ModelsResponse, QueryEvent, QueryRequest, RagSettings,
    ReindexProgress, RetrievalTraceSummary, SourceChunk,
};
use crate::backend::BackendService;
use async_channel::{Receiver, Sender};
use gpui::prelude::*;
use gpui::{
    div, px, App, ClickEvent, ClipboardItem, Context, Entity, ExternalPaths, FocusHandle,
    Focusable, KeyDownEvent, ParentElement, PathPromptOptions, ScrollHandle, SharedString,
    Subscription, Window,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) mod text_input;
mod theme;
pub use text_input::{
    Backspace, Copy, Cut, Delete, Down, End, Home, Left, Newline, Paste, Right, SelectAll,
    SelectDown, SelectLeft, SelectRight, SelectUp, Submit, Up,
};
use text_input::{TextChanged, TextInput, TextSubmitted};
use theme::*;

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
            .child(
                div()
                    .id("right-panel-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(content),
            )
    }

    fn render_history(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut list = div().flex().flex_col().gap_1();
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
            .child(
                div()
                    .id("history-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(list),
            )
            .child(
                if self
                    .data
                    .conversation
                    .as_ref()
                    .is_some_and(|conversation| conversation.has_more)
                {
                    ui_button(
                        "load-older-messages",
                        "Load older messages",
                        false,
                        cx.listener(|this, _, _, cx| this.load_older_messages(cx)),
                    )
                } else {
                    ui_button(
                        "load-older-messages-disabled",
                        "All messages loaded",
                        false,
                        cx.listener(|_, _, _, _| {}),
                    )
                },
            )
            .child(input_field(
                "rename-chat",
                self.inputs.rename.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::RenameConversation, window, cx)
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
        let mut chunk_preview = div().flex().flex_col().gap_1();
        for (index, chunk) in document.chunk_preview.iter().enumerate() {
            chunk_preview = chunk_preview.child(
                div()
                    .p_2()
                    .bg(panel_2())
                    .border_1()
                    .border_color(line())
                    .rounded_sm()
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(orange_light())
                            .child(format!("Chunk {}", index + 1)),
                    )
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(muted())
                            .child(pretty_value(chunk)),
                    ),
            );
        }
        let document_id = document.id.clone();
        let document_path = document.path.clone();
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
            .child(detail_line(
                "Type",
                document.kind.as_deref().unwrap_or("file"),
                muted(),
            ))
            .child(detail_line("Chunks", &document.chunks.to_string(), text()))
            .child(detail_line(
                "Size",
                &document
                    .size_bytes
                    .map(|size| format_bytes(size))
                    .unwrap_or_else(|| "Unknown".into()),
                muted(),
            ))
            .child(detail_line(
                "Modified",
                &document
                    .modified_at
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "Unknown".into()),
                muted(),
            ))
            .child(detail_line(
                "Last indexed",
                &document
                    .last_indexed_at
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "Unknown".into()),
                muted(),
            ))
            .child(detail_line(
                "Extraction",
                document.extraction_mode.as_deref().unwrap_or("Unknown"),
                muted(),
            ))
            .child(detail_line(
                "Embedding",
                &document
                    .embedding_model_id
                    .clone()
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
            .child(if !document.stale_reasons.is_empty() {
                disclosure_value(
                    "Stale / reindex reasons",
                    &Value::Array(document.stale_reasons.clone()),
                )
            } else {
                div()
            })
            .child(if !document.chunk_preview.is_empty() {
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(muted())
                            .child("Chunk preview"),
                    )
                    .child(chunk_preview)
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
                        self.inputs.tag.clone(),
                        cx.listener(|this, _, window, cx| {
                            this.focus_input(InputTarget::Tag, window, cx)
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
                self.inputs.rename.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::RenameDocument, window, cx)
                }),
            ))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(ui_button(
                        "open-document",
                        "Open",
                        false,
                        cx.listener({
                            let path = document_path.clone();
                            move |this, _, _, cx| this.open_document_path(path.clone(), false, cx)
                        }),
                    ))
                    .child(ui_button(
                        "reveal-document",
                        "Reveal",
                        false,
                        cx.listener({
                            let path = document_path.clone();
                            move |this, _, _, cx| this.open_document_path(path.clone(), true, cx)
                        }),
                    ))
                    .child(ui_button(
                        "save-document-name",
                        "Save name",
                        false,
                        cx.listener(|this, _, _, cx| this.rename_document(cx)),
                    ))
                    .child(ui_button("reindex-document", "Reindex", false, {
                        let id = document_id.clone();
                        cx.listener(move |this, _, _, cx| this.reindex_document(id.clone(), cx))
                    }))
                    .child(ui_button("delete-document", "Delete", false, {
                        let id = document_id.clone();
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
                    "#{} {} · {}",
                    source.rank,
                    source.doc_name,
                    source.source_id.as_deref().unwrap_or("no source id")
                )))
                .child(div().text_size(px(11.)).text_color(muted()).child(format!(
                    "score {:.3} · chunk {}",
                    source.final_score.unwrap_or(source.score),
                    source.chunk_id
                )))
                .child(div().text_color(text()).child(source.snippet.clone()));
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
            if expanded {
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
        div().flex().flex_col().gap_2().flex_1().child(
            div()
                .id("sources-scroll")
                .flex_1()
                .min_h_0()
                .overflow_y_scroll()
                .child(list),
        )
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

    fn render_trace(&mut self, cx: &mut Context<Self>) -> gpui::Div {
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
                cx.listener(|this, _, _, cx| this.refresh_traces(cx)),
            ))
            .child(
                div()
                    .id("trace-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(list),
            )
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
                        "Server URL",
                        llama.server_url.as_deref().unwrap_or("not configured"),
                        muted(),
                    ))
                    .child(detail_line(
                        "Model",
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

    fn render_evaluation(&mut self, cx: &mut Context<Self>) -> gpui::Div {
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
            .child(
                div()
                    .id("evaluation-runs-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(runs),
            )
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
                    .id(SharedString::from(format!("notice-{}", notice.id)))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    RagTopK,
    RagRerankTopN,
    RagMaxTokens,
    RagTemperature,
    RagParentTargetTokens,
    RagParentMaxTokens,
    RagChildTargetTokens,
    RagChildMaxTokens,
    RagChildOverlapTokens,
    RagContextTokens,
    RagMinConfidence,
    RagMinRerankScore,
    RagMinVectorScore,
    RagMinSourceCount,
}

struct InputEntities {
    composer: Entity<TextInput>,
    search: Entity<TextInput>,
    server_url: Entity<TextInput>,
    model_name: Entity<TextInput>,
    context_tokens: Entity<TextInput>,
    eval_question: Entity<TextInput>,
    eval_document: Entity<TextInput>,
    rename: Entity<TextInput>,
    tag: Entity<TextInput>,
    rag_top_k: Entity<TextInput>,
    rag_rerank_top_n: Entity<TextInput>,
    rag_max_tokens: Entity<TextInput>,
    rag_temperature: Entity<TextInput>,
    rag_parent_target_tokens: Entity<TextInput>,
    rag_parent_max_tokens: Entity<TextInput>,
    rag_child_target_tokens: Entity<TextInput>,
    rag_child_max_tokens: Entity<TextInput>,
    rag_child_overlap_tokens: Entity<TextInput>,
    rag_context_tokens: Entity<TextInput>,
    rag_min_confidence: Entity<TextInput>,
    rag_min_rerank_score: Entity<TextInput>,
    rag_min_vector_score: Entity<TextInput>,
    rag_min_source_count: Entity<TextInput>,
}

impl InputEntities {
    fn new(cx: &mut Context<NativeApp>) -> Self {
        Self {
            composer: cx.new(|cx| {
                TextInput::new(
                    cx,
                    "composer-input",
                    "",
                    "Ask Cephalon about your documents…",
                    true,
                )
            }),
            search: cx.new(|cx| TextInput::new(cx, "search-input", "", "Search library…", false)),
            server_url: cx.new(|cx| {
                TextInput::new(cx, "server-url-input", "", "http://127.0.0.1:8080", false)
            }),
            model_name: cx
                .new(|cx| TextInput::new(cx, "model-name-input", "", "Model name", false)),
            context_tokens: cx
                .new(|cx| TextInput::new(cx, "context-tokens-input", "", "Context tokens", false)),
            eval_question: cx.new(|cx| {
                TextInput::new(cx, "eval-question-input", "", "Evaluation question", false)
            }),
            eval_document: cx.new(|cx| {
                TextInput::new(cx, "eval-document-input", "", "Expected document id", false)
            }),
            rename: cx.new(|cx| TextInput::new(cx, "rename-input", "", "Rename", false)),
            tag: cx.new(|cx| TextInput::new(cx, "tag-input", "", "Add a tag", false)),
            rag_top_k: cx.new(|cx| TextInput::new(cx, "rag-top-k-input", "", "Top K", false)),
            rag_rerank_top_n: cx
                .new(|cx| TextInput::new(cx, "rag-rerank-top-n-input", "", "Rerank top N", false)),
            rag_max_tokens: cx.new(|cx| {
                TextInput::new(cx, "rag-max-tokens-input", "", "Answer max tokens", false)
            }),
            rag_temperature: cx
                .new(|cx| TextInput::new(cx, "rag-temperature-input", "", "Temperature", false)),
            rag_parent_target_tokens: cx.new(|cx| {
                TextInput::new(cx, "rag-parent-target-input", "", "Parent target", false)
            }),
            rag_parent_max_tokens: cx
                .new(|cx| TextInput::new(cx, "rag-parent-max-input", "", "Parent max", false)),
            rag_child_target_tokens: cx
                .new(|cx| TextInput::new(cx, "rag-child-target-input", "", "Child target", false)),
            rag_child_max_tokens: cx
                .new(|cx| TextInput::new(cx, "rag-child-max-input", "", "Child max", false)),
            rag_child_overlap_tokens: cx.new(|cx| {
                TextInput::new(cx, "rag-child-overlap-input", "", "Child overlap", false)
            }),
            rag_context_tokens: cx.new(|cx| {
                TextInput::new(
                    cx,
                    "rag-context-tokens-input",
                    "",
                    "Retrieval context",
                    false,
                )
            }),
            rag_min_confidence: cx.new(|cx| {
                TextInput::new(cx, "rag-min-confidence-input", "", "Min confidence", false)
            }),
            rag_min_rerank_score: cx
                .new(|cx| TextInput::new(cx, "rag-min-rerank-input", "", "Min rerank", false)),
            rag_min_vector_score: cx
                .new(|cx| TextInput::new(cx, "rag-min-vector-input", "", "Min vector", false)),
            rag_min_source_count: cx.new(|cx| {
                TextInput::new(
                    cx,
                    "rag-min-source-count-input",
                    "",
                    "Min source count",
                    false,
                )
            }),
        }
    }

    fn get(&self, target: InputTarget) -> Entity<TextInput> {
        match target {
            InputTarget::Composer => self.composer.clone(),
            InputTarget::Search => self.search.clone(),
            InputTarget::ServerUrl => self.server_url.clone(),
            InputTarget::ModelName => self.model_name.clone(),
            InputTarget::ContextTokens => self.context_tokens.clone(),
            InputTarget::EvalQuestion => self.eval_question.clone(),
            InputTarget::EvalDocument => self.eval_document.clone(),
            InputTarget::RenameDocument | InputTarget::RenameConversation => self.rename.clone(),
            InputTarget::Tag => self.tag.clone(),
            InputTarget::RagTopK => self.rag_top_k.clone(),
            InputTarget::RagRerankTopN => self.rag_rerank_top_n.clone(),
            InputTarget::RagMaxTokens => self.rag_max_tokens.clone(),
            InputTarget::RagTemperature => self.rag_temperature.clone(),
            InputTarget::RagParentTargetTokens => self.rag_parent_target_tokens.clone(),
            InputTarget::RagParentMaxTokens => self.rag_parent_max_tokens.clone(),
            InputTarget::RagChildTargetTokens => self.rag_child_target_tokens.clone(),
            InputTarget::RagChildMaxTokens => self.rag_child_max_tokens.clone(),
            InputTarget::RagChildOverlapTokens => self.rag_child_overlap_tokens.clone(),
            InputTarget::RagContextTokens => self.rag_context_tokens.clone(),
            InputTarget::RagMinConfidence => self.rag_min_confidence.clone(),
            InputTarget::RagMinRerankScore => self.rag_min_rerank_score.clone(),
            InputTarget::RagMinVectorScore => self.rag_min_vector_score.clone(),
            InputTarget::RagMinSourceCount => self.rag_min_source_count.clone(),
        }
    }
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
    raw_content: String,
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
            content: visible_answer(&message.content),
            raw_content: message.content.clone(),
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
    reindex_progress: Option<ReindexProgress>,
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
    expanded_sources: std::collections::HashSet<String>,
    selected_support: Option<Value>,
    composer: String,
    retrieval_scope: String,
    response_effort: String,
    response_phase: String,
    messages: Vec<ChatMessage>,
    is_typing: bool,
    query_stop: Option<Arc<AtomicBool>>,
    chat_scroll: ScrollHandle,
    chat_following: bool,
    regenerate_without_user: bool,
    active_input: InputTarget,
    server_url_draft: String,
    model_name_draft: String,
    context_tokens_draft: String,
    eval_question: String,
    eval_document: String,
    rename_draft: String,
    tag_draft: String,
    rag_drafts: std::collections::HashMap<InputTarget, String>,
    notice_counter: u64,
    notices: Vec<Notice>,
    confirmation: Option<Confirmation>,
    inputs: InputEntities,
    input_subscriptions: Vec<Subscription>,
    documents_refresh_generation: u64,
    conversations_refresh_generation: u64,
    settings_refresh_generation: u64,
    server_refresh_generation: u64,
    retrieval_refresh_generation: u64,
    health_refresh_generation: u64,
    eval_refresh_generation: u64,
    traces_refresh_generation: u64,
    conversation_request_generation: u64,
    trace_request_generation: u64,
    query_generation: u64,
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
    reindex_progress: Option<ReindexProgress>,
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
        let inputs = InputEntities::new(cx);
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
            expanded_sources: std::collections::HashSet::new(),
            selected_support: None,
            composer: String::new(),
            retrieval_scope: "medium".into(),
            response_effort: "balanced".into(),
            response_phase: String::new(),
            messages: Vec::new(),
            is_typing: false,
            query_stop: None,
            chat_scroll: ScrollHandle::new(),
            chat_following: true,
            regenerate_without_user: false,
            active_input: InputTarget::Composer,
            server_url_draft: String::new(),
            model_name_draft: String::new(),
            context_tokens_draft: String::new(),
            eval_question: String::new(),
            eval_document: String::new(),
            rename_draft: String::new(),
            tag_draft: String::new(),
            rag_drafts: std::collections::HashMap::new(),
            notice_counter: 0,
            notices: Vec::new(),
            confirmation: None,
            inputs,
            input_subscriptions: Vec::new(),
            documents_refresh_generation: 0,
            conversations_refresh_generation: 0,
            settings_refresh_generation: 0,
            server_refresh_generation: 0,
            retrieval_refresh_generation: 0,
            health_refresh_generation: 0,
            eval_refresh_generation: 0,
            traces_refresh_generation: 0,
            conversation_request_generation: 0,
            trace_request_generation: 0,
            query_generation: 0,
        };
        app.subscribe_input(InputTarget::Composer, cx);
        app.subscribe_input(InputTarget::Search, cx);
        app.subscribe_input(InputTarget::ServerUrl, cx);
        app.subscribe_input(InputTarget::ModelName, cx);
        app.subscribe_input(InputTarget::ContextTokens, cx);
        app.subscribe_input(InputTarget::EvalQuestion, cx);
        app.subscribe_input(InputTarget::EvalDocument, cx);
        app.subscribe_input(InputTarget::RenameConversation, cx);
        app.subscribe_input(InputTarget::Tag, cx);
        for target in [
            InputTarget::RagTopK,
            InputTarget::RagRerankTopN,
            InputTarget::RagMaxTokens,
            InputTarget::RagTemperature,
            InputTarget::RagParentTargetTokens,
            InputTarget::RagParentMaxTokens,
            InputTarget::RagChildTargetTokens,
            InputTarget::RagChildMaxTokens,
            InputTarget::RagChildOverlapTokens,
            InputTarget::RagContextTokens,
            InputTarget::RagMinConfidence,
            InputTarget::RagMinRerankScore,
            InputTarget::RagMinVectorScore,
            InputTarget::RagMinSourceCount,
        ] {
            app.subscribe_input(target, cx);
        }
        app.subscribe_submit(cx);
        app.start_boot(cx);
        app.start_event_stream(cx);
        app
    }

    fn subscribe_input(&mut self, target: InputTarget, cx: &mut Context<Self>) {
        let input = self.inputs.get(target);
        let subscription = cx.subscribe(&input, move |this, entity, _: &TextChanged, cx| {
            let value = entity.read_with(cx, |input, _| input.text().to_owned());
            this.set_input_mirror(target, value);
            cx.notify();
        });
        self.input_subscriptions.push(subscription);
    }

    fn subscribe_submit(&mut self, cx: &mut Context<Self>) {
        let input = self.inputs.composer.clone();
        let subscription = cx.subscribe(&input, |this, _, _: &TextSubmitted, cx| {
            this.send_message(cx);
        });
        self.input_subscriptions.push(subscription);
    }

    fn set_input_mirror(&mut self, target: InputTarget, value: String) {
        match target {
            InputTarget::Composer => self.composer = value,
            InputTarget::Search => self.search = value,
            InputTarget::ServerUrl => self.server_url_draft = value,
            InputTarget::ModelName => self.model_name_draft = value,
            InputTarget::ContextTokens => self.context_tokens_draft = value,
            InputTarget::EvalQuestion => self.eval_question = value,
            InputTarget::EvalDocument => self.eval_document = value,
            InputTarget::RenameDocument | InputTarget::RenameConversation => {
                self.rename_draft = value
            }
            InputTarget::Tag => self.tag_draft = value,
            target => {
                self.rag_drafts.insert(target, value);
            }
        }
    }

    fn set_input_text(
        &mut self,
        target: InputTarget,
        value: impl Into<String>,
        cx: &mut Context<Self>,
    ) {
        let value = value.into();
        self.set_input_mirror(target, value.clone());
        let input = self.inputs.get(target);
        let _ = input.update(cx, |input, cx| input.set_text(value, cx));
    }

    fn rag_input(
        &self,
        target: InputTarget,
        id: &'static str,
        label: &'static str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .child(div().text_size(px(11.)).text_color(muted()).child(label))
            .child(input_field(
                id,
                self.inputs.get(target),
                cx.listener(move |this, _, window, cx| this.focus_input(target, window, cx)),
            ))
    }

    fn sync_rag_inputs(&mut self, settings: &RagSettings, cx: &mut Context<Self>) {
        for target in rag_input_targets() {
            if self.active_input == target {
                continue;
            }
            let value = rag_setting_value(settings, target);
            self.rag_drafts.insert(target, value.clone());
            self.set_input_text(target, value, cx);
        }
    }

    fn start_boot(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        let backend = self.backend.clone();
        let selected_conversation = self.selected_conversation.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || {
                    backend.start().map_err(|message| ApiError {
                        status: None,
                        message,
                    })?;
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
                            EventStreamEvent::Error(message) => {
                                this.event_status = EventStatus::Reconnecting;
                                this.boot_error = Some(message);
                            }
                            event => {
                                this.event_status = EventStatus::Connected;
                                match event.refresh_target() {
                                    Some(EventStreamRefresh::Documents) => {
                                        this.refresh_documents(cx)
                                    }
                                    Some(EventStreamRefresh::Jobs) => {
                                        this.refresh_documents(cx);
                                        this.refresh_reindex_progress(cx);
                                        this.refresh_index_health(cx);
                                    }
                                    Some(EventStreamRefresh::Conversations) => {
                                        this.refresh_conversations(cx)
                                    }
                                    Some(EventStreamRefresh::Settings) => this.refresh_settings(cx),
                                    Some(EventStreamRefresh::LlamaServer) => {
                                        this.refresh_server_and_models(cx)
                                    }
                                    None => {}
                                }
                            }
                        }
                        cx.notify();
                    });
                }
                let _ = this.update(&mut *cx, |this, cx| {
                    this.event_status = EventStatus::Offline;
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_documents(&mut self, cx: &mut Context<Self>) {
        self.documents_refresh_generation = self.documents_refresh_generation.wrapping_add(1);
        let generation = self.documents_refresh_generation;
        let selected_document = self.selected_document.clone();
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || {
                    let documents = api.documents()?;
                    let selected = selected_document
                        .as_deref()
                        .and_then(|id| api.document(id).ok());
                    Ok::<_, ApiError>((documents, selected))
                })
                .await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.documents_refresh_generation {
                        if let Ok((documents, selected)) = result {
                            this.data.documents = documents;
                            if let Some(selected) = selected {
                                if let Some(document) = this
                                    .data
                                    .documents
                                    .iter_mut()
                                    .find(|document| document.id == selected.id)
                                {
                                    *document = selected;
                                }
                            }
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_conversations(&mut self, cx: &mut Context<Self>) {
        self.conversations_refresh_generation =
            self.conversations_refresh_generation.wrapping_add(1);
        let generation = self.conversations_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.conversations()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.conversations_refresh_generation {
                        if let Ok(conversations) = result {
                            this.data.conversations = conversations;
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_retrieval_status(&mut self, cx: &mut Context<Self>) {
        self.retrieval_refresh_generation = self.retrieval_refresh_generation.wrapping_add(1);
        let generation = self.retrieval_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.fixed_retrieval_status()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.retrieval_refresh_generation {
                        if let Ok(retrieval) = result {
                            this.data.retrieval = Some(retrieval);
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_health(&mut self, cx: &mut Context<Self>) {
        self.health_refresh_generation = self.health_refresh_generation.wrapping_add(1);
        let generation = self.health_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.health()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.health_refresh_generation {
                        if let Ok(health) = result {
                            this.data.health = Some(health);
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_eval_runs(&mut self, cx: &mut Context<Self>) {
        self.eval_refresh_generation = self.eval_refresh_generation.wrapping_add(1);
        let generation = self.eval_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.eval_runs()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.eval_refresh_generation {
                        if let Ok(runs) = result {
                            this.data.eval_runs = runs;
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_traces(&mut self, cx: &mut Context<Self>) {
        self.traces_refresh_generation = self.traces_refresh_generation.wrapping_add(1);
        let generation = self.traces_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.retrieval_traces()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.traces_refresh_generation {
                        if let Ok(traces) = result {
                            this.data.traces = traces;
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_settings(&mut self, cx: &mut Context<Self>) {
        self.settings_refresh_generation = self.settings_refresh_generation.wrapping_add(1);
        let generation = self.settings_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.settings()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.settings_refresh_generation {
                        if let Ok(settings) = result {
                            this.data.settings = Some(settings.clone());
                            this.sync_rag_inputs(&settings, cx);
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_server_and_models(&mut self, cx: &mut Context<Self>) {
        self.server_refresh_generation = self.server_refresh_generation.wrapping_add(1);
        let generation = self.server_refresh_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || {
                    let models = api.models()?;
                    let server = api.server_settings().ok();
                    Ok::<_, ApiError>((models, server))
                })
                .await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if generation == this.server_refresh_generation {
                        if let Ok((models, server)) = result {
                            this.data.models = models;
                            if let Some(server) = server {
                                this.data.server = Some(server.clone());
                                if this.active_input != InputTarget::ServerUrl {
                                    this.set_input_text(
                                        InputTarget::ServerUrl,
                                        server.server_url,
                                        cx,
                                    );
                                }
                                if this.active_input != InputTarget::ModelName {
                                    this.set_input_text(
                                        InputTarget::ModelName,
                                        server.model_name,
                                        cx,
                                    );
                                }
                                if this.active_input != InputTarget::ContextTokens {
                                    this.set_input_text(
                                        InputTarget::ContextTokens,
                                        server
                                            .context_tokens
                                            .map(|tokens| tokens.to_string())
                                            .unwrap_or_default(),
                                        cx,
                                    );
                                }
                            }
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_reindex_progress(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.reindex_progress()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if let Ok(progress) = result {
                        this.data.reindex_progress = Some(progress);
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn refresh_index_health(&mut self, cx: &mut Context<Self>) {
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.index_health()).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if let Ok(health) = result {
                        this.data.index_health = Some(health);
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
        if let Some(settings) = self.data.settings.clone() {
            self.sync_rag_inputs(&settings, cx);
        }
        self.data.server = snapshot.server;
        self.data.retrieval = snapshot.retrieval;
        self.data.index_health = snapshot.index_health;
        self.data.traces = snapshot.traces;
        self.data.eval_runs = snapshot.eval_runs;
        self.data.reindex_progress = snapshot.reindex_progress;
        self.data.conversation = snapshot.conversation;
        if let Some(server) = self.data.server.clone() {
            if self.active_input != InputTarget::ServerUrl {
                self.set_input_text(InputTarget::ServerUrl, server.server_url, cx);
            }
            if self.active_input != InputTarget::ModelName {
                self.set_input_text(InputTarget::ModelName, server.model_name, cx);
            }
            if self.active_input != InputTarget::ContextTokens {
                self.set_input_text(
                    InputTarget::ContextTokens,
                    server
                        .context_tokens
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                    cx,
                );
            }
        }
        if self.selected_conversation.is_none() {
            self.selected_conversation = first_conversation;
        }
        let current_conversation = self.selected_conversation.clone();
        if let Some(conversation) = &self.data.conversation {
            if current_conversation
                .as_deref()
                .is_some_and(|id| id != conversation.id)
            {
                self.load_selected_conversation(cx);
                return;
            }
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
        self.conversation_request_generation = self.conversation_request_generation.wrapping_add(1);
        let request_generation = self.conversation_request_generation;
        let expected_id = id.clone();
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.conversation(&id)).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if request_generation == this.conversation_request_generation
                        && this.selected_conversation.as_deref() == Some(expected_id.as_str())
                    {
                        if let Ok(conversation) = result {
                            this.data.conversation = Some(conversation.clone());
                            this.messages = conversation
                                .messages
                                .iter()
                                .map(ChatMessage::from)
                                .collect();
                        }
                    }
                    cx.notify();
                });
            },
        )
        .detach();
    }

    fn load_older_messages(&mut self, cx: &mut Context<Self>) {
        let Some(conversation) = self.data.conversation.as_ref() else {
            return;
        };
        let Some(before) = conversation.next_before else {
            return;
        };
        let id = conversation.id.clone();
        let expected_id = id.clone();
        self.conversation_request_generation = self.conversation_request_generation.wrapping_add(1);
        let request_generation = self.conversation_request_generation;
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result =
                    smol::unblock(move || api.conversation_page(&id, 100, Some(before))).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if request_generation == this.conversation_request_generation
                        && this.selected_conversation.as_deref() == Some(expected_id.as_str())
                    {
                        if let Ok(page) = result {
                            if let Some(current) = this.data.conversation.as_mut() {
                                merge_conversation_messages(&mut current.messages, &page);
                                current.has_more = page.has_more;
                                current.next_before = page.next_before;
                            }
                            if let Some(current) = this.data.conversation.as_ref() {
                                this.messages =
                                    current.messages.iter().map(ChatMessage::from).collect();
                            }
                        }
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

    fn focus_input(&mut self, target: InputTarget, window: &mut Window, cx: &mut Context<Self>) {
        self.active_input = target;
        let input = self.inputs.get(target);
        let _ = input.update(cx, |input, _| {
            window.focus(&input.focus_handle());
        });
    }

    fn handle_key(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
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
    }

    fn retry_backend(&mut self, cx: &mut Context<Self>) {
        self.boot = BootState::Starting;
        self.boot_status = "Retrying local backend…".into();
        self.boot_error = None;
        self.start_boot(cx);
    }

    fn select_conversation(&mut self, id: String, cx: &mut Context<Self>) {
        if self.is_typing {
            self.stop_query(cx);
        }
        let title = self
            .data
            .conversations
            .iter()
            .find(|conversation| conversation.id == id)
            .map(|conversation| conversation.title.clone())
            .unwrap_or_default();
        self.set_input_text(InputTarget::RenameConversation, title, cx);
        self.selected_conversation = Some(id);
        self.panel = Panel::History;
        self.right_open = true;
        self.load_selected_conversation(cx);
        cx.notify();
    }

    fn select_document(&mut self, id: String, cx: &mut Context<Self>) {
        let name = self
            .data
            .documents
            .iter()
            .find(|document| document.id == id)
            .map(|document| document.name.clone())
            .unwrap_or_default();
        self.set_input_text(InputTarget::RenameDocument, name, cx);
        self.selected_document = Some(id);
        self.panel = Panel::Document;
        self.right_open = true;
        cx.notify();
    }

    fn open_document_path(&mut self, path: String, reveal: bool, cx: &mut Context<Self>) {
        match open_path_on_disk(&path, reveal) {
            Ok(()) => self.notify(
                if reveal {
                    "Document location opened."
                } else {
                    "Document opened."
                },
                green(),
                cx,
            ),
            Err(error) => self.notify(error, red(), cx),
        }
    }

    fn open_document_by_id(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(path) = self
            .data
            .documents
            .iter()
            .find(|document| document.id == id)
            .map(|document| document.path.clone())
        else {
            self.notify(
                "The source document is no longer in the library.",
                yellow(),
                cx,
            );
            return;
        };
        self.open_document_path(path, false, cx);
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
                        this.refresh_conversations(cx);
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
        self.query_generation = self.query_generation.wrapping_add(1);
        let request_generation = self.query_generation;
        let regenerate = self.regenerate_without_user;
        self.regenerate_without_user = false;
        self.update_chat_following();
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
        if !regenerate {
            self.messages.push(ChatMessage {
                id: None,
                role: "user".into(),
                content: prompt.clone(),
                raw_content: prompt.clone(),
                sources: Vec::new(),
                support: None,
                streaming: false,
                error: false,
            });
        }
        self.messages.push(ChatMessage {
            id: Some(assistant_id),
            role: "assistant".into(),
            content: String::new(),
            raw_content: String::new(),
            sources: Vec::new(),
            support: None,
            streaming: true,
            error: false,
        });
        self.set_input_text(InputTarget::Composer, "", cx);
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
                        if request_generation == this.query_generation {
                            this.apply_query_event(event, cx);
                        }
                    });
                    if terminal {
                        break;
                    }
                }
                let _ = this.update(&mut *cx, |this, cx| {
                    if request_generation == this.query_generation {
                        this.is_typing = false;
                        this.query_stop = None;
                        this.response_phase.clear();
                        this.refresh_conversations(cx);
                        this.load_selected_conversation(cx);
                    }
                    cx.notify();
                });
            },
        )
        .detach();
        cx.notify();
    }

    fn apply_query_event(&mut self, event: QueryEvent, cx: &mut Context<Self>) {
        self.update_chat_following();
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        match event {
            QueryEvent::Phase(phase) => self.response_phase = phase_label(&phase).into(),
            QueryEvent::Token(text) => {
                last.raw_content.push_str(&text);
                last.content = visible_answer(&last.raw_content);
                if self.chat_following {
                    self.chat_scroll.scroll_to_bottom();
                }
            }
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
                last.raw_content = message.clone();
                last.content = message;
                last.error = true;
                last.streaming = false;
            }
            QueryEvent::Done => last.streaming = false,
        }
        cx.notify();
    }

    fn stop_query(&mut self, cx: &mut Context<Self>) {
        self.query_generation = self.query_generation.wrapping_add(1);
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

    fn update_chat_following(&mut self) {
        let max_offset = self.chat_scroll.max_offset().height;
        let remaining = max_offset + self.chat_scroll.offset().y;
        self.chat_following = max_offset <= px(1.) || remaining <= px(24.);
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
                        this.notify(ingestion_notice(&response), green(), cx);
                        this.refresh_documents(cx);
                        this.refresh_reindex_progress(cx);
                        this.refresh_index_health(cx);
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
                    this.refresh_documents(cx);
                    this.refresh_reindex_progress(cx);
                    this.refresh_index_health(cx);
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
                                this.refresh_documents(cx);
                                this.refresh_reindex_progress(cx);
                                this.refresh_index_health(cx);
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
                                this.refresh_conversations(cx);
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
                                this.refresh_retrieval_status(cx);
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
                        this.refresh_server_and_models(cx);
                        this.refresh_health(cx);
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
                        let model = response
                            .active_model
                            .unwrap_or_else(|| "llama.cpp server".into());
                        let backend = response
                            .llama_backend
                            .as_ref()
                            .and_then(|status| status.backend_label.as_deref())
                            .unwrap_or("external");
                        let context = response
                            .active_context_tokens
                            .map(|tokens| format!(", {tokens} context tokens"))
                            .unwrap_or_default();
                        this.notify(
                            format!("Connected to {model} via {backend}{context}."),
                            green(),
                            cx,
                        );
                        this.refresh_server_and_models(cx);
                        this.refresh_health(cx);
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
                        this.refresh_documents(cx);
                        this.refresh_reindex_progress(cx);
                        this.refresh_index_health(cx);
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
                        this.refresh_eval_runs(cx);
                    }
                    Err(error) => this.notify(error.to_string(), red(), cx),
                });
            },
        )
        .detach();
    }

    fn load_trace(&mut self, id: String, cx: &mut Context<Self>) {
        self.data.selected_trace = None;
        self.trace_request_generation = self.trace_request_generation.wrapping_add(1);
        let request_generation = self.trace_request_generation;
        let expected_id = id.clone();
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.retrieval_trace(&id)).await;
                let _ = this.update(&mut *cx, |this, cx| {
                    if request_generation == this.trace_request_generation
                        && this
                            .data
                            .traces
                            .iter()
                            .any(|trace| trace.query_id == expected_id)
                    {
                        if let Ok(trace) = result {
                            this.data.selected_trace = Some(trace);
                        }
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
        if let Some(query_stop) = &self.query_stop {
            query_stop.store(true, Ordering::Relaxed);
        }
        self.backend.shutdown();
    }
}

impl NativeApp {
    fn set_theme(&mut self, graphite: bool, cx: &mut Context<Self>) {
        self.theme_graphite = graphite;
        theme::set_graphite(graphite);
        cx.notify();
    }

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
        let Some(mut settings) = self.data.settings.clone() else {
            return;
        };
        apply_rag_drafts(&mut settings, &self.rag_drafts);
        self.data.settings = Some(settings.clone());
        let api = self.api.clone();
        cx.spawn(
            async move |this: gpui::WeakEntity<NativeApp>, cx: &mut gpui::AsyncApp| {
                let result = smol::unblock(move || api.update_settings(&settings)).await;
                let _ = this.update(cx, |this, cx| match result {
                    Ok(settings) => {
                        this.data.settings = Some(settings.clone());
                        this.sync_rag_inputs(&settings, cx);
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
                    Ok(response) => {
                        this.notify(
                            if response.status.is_empty() {
                                format!("Queued {} document(s) for reindexing.", response.total)
                            } else {
                                format!(
                                    "{} · queued {} document(s) for reindexing.",
                                    response.status, response.total
                                )
                            },
                            green(),
                            cx,
                        );
                        this.refresh_documents(cx);
                        this.refresh_reindex_progress(cx);
                        this.refresh_index_health(cx);
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
                        this.set_input_text(InputTarget::RenameDocument, document.name.clone(), cx);
                        this.notify("Document renamed.", green(), cx);
                        this.refresh_documents(cx);
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
                        this.set_input_text(
                            InputTarget::RenameConversation,
                            conversation.title.clone(),
                            cx,
                        );
                        this.data.conversation = Some(conversation);
                        this.notify("Chat renamed.", green(), cx);
                        this.refresh_conversations(cx);
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
                        this.set_input_text(InputTarget::Tag, "", cx);
                        this.notify("Tag added.", green(), cx);
                        this.refresh_documents(cx);
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
                    Ok(_) => this.refresh_documents(cx),
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
                        this.refresh_retrieval_status(cx);
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
        let mut body = div().flex().flex_1().min_h_0();
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
        let mut list = div().flex().flex_col().gap_1();
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
                self.inputs.search.clone(),
                cx.listener(|this, _, window, cx| {
                    this.focus_input(InputTarget::Search, window, cx)
                }),
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
            .child(
                div()
                    .id("library-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(list),
            )
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

    fn render_answer_content(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let mut content = div().flex().flex_col().gap_1();
        let answer = if message.content.is_empty() && message.streaming {
            "…"
        } else {
            message.content.as_str()
        };
        for (line_index, line) in answer.lines().enumerate() {
            let heading =
                line.starts_with("# ") || line.starts_with("## ") || line.starts_with("### ");
            let line_color = if heading { orange_light() } else { text() };
            let mut row = div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_1()
                .text_color(line_color);
            let mut cursor = 0;
            let mut citation_index = 0;
            while let Some(relative_start) = line[cursor..].find("[[src:") {
                let start = cursor + relative_start;
                let Some(relative_end) = line[start..].find("]]") else {
                    break;
                };
                let end = start + relative_end + 2;
                if start > cursor {
                    row = row.child(line[cursor..start].to_string());
                }
                let marker = line[start + "[[src:".len()..end - 2].to_string();
                let sources = message.sources.clone();
                let citation = marker.clone();
                row = row.child(ui_button(
                    format!("citation-{message_index}-{line_index}-{citation_index}"),
                    format!("[{marker}]"),
                    false,
                    cx.listener(move |this, _, _, cx| {
                        this.open_source_citation(citation.clone(), sources.clone(), cx)
                    }),
                ));
                citation_index += 1;
                cursor = end;
            }
            if cursor < line.len() {
                row = row.child(line[cursor..].to_string());
            } else if line.is_empty() {
                row = row.child(" ");
            }
            content = content.child(row);
        }
        if answer.is_empty() {
            content = content.child(div().text_color(faint()).child(" "));
        }
        content
    }

    fn open_source_citation(
        &mut self,
        citation: String,
        sources: Vec<SourceChunk>,
        cx: &mut Context<Self>,
    ) {
        let marker = citation.strip_prefix("src:").unwrap_or(&citation);
        if let Some(source) = sources.iter().find(|source| {
            source.source_id.as_deref() == Some(marker)
                || source.chunk_id == marker
                || format!("S{}", source.rank) == marker
        }) {
            self.expanded_sources.insert(source_key(source));
            self.selected_sources = sources;
            self.choose_panel(Panel::Sources, cx);
        } else {
            self.notify(
                format!("Citation {marker} is not present in this answer's sources."),
                yellow(),
                cx,
            );
        }
    }

    fn regenerate_message(
        &mut self,
        message_index: usize,
        _message: &ChatMessage,
        cx: &mut Context<Self>,
    ) {
        if self.is_typing {
            return;
        }
        let Some(prompt) = self.messages[..message_index]
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .map(|message| message.raw_content.clone())
        else {
            return;
        };
        self.messages.truncate(message_index);
        self.set_input_text(InputTarget::Composer, prompt, cx);
        self.regenerate_without_user = true;
        self.send_message(cx);
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
        let mut messages = div().flex().flex_col().gap_3().p_5();
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
        for index in 0..self.messages.len() {
            let message = self.messages[index].clone();
            let user = message.role == "user";
            let message_key = message
                .id
                .clone()
                .unwrap_or_else(|| format!("draft-{index}"));
            let content = if message.content.is_empty() && message.streaming {
                "…".to_string()
            } else {
                message.content.clone()
            };
            let mut card = div()
                .id(SharedString::from(format!("message-{message_key}")))
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
                        .child(if user {
                            div().child(content)
                        } else {
                            self.render_answer_content(index, &message, cx)
                        }),
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
            if !user && !message.streaming {
                let copy_message = message.clone();
                let regenerate_message = message.clone();
                card = card.child(
                    div()
                        .flex()
                        .gap_1()
                        .child(ui_button(
                            format!("message-copy-{index}"),
                            "Copy answer",
                            false,
                            cx.listener(move |_, _, _, cx| {
                                cx.write_to_clipboard(ClipboardItem::new_string(visible_answer(
                                    &copy_message.raw_content,
                                )));
                            }),
                        ))
                        .child(ui_button(
                            format!("message-regenerate-{index}"),
                            "Regenerate",
                            false,
                            cx.listener(move |this, _, _, cx| {
                                this.regenerate_message(index, &regenerate_message, cx)
                            }),
                        )),
                );
            }
            messages = messages.child(card);
        }
        let composer = input_field(
            "composer",
            self.inputs.composer.clone(),
            cx.listener(|this, _, window, cx| this.focus_input(InputTarget::Composer, window, cx)),
        );
        div()
            .flex()
            .flex_col()
            .flex_1()
            .h_full()
            .min_h_0()
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
            .child(
                div()
                    .id("chat-message-scroll")
                    .flex_1()
                    .h(px(0.))
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.chat_scroll)
                    .child(messages),
            )
            .child(
                div()
                    .p_4()
                    .flex()
                    .flex_col()
                    .flex_none()
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
    input: Entity<TextInput>,
    listener: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<gpui::Div> {
    div().id(id.into()).w_full().child(input).on_click(listener)
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

fn ingestion_notice(response: &IngestResponse) -> String {
    let message = response.message.as_deref().unwrap_or("Ingestion queued.");
    if response.job_id.is_empty() {
        if response.status.is_empty() {
            message.to_string()
        } else {
            format!("{message} · {}", response.status)
        }
    } else {
        format!(
            "{message} · {} · job {}",
            if response.status.is_empty() {
                "queued"
            } else {
                response.status.as_str()
            },
            response.job_id
        )
    }
}

fn rag_input_targets() -> [InputTarget; 14] {
    [
        InputTarget::RagTopK,
        InputTarget::RagRerankTopN,
        InputTarget::RagMaxTokens,
        InputTarget::RagTemperature,
        InputTarget::RagParentTargetTokens,
        InputTarget::RagParentMaxTokens,
        InputTarget::RagChildTargetTokens,
        InputTarget::RagChildMaxTokens,
        InputTarget::RagChildOverlapTokens,
        InputTarget::RagContextTokens,
        InputTarget::RagMinConfidence,
        InputTarget::RagMinRerankScore,
        InputTarget::RagMinVectorScore,
        InputTarget::RagMinSourceCount,
    ]
}

fn rag_setting_value(settings: &RagSettings, target: InputTarget) -> String {
    match target {
        InputTarget::RagTopK => settings.top_k.to_string(),
        InputTarget::RagRerankTopN => settings.rerank_top_n.to_string(),
        InputTarget::RagMaxTokens => settings.max_tokens.to_string(),
        InputTarget::RagTemperature => settings.temperature.to_string(),
        InputTarget::RagParentTargetTokens => settings.parent_target_tokens.to_string(),
        InputTarget::RagParentMaxTokens => settings.parent_max_tokens.to_string(),
        InputTarget::RagChildTargetTokens => settings.child_target_tokens.to_string(),
        InputTarget::RagChildMaxTokens => settings.child_max_tokens.to_string(),
        InputTarget::RagChildOverlapTokens => settings.child_overlap_tokens.to_string(),
        InputTarget::RagContextTokens => settings.context_tokens.to_string(),
        InputTarget::RagMinConfidence => settings.no_answer_min_confidence.to_string(),
        InputTarget::RagMinRerankScore => settings.no_answer_min_rerank_score.to_string(),
        InputTarget::RagMinVectorScore => settings.no_answer_min_vector_score.to_string(),
        InputTarget::RagMinSourceCount => settings.no_answer_min_source_count.to_string(),
        _ => String::new(),
    }
}

fn apply_rag_drafts(
    settings: &mut RagSettings,
    drafts: &std::collections::HashMap<InputTarget, String>,
) {
    let integer = |target| {
        drafts
            .get(&target)
            .and_then(|value| value.trim().parse().ok())
    };
    let decimal = |target| {
        drafts
            .get(&target)
            .and_then(|value| value.trim().parse().ok())
    };
    if let Some(value) = integer(InputTarget::RagTopK) {
        settings.top_k = value;
    }
    if let Some(value) = integer(InputTarget::RagRerankTopN) {
        settings.rerank_top_n = value;
    }
    if let Some(value) = integer(InputTarget::RagMaxTokens) {
        settings.max_tokens = value;
    }
    if let Some(value) = decimal(InputTarget::RagTemperature) {
        settings.temperature = value;
    }
    if let Some(value) = integer(InputTarget::RagParentTargetTokens) {
        settings.parent_target_tokens = value;
    }
    if let Some(value) = integer(InputTarget::RagParentMaxTokens) {
        settings.parent_max_tokens = value;
    }
    if let Some(value) = integer(InputTarget::RagChildTargetTokens) {
        settings.child_target_tokens = value;
    }
    if let Some(value) = integer(InputTarget::RagChildMaxTokens) {
        settings.child_max_tokens = value;
    }
    if let Some(value) = integer(InputTarget::RagChildOverlapTokens) {
        settings.child_overlap_tokens = value;
    }
    if let Some(value) = integer(InputTarget::RagContextTokens) {
        settings.context_tokens = value;
    }
    if let Some(value) = decimal(InputTarget::RagMinConfidence) {
        settings.no_answer_min_confidence = value;
    }
    if let Some(value) = decimal(InputTarget::RagMinRerankScore) {
        settings.no_answer_min_rerank_score = value;
    }
    if let Some(value) = decimal(InputTarget::RagMinVectorScore) {
        settings.no_answer_min_vector_score = value;
    }
    if let Some(value) = integer(InputTarget::RagMinSourceCount) {
        settings.no_answer_min_source_count = value;
    }
}

fn pretty_value(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn source_key(source: &SourceChunk) -> String {
    source
        .source_id
        .clone()
        .or_else(|| (!source.chunk_id.is_empty()).then(|| source.chunk_id.clone()))
        .unwrap_or_else(|| format!("rank-{}", source.rank))
}

fn score_badge(label: &str, value: Option<f64>) -> gpui::Div {
    div()
        .px_1()
        .py(px(1.))
        .bg(panel_3())
        .rounded_sm()
        .text_size(px(10.))
        .text_color(muted())
        .child(format!(
            "{label} {}",
            value
                .map(|score| format!("{score:.3}"))
                .unwrap_or_else(|| "–".into())
        ))
}

fn disclosure_text(title: &str, value: &str) -> gpui::Div {
    div()
        .p_2()
        .bg(panel_3())
        .border_1()
        .border_color(line())
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(11.))
                .text_color(orange_light())
                .child(title.to_string()),
        )
        .child(
            div()
                .text_size(px(11.))
                .text_color(muted())
                .child(value.to_string()),
        )
}

fn disclosure_value(title: &str, value: &Value) -> gpui::Div {
    disclosure_text(title, &pretty_value(value))
}

fn open_path_on_disk(path: &str, reveal: bool) -> Result<(), String> {
    let path = std::path::Path::new(path);
    if path.as_os_str().is_empty() {
        return Err("The document has no local path.".into());
    }
    #[cfg(windows)]
    {
        let mut command = if reveal {
            let mut command = std::process::Command::new("explorer.exe");
            command.arg(format!("/select,{}", path.display()));
            command
        } else {
            let mut command = std::process::Command::new("cmd.exe");
            command.args(["/C", "start", "", &path.to_string_lossy()]);
            command
        };
        command
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not open the document: {error}"))
    }
    #[cfg(target_os = "macos")]
    {
        let mut command = std::process::Command::new("open");
        if reveal {
            command.arg("-R");
        }
        command
            .arg(path)
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not open the document: {error}"))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let target = if reveal {
            path.parent().unwrap_or(path)
        } else {
            path
        };
        std::process::Command::new("xdg-open")
            .arg(target)
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not open the document: {error}"))
    }
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
    let reindex_progress = api.reindex_progress().ok();
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
        reindex_progress,
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

fn visible_answer(raw: &str) -> String {
    let mut visible = String::with_capacity(raw.len());
    let mut cursor = 0;
    while let Some(relative_start) = raw[cursor..].find("<think>") {
        let start = cursor + relative_start;
        visible.push_str(&raw[cursor..start]);
        let content_start = start + "<think>".len();
        let Some(relative_end) = raw[content_start..].find("</think>") else {
            return visible;
        };
        cursor = content_start + relative_end + "</think>".len();
    }
    visible.push_str(&raw[cursor..]);
    visible
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
            .on_key_down(
                cx.listener(|this, event: &KeyDownEvent, _, cx| this.handle_key(event, cx)),
            )
            .on_drop::<ExternalPaths>(
                cx.listener(|this, paths, _, cx| this.ingest_dropped(paths, cx)),
            );
        root
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_rag_drafts, visible_answer, InputTarget};
    use crate::api::RagSettings;
    use std::collections::HashMap;

    #[test]
    fn settings_drafts_change_selected_fields_without_resetting_the_rest() {
        let mut settings = RagSettings {
            top_k: 8,
            parent_target_tokens: 512,
            parent_max_tokens: 1024,
            child_target_tokens: 160,
            child_max_tokens: 320,
            child_overlap_tokens: 32,
            temperature: 0.2,
            evidence_required: true,
            ..RagSettings::default()
        };
        let untouched = settings.clone();
        let drafts = HashMap::from([
            (InputTarget::RagTopK, "24".to_string()),
            (InputTarget::RagParentTargetTokens, "768".to_string()),
            (InputTarget::RagTemperature, "0.35".to_string()),
            (InputTarget::RagChildMaxTokens, "not-a-number".to_string()),
        ]);
        apply_rag_drafts(&mut settings, &drafts);
        assert_eq!(settings.top_k, 24);
        assert_eq!(settings.parent_target_tokens, 768);
        assert_eq!(settings.temperature, 0.35);
        assert_eq!(settings.parent_max_tokens, untouched.parent_max_tokens);
        assert_eq!(settings.child_max_tokens, untouched.child_max_tokens);
        assert_eq!(settings.evidence_required, untouched.evidence_required);
    }

    #[test]
    fn hidden_thinking_is_removed_from_streamed_and_saved_answers() {
        assert_eq!(
            visible_answer("before<think>private reasoning</think>after"),
            "beforeafter"
        );
        assert_eq!(visible_answer("<think>still private"), "");
        assert_eq!(visible_answer("a<think>x</think>b<think>y</think>c"), "abc");
    }
}
