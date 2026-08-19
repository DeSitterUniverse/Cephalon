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
    div, px, App, ClickEvent, Context, Entity, ExternalPaths, FocusHandle, Focusable, KeyDownEvent,
    ParentElement, PathPromptOptions, ScrollHandle, SharedString, Subscription, Window,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod app;
mod chat;
mod diagnostics;
mod document;
mod history;
mod layout;
mod library;
mod markdown;
mod settings;
mod sources;
pub(crate) mod text_input;
mod theme;
use layout::{compact_navigation, layout_mode, LayoutMode};
use markdown::visible_answer;
pub use text_input::{CutSelectionOnly, FocusNextInput, FocusPreviousInput, Submit};
use text_input::{TextChanged, TextInput, TextSubmitted};
use theme::*;

fn selected_request_is_current(
    request_generation: u64,
    current_generation: u64,
    expected_id: &str,
    selected_id: Option<&str>,
) -> bool {
    request_generation == current_generation && selected_id == Some(expected_id)
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
    error_detail: Option<String>,
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
            error_detail: None,
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
                    if selected_request_is_current(
                        request_generation,
                        this.conversation_request_generation,
                        expected_id.as_str(),
                        this.selected_conversation.as_deref(),
                    ) {
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
                    if selected_request_is_current(
                        request_generation,
                        this.conversation_request_generation,
                        expected_id.as_str(),
                        this.selected_conversation.as_deref(),
                    ) {
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
        let focus_handle = input.read(cx).focus_handle();
        window.focus(&focus_handle, cx);
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
        self.left_open = false;
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
                error_detail: None,
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
            error_detail: None,
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
                let mut failed = false;
                while let Ok(event) = rx.recv().await {
                    let terminal = matches!(event, QueryEvent::Done | QueryEvent::Error(_));
                    failed = matches!(event, QueryEvent::Error(_));
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
                        // Keep a transport/retrieval error visible in the current
                        // draft instead of immediately replacing it with the last
                        // persisted conversation. Successful streams are persisted
                        // by the backend and can safely reload their conversation.
                        if !failed {
                            this.load_selected_conversation(cx);
                        }
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
                if last.content.is_empty() {
                    last.raw_content = message.clone();
                    last.content = message;
                } else {
                    last.error_detail = Some(message);
                }
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
        let max_offset = self.chat_scroll.max_offset().y;
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
                    let trace_is_available = this
                        .data
                        .traces
                        .iter()
                        .any(|trace| trace.query_id == expected_id);
                    if selected_request_is_current(
                        request_generation,
                        this.trace_request_generation,
                        expected_id.as_str(),
                        trace_is_available.then_some(expected_id.as_str()),
                    ) {
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
            "auto" => "off",
            "off" => "low",
            "low" => "medium",
            "medium" => "high",
            _ => "auto",
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
                .flex_1()
                .overflow_hidden()
                .truncate()
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

fn clamp_text(value: &str, max_chars: usize) -> String {
    let mut text = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        text.push('…');
    }
    text
}

fn value_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value.get(*key).and_then(|value| match value {
            Value::String(value) if !value.is_empty() => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            _ => None,
        })
    })
}

fn diagnostic_value(title: &str, value: &Value) -> gpui::Div {
    let mut panel = div()
        .flex()
        .flex_col()
        .gap_1()
        .p_2()
        .bg(panel_3())
        .border_1()
        .border_color(line())
        .rounded_sm()
        .child(
            div()
                .text_size(px(11.))
                .text_color(orange_light())
                .child(title.to_string()),
        );

    if let Value::Object(fields) = value {
        let mut displayed = 0;
        for (key, field) in fields {
            if let Some(summary) = diagnostic_summary(field) {
                panel = panel.child(detail_line(&humanize_key(key), &summary, muted()));
                displayed += 1;
            }
            if displayed >= 12 {
                break;
            }
        }
        if displayed == 0 {
            panel = panel.child(
                div()
                    .text_size(px(11.))
                    .text_color(muted())
                    .child("Structured details are available below."),
            );
        }
    } else if let Some(summary) = diagnostic_summary(value) {
        panel = panel.child(detail_line("Value", &summary, muted()));
    }

    panel.child(disclosure_value("Raw details", value))
}

fn diagnostic_summary(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) if value.is_empty() => None,
        Value::String(value) => Some(clamp_text(value, 320)),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Array(values) => Some(format!("{} items", values.len())),
        Value::Object(fields) => Some(format!("{} fields", fields.len())),
    }
}

fn humanize_key(key: &str) -> String {
    key.replace(['_', '-'], " ")
        .split_whitespace()
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_chunk_preview(index: usize, chunk: &Value) -> gpui::Div {
    let kind = value_string(chunk, &["block_type", "source_kind", "type"])
        .unwrap_or_else(|| "paragraph".into());
    let tokens = value_string(chunk, &["token_count", "tokens"])
        .map(|value| format!(" · {value} tokens"))
        .unwrap_or_default();
    let section = value_string(chunk, &["section_heading", "section", "heading"]);
    let chunk_text = value_string(chunk, &["text", "content", "chunk", "raw_text"])
        .unwrap_or_else(|| clamp_text(&pretty_value(chunk), 560));
    let mut card = div()
        .p_2()
        .bg(panel_2())
        .border_1()
        .border_color(line())
        .rounded_sm()
        .child(
            div()
                .text_size(px(11.))
                .text_color(orange_light())
                .child(format!("Chunk {} · {}{}", index + 1, kind, tokens)),
        );
    if let Some(section) = section {
        card = card.child(detail_line("Section", &section, muted()));
    }
    card.child(
        div()
            .text_size(px(11.))
            .text_color(text())
            .child(chunk_text),
    )
    .child(disclosure_value("Raw chunk details", chunk))
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

fn phase_label(phase: &str) -> &'static str {
    if phase == "Connecting…" {
        "Connecting…"
    } else {
        response_phase_label(phase)
    }
}

impl Render for NativeApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut root = if self.boot == BootState::Ready {
            self.render_shell(window, cx)
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
    use super::{apply_rag_drafts, selected_request_is_current, visible_answer, InputTarget};
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

    #[test]
    fn stale_selection_requests_cannot_replace_newer_results() {
        assert!(selected_request_is_current(4, 4, "new", Some("new")));
        assert!(!selected_request_is_current(3, 4, "old", Some("old")));
        assert!(!selected_request_is_current(4, 4, "old", Some("new")));
        assert!(!selected_request_is_current(4, 4, "missing", None));
    }
}
