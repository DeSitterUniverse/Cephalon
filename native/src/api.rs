use reqwest::blocking::{Client, Response};
use reqwest::Method;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt::{Display, Formatter};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: Option<u16>,
    pub message: String,
}

impl Display for ApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(status) = self.status {
            write!(f, "HTTP {status}: {}", self.message)
        } else {
            f.write_str(&self.message)
        }
    }
}

impl std::error::Error for ApiError {}

impl From<reqwest::Error> for ApiError {
    fn from(error: reqwest::Error) -> Self {
        Self {
            status: error.status().map(|status| status.as_u16()),
            message: error.to_string(),
        }
    }
}

impl From<std::io::Error> for ApiError {
    fn from(error: std::io::Error) -> Self {
        Self {
            status: None,
            message: error.to_string(),
        }
    }
}

#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    client: Client,
}

impl ApiClient {
    pub fn configured() -> Self {
        let base_url = std::env::var("CEPHALON_API_URL").unwrap_or_else(|_| {
            let host = std::env::var("CEPHALON_HOST").unwrap_or_else(|_| "127.0.0.1".into());
            let port = std::env::var("CEPHALON_PORT").unwrap_or_else(|_| "8765".into());
            format!("http://{host}:{port}")
        });
        Self::new(base_url)
    }

    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Client::builder()
                .connect_timeout(Duration::from_millis(800))
                // Local retrieval and CPU reranking can take longer than a
                // normal request, while the backend's model call is bounded
                // at five minutes.
                .timeout(Duration::from_secs(300))
                .build()
                .expect("reqwest client configuration is valid"),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn request_json<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<T, ApiError> {
        let mut request = self.client.request(method, self.url(path));
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send()?;
        parse_json_response(response)
    }

    fn request_value(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, ApiError> {
        self.request_json(method, path, body)
    }

    pub fn health(&self) -> Result<HealthResponse, ApiError> {
        self.request_json(Method::GET, "/health", None)
    }

    pub fn models(&self) -> Result<ModelsResponse, ApiError> {
        self.request_json(Method::GET, "/models", None)
    }

    pub fn load_model(&self) -> Result<LoadModelResponse, ApiError> {
        self.request_json(Method::POST, "/models/load", Some(json!({"model": ""})))
    }

    pub fn server_settings(&self) -> Result<LlamaServerSettings, ApiError> {
        self.request_json(Method::GET, "/models/server", None)
    }

    pub fn update_server_settings(
        &self,
        settings: &LlamaServerSettings,
    ) -> Result<LlamaServerSettings, ApiError> {
        self.request_json(
            Method::PUT,
            "/models/server",
            Some(serde_json::to_value(settings).expect("server settings serialize")),
        )
    }

    pub fn fixed_retrieval_status(&self) -> Result<FixedRetrievalStatus, ApiError> {
        self.request_json(Method::GET, "/models/status", None)
    }

    pub fn download_fixed_model(&self, kind: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::POST,
            "/models/download",
            Some(json!({"kind": kind})),
        )
    }

    pub fn verify_fixed_model(&self, kind: &str) -> Result<FixedModelInfo, ApiError> {
        self.request_json(Method::POST, "/models/verify", Some(json!({"kind": kind})))
    }

    pub fn delete_fixed_model(&self, kind: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::POST,
            "/models/delete",
            Some(json!({"kind": kind, "confirmed": true})),
        )
    }

    pub fn open_fixed_model_directory(&self, kind: &str) -> Result<Value, ApiError> {
        self.request_value(Method::POST, "/models/open", Some(json!({"kind": kind})))
    }

    pub fn reindex_progress(&self) -> Result<ReindexProgress, ApiError> {
        self.request_json(Method::GET, "/reindex/progress", None)
    }

    pub fn reindex_all(&self) -> Result<ReindexResponse, ApiError> {
        self.request_json(Method::POST, "/reindex/full", None)
    }

    pub fn reindex_stale(&self) -> Result<ReindexResponse, ApiError> {
        self.request_json(Method::POST, "/reindex/stale", None)
    }

    pub fn documents(&self) -> Result<Vec<Document>, ApiError> {
        let response: DocumentsResponse = self.request_json(Method::GET, "/documents", None)?;
        Ok(response.documents)
    }

    pub fn document(&self, id: &str) -> Result<Document, ApiError> {
        self.request_json(
            Method::GET,
            &format!("/documents/{}", path_segment(id)),
            None,
        )
    }

    pub fn rename_document(&self, id: &str, display_name: &str) -> Result<Document, ApiError> {
        self.request_json(
            Method::PATCH,
            &format!("/documents/{}", path_segment(id)),
            Some(json!({"display_name": display_name})),
        )
    }

    pub fn ingest_path(&self, path: &str, force_text: bool) -> Result<IngestResponse, ApiError> {
        self.request_json(
            Method::POST,
            "/ingest",
            Some(json!({"path": path, "force_text": force_text})),
        )
    }

    pub fn reindex_document(&self, id: &str) -> Result<IngestResponse, ApiError> {
        self.request_json(
            Method::POST,
            &format!("/documents/{}/reindex", path_segment(id)),
            None,
        )
    }

    pub fn delete_document(&self, id: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::DELETE,
            &format!("/documents/{}", path_segment(id)),
            None,
        )
    }

    pub fn add_document_tag(&self, id: &str, tag: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::POST,
            &format!("/documents/{}/tags", path_segment(id)),
            Some(json!({"tag": tag})),
        )
    }

    pub fn delete_document_tag(&self, id: &str, tag: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::DELETE,
            &format!("/documents/{}/tags/{}", path_segment(id), path_segment(tag)),
            None,
        )
    }

    pub fn conversations(&self) -> Result<Vec<Conversation>, ApiError> {
        let response: ConversationsResponse =
            self.request_json(Method::GET, "/conversations", None)?;
        Ok(response.conversations)
    }

    pub fn create_conversation(&self) -> Result<Conversation, ApiError> {
        self.request_json(Method::POST, "/conversations", None)
    }

    pub fn conversation(&self, id: &str) -> Result<Conversation, ApiError> {
        self.conversation_page(id, 100, None)
    }

    pub fn conversation_page(
        &self,
        id: &str,
        limit: i64,
        before: Option<i64>,
    ) -> Result<Conversation, ApiError> {
        let limit = limit.clamp(1, 200);
        let query = before
            .map(|cursor| format!("?limit={limit}&before={cursor}"))
            .unwrap_or_else(|| format!("?limit={limit}"));
        self.request_json(
            Method::GET,
            &format!("/conversations/{}{}", path_segment(id), query),
            None,
        )
    }

    pub fn rename_conversation(&self, id: &str, title: &str) -> Result<Conversation, ApiError> {
        self.request_json(
            Method::PATCH,
            &format!("/conversations/{}", path_segment(id)),
            Some(json!({"title": title})),
        )
    }

    pub fn delete_conversation(&self, id: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::DELETE,
            &format!("/conversations/{}", path_segment(id)),
            None,
        )
    }

    pub fn settings(&self) -> Result<RagSettings, ApiError> {
        self.request_json(Method::GET, "/settings", None)
    }

    pub fn update_settings(&self, settings: &RagSettings) -> Result<RagSettings, ApiError> {
        self.request_json(
            Method::PUT,
            "/settings",
            Some(serde_json::to_value(settings).expect("RAG settings serialize")),
        )
    }

    pub fn export_metrics(&self) -> Result<MetricsExport, ApiError> {
        self.request_json(Method::POST, "/metrics/export", None)
    }

    pub fn retrieval_traces(&self) -> Result<Vec<RetrievalTraceSummary>, ApiError> {
        let response: RetrievalTracesResponse =
            self.request_json(Method::GET, "/retrieval/traces", None)?;
        Ok(response.traces)
    }

    pub fn retrieval_trace(&self, id: &str) -> Result<Value, ApiError> {
        self.request_value(
            Method::GET,
            &format!("/retrieval/traces/{}", path_segment(id)),
            None,
        )
    }

    pub fn index_health(&self) -> Result<IndexHealth, ApiError> {
        self.request_json(Method::GET, "/observability/index-health", None)
    }

    pub fn eval_runs(&self) -> Result<Vec<EvalRun>, ApiError> {
        let response: EvalRunsResponse = self.request_json(Method::GET, "/eval/runs", None)?;
        Ok(response.runs)
    }

    pub fn run_manual_eval(&self, question: &str, expected_doc: &str) -> Result<EvalRun, ApiError> {
        self.request_json(
            Method::POST,
            "/eval/runs",
            Some(json!({
                "evals": [{
                    "id": format!("manual-{}", unix_seconds()),
                    "question": question,
                    "expected_doc_ids": [expected_doc]
                }],
                "pipeline": "hybrid_rerank",
                "top_k": 10,
                "answers": {},
                "sources": {},
                "run_meta": {}
            })),
        )
    }

    pub fn query_stream(
        &self,
        request: QueryRequest,
        stop: &AtomicBool,
        mut on_event: impl FnMut(QueryEvent),
    ) -> Result<(), ApiError> {
        let response = self.client.post(self.url("/query")).json(&request).send()?;
        if !response.status().is_success() {
            return Err(parse_error_response(response));
        }

        let mut event_name = String::from("message");
        let mut data_lines = Vec::new();
        let mut terminal_event = false;
        for line in BufReader::new(response).lines() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let line = line?;
            if line.is_empty() {
                if let Some(event) = decode_query_event(&event_name, &data_lines.join("\n")) {
                    let is_terminal = matches!(event, QueryEvent::Done | QueryEvent::Error(_));
                    terminal_event |= is_terminal;
                    on_event(event);
                    if is_terminal {
                        break;
                    }
                }
                event_name.clear();
                event_name.push_str("message");
                data_lines.clear();
            } else if let Some(value) = line.strip_prefix("event: ") {
                event_name = value.trim().to_string();
            } else if let Some(value) = line.strip_prefix("data: ") {
                data_lines.push(value.to_string());
            }
        }
        if !data_lines.is_empty() {
            if let Some(event) = decode_query_event(&event_name, &data_lines.join("\n")) {
                terminal_event |= matches!(event, QueryEvent::Done | QueryEvent::Error(_));
                on_event(event);
            }
        }
        if !stop.load(Ordering::Relaxed) && !terminal_event {
            return Err(ApiError {
                status: None,
                message: "Cephalon query stream ended before a terminal event.".into(),
            });
        }
        Ok(())
    }

    pub fn event_loop(&self, stop: &AtomicBool, mut on_event: impl FnMut(EventStreamEvent)) {
        while !stop.load(Ordering::Relaxed) {
            match self.event_stream_once(stop, &mut on_event) {
                Ok(()) => {}
                Err(error) => on_event(EventStreamEvent::Error(error.message)),
            }
            if !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(1200));
            }
        }
    }

    fn event_stream_once(
        &self,
        stop: &AtomicBool,
        on_event: &mut impl FnMut(EventStreamEvent),
    ) -> Result<(), ApiError> {
        let response = self.client.get(self.url("/events")).send()?;
        if !response.status().is_success() {
            return Err(parse_error_response(response));
        }
        on_event(EventStreamEvent::Connected);

        let mut event_name = String::from("message");
        let mut data_lines = Vec::new();
        for line in BufReader::new(response).lines() {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let line = line?;
            if line.is_empty() {
                if let Some(event) = decode_event_stream_event(&event_name, &data_lines.join("\n"))
                {
                    on_event(event);
                }
                event_name.clear();
                event_name.push_str("message");
                data_lines.clear();
            } else if let Some(value) = line.strip_prefix("event: ") {
                event_name = value.trim().to_string();
            } else if let Some(value) = line.strip_prefix("data: ") {
                data_lines.push(value.to_string());
            }
        }
        Ok(())
    }
}

fn parse_json_response<T: DeserializeOwned>(response: Response) -> Result<T, ApiError> {
    if !response.status().is_success() {
        return Err(parse_error_response(response));
    }
    response.json().map_err(ApiError::from)
}

fn parse_error_response(response: Response) -> ApiError {
    let status = response.status().as_u16();
    let body = response.text().unwrap_or_default();
    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("detail")
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| {
            if body.trim().is_empty() {
                "The Cephalon service returned an empty error.".to_string()
            } else {
                body
            }
        });
    ApiError {
        status: Some(status),
        message,
    }
}

fn path_segment(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    encoded
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn decode_query_event(event_name: &str, data: &str) -> Option<QueryEvent> {
    let value = serde_json::from_str::<Value>(data).unwrap_or_else(|_| json!({"text": data}));
    match event_name {
        "phase" => Some(QueryEvent::Phase(
            value
                .get("phase")
                .and_then(Value::as_str)
                .unwrap_or("answering")
                .to_string(),
        )),
        "token" => Some(QueryEvent::Token(
            value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        )),
        "source" => serde_json::from_value(value).ok().map(QueryEvent::Source),
        "conversation" => Some(QueryEvent::Conversation(
            value
                .get("conversation_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        )),
        "answer_meta" => Some(QueryEvent::AnswerMeta(value)),
        "error" => Some(QueryEvent::Error(
            value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Query stream failed.")
                .to_string(),
        )),
        "done" => Some(QueryEvent::Done),
        _ => None,
    }
}

fn decode_event_stream_event(event_name: &str, data: &str) -> Option<EventStreamEvent> {
    let payload = serde_json::from_str::<Value>(data).unwrap_or_else(|_| json!({"raw": data}));
    match event_name {
        "ready" | "heartbeat" => Some(EventStreamEvent::Heartbeat),
        "job" => Some(EventStreamEvent::JobChanged(payload)),
        "document" | "documents" => Some(EventStreamEvent::DocumentChanged(payload)),
        "conversation" => Some(EventStreamEvent::ConversationChanged(payload)),
        "settings" => Some(EventStreamEvent::SettingsChanged(payload)),
        "llama_server" => Some(EventStreamEvent::LlamaServerChanged(payload)),
        "" | "message" => None,
        _ => {
            if data.is_empty() {
                None
            } else {
                Some(EventStreamEvent::Other {
                    name: event_name.to_string(),
                    payload,
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum QueryEvent {
    Phase(String),
    Token(String),
    Source(SourceChunk),
    Conversation(String),
    AnswerMeta(Value),
    Error(String),
    Done,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum EventStreamEvent {
    Connected,
    Heartbeat,
    DocumentChanged(Value),
    JobChanged(Value),
    ConversationChanged(Value),
    SettingsChanged(Value),
    LlamaServerChanged(Value),
    Other { name: String, payload: Value },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventStreamRefresh {
    Documents,
    Jobs,
    Conversations,
    Settings,
    LlamaServer,
}

impl EventStreamEvent {
    pub fn refresh_target(&self) -> Option<EventStreamRefresh> {
        match self {
            Self::DocumentChanged(_) => Some(EventStreamRefresh::Documents),
            Self::JobChanged(_) => Some(EventStreamRefresh::Jobs),
            Self::ConversationChanged(_) => Some(EventStreamRefresh::Conversations),
            Self::SettingsChanged(_) => Some(EventStreamRefresh::Settings),
            Self::LlamaServerChanged(_) => Some(EventStreamRefresh::LlamaServer),
            Self::Connected | Self::Heartbeat | Self::Other { .. } | Self::Error(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryRequest {
    pub prompt: String,
    pub model: String,
    pub history: Vec<Message>,
    pub settings: Option<RagSettings>,
    pub conversation_id: Option<String>,
    pub retrieval_scope: String,
    pub response_effort: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub name: String,
    pub status: String,
    #[serde(default)]
    pub chunks: i64,
    #[serde(default)]
    pub path: String,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<i64>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub stale_embedding: bool,
    #[serde(default)]
    pub chunk_preview: Vec<Value>,
    #[serde(default)]
    pub modified_at: Option<i64>,
    #[serde(default)]
    pub last_indexed_at: Option<i64>,
    #[serde(default)]
    pub embedding_model_id: Option<String>,
    #[serde(default)]
    pub embedding_dim: Option<i64>,
    #[serde(default)]
    pub stale_reasons: Vec<Value>,
    #[serde(default)]
    pub extraction_mode: Option<String>,
    #[serde(default)]
    pub last_retrieved_at: Option<i64>,
    #[serde(default)]
    pub retrieval_count: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceChunk {
    #[serde(default)]
    pub rank: i64,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub doc_id: String,
    #[serde(default)]
    pub doc_name: String,
    #[serde(default)]
    pub chunk_id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub source_kind: Option<String>,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub final_score: Option<f64>,
    #[serde(default)]
    pub snippet: String,
    #[serde(default)]
    pub evidence_text: Option<String>,
    #[serde(default)]
    pub vector_score: Option<f64>,
    #[serde(default)]
    pub lexical_score: Option<f64>,
    #[serde(default)]
    pub fusion_score: Option<f64>,
    #[serde(default)]
    pub rerank_score: Option<f64>,
    #[serde(default)]
    pub reranker_raw_score: Option<f64>,
    #[serde(default)]
    pub listwise_rank: Option<i64>,
    #[serde(default)]
    pub subquery_id: Option<String>,
    #[serde(default)]
    pub block_type: Option<String>,
    #[serde(default)]
    pub page_number: Option<i64>,
    #[serde(default)]
    pub page_end: Option<i64>,
    #[serde(default)]
    pub section_heading: Option<String>,
    #[serde(default)]
    pub heading_path: Vec<String>,
    #[serde(default)]
    pub block_index: Option<i64>,
    #[serde(default)]
    pub bounding_box: Option<Vec<f64>>,
    #[serde(default)]
    pub table_id: Option<String>,
    #[serde(default)]
    pub table_title: Option<String>,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub table_bounding_box: Option<Vec<f64>>,
    #[serde(default)]
    pub cell_refs: Vec<String>,
    #[serde(default)]
    pub verification_cell_refs: Vec<String>,
    #[serde(default)]
    pub header_refs: Vec<String>,
    #[serde(default)]
    pub table_operation: Option<String>,
    #[serde(default)]
    pub table_result: Vec<Value>,
    #[serde(default)]
    pub cells: Vec<Value>,
    #[serde(default)]
    pub element_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Value,
    #[serde(default)]
    pub context_assembly: Value,
    #[serde(default)]
    pub context_selection: Value,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    #[serde(default)]
    pub requirement_ids: Vec<String>,
    #[serde(default)]
    pub retrieval_round: i64,
    #[serde(default)]
    pub triggering_gap: Option<String>,
    #[serde(default)]
    pub assets: Vec<Value>,
    #[serde(default)]
    pub raw_chunk: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoredMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub settings: Option<Value>,
    #[serde(default)]
    pub meta: Option<Value>,
    #[serde(default)]
    pub sources: Vec<SourceChunk>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: i64,
    #[serde(default)]
    pub messages: Vec<StoredMessage>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub next_before: Option<i64>,
}

/// Merge an older conversation page into the currently visible messages.
///
/// The service returns each page in display order, so the result is rebuilt by
/// message id and then sorted chronologically. This keeps a retry or overlapping
/// page from duplicating messages already on screen.
pub fn merge_conversation_messages(existing: &mut Vec<StoredMessage>, page: &Conversation) {
    let mut by_id = std::collections::HashMap::with_capacity(existing.len() + page.messages.len());
    for message in existing.drain(..) {
        by_id.insert(message.id.clone(), message);
    }
    for message in &page.messages {
        by_id
            .entry(message.id.clone())
            .or_insert_with(|| message.clone());
    }
    let mut merged: Vec<_> = by_id.into_values().collect();
    merged.sort_by(|left, right| {
        left.created_at
            .cmp(&right.created_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    *existing = merged;
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HealthResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub startup_error: Option<String>,
    #[serde(default)]
    pub engines_ready: bool,
    #[serde(default)]
    pub active_model: Option<String>,
    #[serde(default)]
    pub active_context_tokens: Option<i64>,
    #[serde(default)]
    pub last_model_load_error: Option<String>,
    #[serde(default)]
    pub last_model_error: Option<String>,
    #[serde(default)]
    pub llama_backend: Option<LlamaBackendStatus>,
    #[serde(default)]
    pub retrieval_error: Option<String>,
    #[serde(default)]
    pub retrieval_stack: Option<FixedRetrievalStatus>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LlamaBackendStatus {
    #[serde(default)]
    pub server_url: Option<String>,
    #[serde(default)]
    pub server_available: Option<bool>,
    #[serde(default)]
    pub connection_status: Option<String>,
    #[serde(default)]
    pub server_error: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub backend_label: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelsResponse {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub active_model: Option<String>,
    #[serde(default)]
    pub active_context_tokens: Option<i64>,
    #[serde(default)]
    pub llama_backend: Option<LlamaBackendStatus>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlamaServerSettings {
    #[serde(default)]
    pub server_url: String,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub context_tokens: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadModelResponse {
    #[serde(default)]
    pub active_model: Option<String>,
    #[serde(default)]
    pub active_context_tokens: Option<i64>,
    #[serde(default)]
    pub llama_backend: Option<LlamaBackendStatus>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RagSettings {
    #[serde(default)]
    pub top_k: i64,
    #[serde(default)]
    pub rerank_top_n: i64,
    #[serde(default)]
    pub max_tokens: i64,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default)]
    pub parent_target_tokens: i64,
    #[serde(default)]
    pub parent_max_tokens: i64,
    #[serde(default)]
    pub child_target_tokens: i64,
    #[serde(default)]
    pub child_max_tokens: i64,
    #[serde(default)]
    pub child_overlap_tokens: i64,
    #[serde(default)]
    pub context_tokens: i64,
    #[serde(default)]
    pub evidence_required: bool,
    #[serde(default)]
    pub conversation_memory: bool,
    #[serde(default)]
    pub trace_persistence: bool,
    #[serde(default)]
    pub hierarchical_context: bool,
    #[serde(default)]
    pub layout_evidence: bool,
    #[serde(default)]
    pub evidence_ledger: bool,
    #[serde(default)]
    pub coverage_selection: bool,
    #[serde(default)]
    pub gap_retrieval: bool,
    #[serde(default)]
    pub verified_answer_repair: bool,
    #[serde(default)]
    pub no_answer_min_confidence: f64,
    #[serde(default)]
    pub no_answer_min_rerank_score: f64,
    #[serde(default)]
    pub no_answer_min_vector_score: f64,
    #[serde(default)]
    pub no_answer_min_source_count: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct FixedRetrievalStatus {
    #[serde(default)]
    pub reindex_required: bool,
    #[serde(default)]
    pub embedder: Option<FixedModelInfo>,
    #[serde(default)]
    pub reranker: Option<FixedModelInfo>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct FixedModelInfo {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub verified: bool,
    #[serde(default)]
    pub selected_backend: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReindexProgress {
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub processed: i64,
    #[serde(default)]
    pub total: i64,
    #[serde(default)]
    pub succeeded: i64,
    #[serde(default)]
    pub failed: i64,
    #[serde(default)]
    pub stale_document_count: i64,
    #[serde(default)]
    pub reindex_required: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct IndexHealth {
    #[serde(default)]
    pub document_count: i64,
    #[serde(default)]
    pub chunk_count: i64,
    #[serde(default)]
    pub embedded_chunk_count: i64,
    #[serde(default)]
    pub stale_document_count: i64,
    #[serde(default)]
    pub failed_ingestion_count: i64,
    #[serde(default)]
    pub index_size_bytes: i64,
    #[serde(default)]
    pub average_chunk_length: f64,
    #[serde(default)]
    pub duplicate_chunk_rate: f64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RetrievalTraceSummary {
    #[serde(default)]
    pub query_id: String,
    #[serde(default)]
    pub raw_query: String,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub total_ms: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EvalRun {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub pipeline: String,
    #[serde(default)]
    pub top_k: i64,
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub aggregate: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReindexResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub total: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct IngestResponse {
    #[serde(default)]
    pub job_id: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MetricsExport {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct DocumentsResponse {
    #[serde(default)]
    documents: Vec<Document>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ConversationsResponse {
    #[serde(default)]
    conversations: Vec<Conversation>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RetrievalTracesResponse {
    #[serde(default)]
    traces: Vec<RetrievalTraceSummary>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct EvalRunsResponse {
    #[serde(default)]
    runs: Vec<EvalRun>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        decode_event_stream_event, decode_query_event, merge_conversation_messages, path_segment,
        Conversation, EventStreamEvent, EventStreamRefresh, QueryEvent, StoredMessage,
    };

    #[test]
    fn path_segments_are_url_encoded() {
        assert_eq!(path_segment("a document/#1"), "a%20document%2F%231");
        assert_eq!(path_segment("café + notes"), "caf%C3%A9%20%2B%20notes");
    }

    #[test]
    fn query_sse_packets_keep_typed_events() {
        assert!(matches!(
            decode_query_event("token", r#"{"text":"hello"}"#),
            Some(QueryEvent::Token(text)) if text == "hello"
        ));
        assert!(matches!(
            decode_query_event("conversation", r#"{"conversation_id":"c1"}"#),
            Some(QueryEvent::Conversation(id)) if id == "c1"
        ));
    }

    #[test]
    fn backend_sse_packets_keep_targeted_event_types() {
        let settings_event =
            decode_event_stream_event("settings", r#"{"type":"settings","payload":{"top_k":4}}"#)
                .expect("settings event");
        assert_eq!(
            settings_event.refresh_target(),
            Some(EventStreamRefresh::Settings)
        );
        assert!(matches!(
            settings_event,
            EventStreamEvent::SettingsChanged(value)
                if value["payload"]["top_k"] == json!(4)
        ));
        assert!(matches!(
            decode_event_stream_event("job", r#"{"type":"job","payload":{"status":"running"}}"#),
            Some(EventStreamEvent::JobChanged(value))
                if value["payload"]["status"] == json!("running")
        ));
        assert!(matches!(
            decode_event_stream_event("other", r#"{"ok":true}"#),
            Some(EventStreamEvent::Other { name, .. }) if name == "other"
        ));
    }

    #[test]
    fn older_conversation_pages_prepend_without_duplicates() {
        let mut existing = vec![StoredMessage {
            id: "m2".into(),
            created_at: 20,
            ..StoredMessage::default()
        }];
        let page = Conversation {
            messages: vec![
                StoredMessage {
                    id: "m1".into(),
                    created_at: 10,
                    ..StoredMessage::default()
                },
                StoredMessage {
                    id: "m2".into(),
                    created_at: 20,
                    role: "assistant".into(),
                    ..StoredMessage::default()
                },
            ],
            ..Conversation::default()
        };
        merge_conversation_messages(&mut existing, &page);
        assert_eq!(
            existing
                .iter()
                .map(|message| message.id.as_str())
                .collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        assert_eq!(existing[1].role, "");
    }
}
