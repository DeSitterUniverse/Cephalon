//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_document(&mut self, cx: &mut Context<Self>) -> gpui::Div {
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
            chunk_preview = chunk_preview.child(render_chunk_preview(index, chunk));
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
                    .flex_wrap()
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
                        cx.listener(move |this, _, window, cx| {
                            this.ask_delete_document(id.clone(), name.clone(), window, cx)
                        })
                    })),
            )
    }
}
