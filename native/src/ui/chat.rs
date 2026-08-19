//! Chat presentation: native CommonMark/GFM rendering, citations, and actions.

use gpui::prelude::*;
use gpui::{div, px, Context, FontWeight, SharedString};
use pulldown_cmark::{Alignment, CodeBlockKind, HeadingLevel, Tag};

use super::markdown::{
    external_url_allowed, parse_markdown, split_citations, text_content, visible_answer,
    InlineFragment, MarkdownNode,
};
use super::theme::{
    bg, faint, green, line, link as theme_link, muted, orange, orange_light, panel_2, panel_3, red,
    text, yellow,
};
use super::{ChatMessage, NativeApp, Panel, SourceChunk};

fn is_inline_markdown_node(node: &MarkdownNode) -> bool {
    match node {
        MarkdownNode::Text(_)
        | MarkdownNode::Code(_)
        | MarkdownNode::Html(_)
        | MarkdownNode::InlineHtml(_)
        | MarkdownNode::InlineMath(_)
        | MarkdownNode::DisplayMath(_)
        | MarkdownNode::FootnoteReference(_)
        | MarkdownNode::SoftBreak
        | MarkdownNode::HardBreak => true,
        MarkdownNode::Container { tag, .. } => matches!(
            tag,
            Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Superscript
                | Tag::Subscript
                | Tag::Link { .. }
                | Tag::Image { .. }
        ),
        MarkdownNode::Rule | MarkdownNode::TaskListMarker(_) => false,
    }
}

pub(crate) fn code_block(id: &str, value: &str, language: &str) -> gpui::Div {
    let mut block = div()
        .w_full()
        .min_w_0()
        .p_2()
        .bg(panel_3())
        .border_1()
        .border_color(line())
        .rounded_sm()
        .font_family(".ZedMono")
        .text_size(px(11.))
        .text_color(text());
    if !language.is_empty() {
        block = block.child(
            div()
                .font_family(".ZedSans")
                .text_size(px(10.))
                .text_color(muted())
                .child(language.to_string()),
        );
    }
    block.child(
        div()
            .id(SharedString::from(format!("{id}-horizontal-scroll")))
            .w_full()
            .min_w_0()
            .overflow_x_scroll()
            .whitespace_nowrap()
            .child(value.to_string()),
    )
}

impl NativeApp {
    fn render_answer_content(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let answer = if message.content.is_empty() && message.streaming {
            "…"
        } else {
            message.content.as_str()
        };
        let nodes = parse_markdown(answer);
        let mut content = div()
            .flex()
            .flex_col()
            .gap_2()
            .w_full()
            .min_w_0()
            .whitespace_normal();
        for (block_index, node) in nodes.iter().enumerate() {
            content = content.child(self.render_markdown_block(
                message_index,
                message,
                node,
                &format!("answer-{message_index}-{block_index}"),
                cx,
            ));
        }
        if answer.is_empty() {
            content = content.child(div().text_color(faint()).child(" "));
        }
        content
    }

    fn render_markdown_blocks(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        nodes: &[MarkdownNode],
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let mut blocks = div()
            .flex()
            .flex_col()
            .gap_2()
            .w_full()
            .min_w_0()
            .whitespace_normal();
        for (index, node) in nodes.iter().enumerate() {
            blocks = blocks.child(self.render_markdown_block(
                message_index,
                message,
                node,
                &format!("{id_prefix}-{index}"),
                cx,
            ));
        }
        blocks
    }

    fn render_markdown_block(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        node: &MarkdownNode,
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        match node {
            MarkdownNode::Container { tag, children } => match tag {
                Tag::Paragraph => {
                    self.render_inline_nodes(message_index, message, children, id_prefix, cx)
                }
                Tag::Heading { level, .. } => {
                    let mut heading = self
                        .render_inline_nodes(message_index, message, children, id_prefix, cx)
                        .text_color(orange_light())
                        .font_weight(FontWeight::BOLD);
                    heading = heading.text_size(px(match level {
                        HeadingLevel::H1 => 19.,
                        HeadingLevel::H2 => 17.,
                        HeadingLevel::H3 => 15.,
                        _ => 14.,
                    }));
                    heading
                }
                Tag::BlockQuote(_) => div()
                    .w_full()
                    .pl_3()
                    .border_l_2()
                    .border_color(orange())
                    .text_color(muted())
                    .child(self.render_markdown_blocks(
                        message_index,
                        message,
                        children,
                        id_prefix,
                        cx,
                    )),
                Tag::CodeBlock(kind) => {
                    let language = match kind {
                        CodeBlockKind::Fenced(value) => value.as_ref(),
                        CodeBlockKind::Indented => "",
                    };
                    code_block(id_prefix, &text_content(children), language)
                }
                Tag::HtmlBlock => code_block(id_prefix, &text_content(children), "HTML"),
                Tag::List(start) => {
                    self.render_list(message_index, message, children, *start, id_prefix, cx)
                }
                Tag::Item => {
                    self.render_markdown_blocks(message_index, message, children, id_prefix, cx)
                }
                Tag::Table(alignments) => {
                    self.render_table(message_index, message, children, alignments, id_prefix, cx)
                }
                Tag::TableHead => {
                    self.render_markdown_blocks(message_index, message, children, id_prefix, cx)
                }
                Tag::TableRow => self.render_table_row(
                    message_index,
                    message,
                    children,
                    &[],
                    false,
                    id_prefix,
                    cx,
                ),
                Tag::TableCell => {
                    self.render_inline_nodes(message_index, message, children, id_prefix, cx)
                }
                Tag::FootnoteDefinition(label) => div()
                    .w_full()
                    .p_2()
                    .bg(panel_3())
                    .border_l_2()
                    .border_color(line())
                    .text_size(px(11.))
                    .child(format!("Footnote [{label}]"))
                    .child(self.render_markdown_blocks(
                        message_index,
                        message,
                        children,
                        id_prefix,
                        cx,
                    )),
                Tag::DefinitionList
                | Tag::DefinitionListTitle
                | Tag::DefinitionListDefinition
                | Tag::MetadataBlock(_) => {
                    self.render_markdown_blocks(message_index, message, children, id_prefix, cx)
                }
                Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Superscript
                | Tag::Subscript
                | Tag::Link { .. }
                | Tag::Image { .. } => self.render_inline_nodes(
                    message_index,
                    message,
                    std::slice::from_ref(node),
                    id_prefix,
                    cx,
                ),
            },
            MarkdownNode::Rule => div().w_full().h(px(1.)).bg(line()),
            MarkdownNode::Text(_) => self.render_inline_nodes(
                message_index,
                message,
                std::slice::from_ref(node),
                id_prefix,
                cx,
            ),
            MarkdownNode::Code(_)
            | MarkdownNode::Html(_)
            | MarkdownNode::InlineHtml(_)
            | MarkdownNode::InlineMath(_)
            | MarkdownNode::DisplayMath(_)
            | MarkdownNode::FootnoteReference(_) => div()
                .w_full()
                .min_w_0()
                .child(self.render_inline_node(message_index, message, node, id_prefix, cx)),
            MarkdownNode::SoftBreak | MarkdownNode::HardBreak => div().h(px(4.)),
            MarkdownNode::TaskListMarker(checked) => div()
                .flex_none()
                .text_color(if *checked { green() } else { muted() })
                .child(if *checked { "☑" } else { "☐" }),
        }
    }

    fn render_list(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        children: &[MarkdownNode],
        start: Option<u64>,
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let mut list = div().w_full().flex().flex_col().gap_1();
        let mut item_index = 0;
        for child in children {
            let MarkdownNode::Container {
                tag: Tag::Item,
                children: item_children,
            } = child
            else {
                continue;
            };
            let marker = match start {
                Some(number) => format!("{}.", number + item_index),
                None => "•".to_string(),
            };
            let (task_checked, item_content) = match item_children.first() {
                Some(MarkdownNode::TaskListMarker(checked)) => {
                    (Some(*checked), &item_children[1..])
                }
                _ => (None, item_children.as_slice()),
            };
            let marker = task_checked
                .map(|checked| if checked { "☑" } else { "☐" }.to_string())
                .unwrap_or(marker);
            let item_body = if item_content.iter().all(is_inline_markdown_node) {
                self.render_inline_nodes(
                    message_index,
                    message,
                    item_content,
                    &format!("{id_prefix}-item-{item_index}"),
                    cx,
                )
            } else {
                self.render_markdown_blocks(
                    message_index,
                    message,
                    item_content,
                    &format!("{id_prefix}-item-{item_index}"),
                    cx,
                )
            }
            .flex_1()
            .min_w_0();
            let item = div()
                .w_full()
                .flex()
                .items_start()
                .min_w_0()
                .gap_2()
                .child(
                    div()
                        .flex_none()
                        .w(px(20.))
                        .text_color(orange_light())
                        .child(marker),
                )
                .child(item_body);
            list = list.child(item);
            item_index += 1;
        }
        list
    }

    fn render_table(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        children: &[MarkdownNode],
        alignments: &[Alignment],
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let mut table = div()
            .w_full()
            .flex()
            .flex_col()
            .bg(panel_2())
            .border_1()
            .border_color(line())
            .rounded_sm();
        for (row_index, child) in children.iter().enumerate() {
            match child {
                MarkdownNode::Container {
                    tag: Tag::TableHead,
                    children: head_cells,
                } => {
                    table = table.child(self.render_table_row(
                        message_index,
                        message,
                        head_cells,
                        alignments,
                        true,
                        &format!("{id_prefix}-head"),
                        cx,
                    ));
                }
                MarkdownNode::Container {
                    tag: Tag::TableRow,
                    children: row_children,
                } => {
                    table = table.child(self.render_table_row(
                        message_index,
                        message,
                        row_children,
                        alignments,
                        false,
                        &format!("{id_prefix}-row-{row_index}"),
                        cx,
                    ));
                }
                _ => {}
            }
        }
        table
    }

    fn render_table_row(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        children: &[MarkdownNode],
        alignments: &[Alignment],
        header: bool,
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let row = if let Some(MarkdownNode::Container {
            tag: Tag::TableRow,
            children: row_children,
        }) = children.first()
        {
            row_children.as_slice()
        } else {
            children
        };
        let mut table_row = div()
            .w_full()
            .flex()
            .items_start()
            .border_b_1()
            .border_color(line())
            .px_1()
            .py(px(2.));
        for (cell_index, cell) in row.iter().enumerate() {
            let MarkdownNode::Container {
                tag: Tag::TableCell,
                children: cell_children,
            } = cell
            else {
                continue;
            };
            let mut cell_view = self.render_inline_nodes(
                message_index,
                message,
                cell_children,
                &format!("{id_prefix}-cell-{cell_index}"),
                cx,
            );
            cell_view = cell_view.flex_1().text_size(px(11.)).text_color(if header {
                orange_light()
            } else {
                text()
            });
            if let Some(alignment) = alignments.get(cell_index) {
                cell_view = match alignment {
                    Alignment::Center => cell_view.justify_center(),
                    Alignment::Right => cell_view.justify_end(),
                    Alignment::None | Alignment::Left => cell_view.justify_start(),
                };
            }
            table_row = table_row.child(cell_view);
        }
        table_row
    }

    fn render_inline_nodes(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        nodes: &[MarkdownNode],
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        self.render_inline_nodes_impl(message_index, message, nodes, id_prefix, true, cx)
    }

    fn render_inline_group(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        nodes: &[MarkdownNode],
        id_prefix: &str,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        self.render_inline_nodes_impl(message_index, message, nodes, id_prefix, false, cx)
    }

    fn render_inline_nodes_impl(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        nodes: &[MarkdownNode],
        id_prefix: &str,
        full_width: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let mut row = div()
            .min_w_0()
            .flex()
            .flex_wrap()
            .items_baseline()
            .whitespace_normal()
            .text_color(text());
        if full_width {
            row = row.w_full();
        }
        let mut child_index = 0;
        for node in nodes {
            match node {
                MarkdownNode::Text(value) => {
                    for fragment in split_citations(value) {
                        let id = format!("{id_prefix}-{child_index}");
                        row = row.child(self.render_inline_fragment(
                            message_index,
                            message,
                            fragment,
                            &id,
                            cx,
                        ));
                        child_index += 1;
                    }
                }
                _ => {
                    let id = format!("{id_prefix}-{child_index}");
                    row = row.child(self.render_inline_node(message_index, message, node, &id, cx));
                    child_index += 1;
                }
            }
        }
        row
    }

    fn render_inline_node(
        &mut self,
        message_index: usize,
        message: &ChatMessage,
        node: &MarkdownNode,
        id: &str,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        match node {
            MarkdownNode::Text(value) => div()
                .flex_initial()
                .min_w_0()
                .whitespace_normal()
                .child(value.clone())
                .into_any_element(),
            MarkdownNode::Code(value) => div()
                .flex_initial()
                .min_w_0()
                .px_1()
                .bg(panel_3())
                .rounded_sm()
                .font_family(".ZedMono")
                .text_size(px(11.))
                .child(value.clone())
                .into_any_element(),
            MarkdownNode::InlineMath(value) => code_block(id, value, "math").into_any_element(),
            MarkdownNode::DisplayMath(value) => code_block(id, value, "math").into_any_element(),
            MarkdownNode::InlineHtml(value) | MarkdownNode::Html(value) => div()
                .text_color(muted())
                .font_family(".ZedMono")
                .text_size(px(11.))
                .child(value.clone())
                .into_any_element(),
            MarkdownNode::FootnoteReference(value) => div()
                .px_1()
                .text_size(px(10.))
                .text_color(muted())
                .child(format!("[^{value}]"))
                .into_any_element(),
            MarkdownNode::SoftBreak => div().child(" ").into_any_element(),
            MarkdownNode::HardBreak => div().w_full().h(px(1.)).into_any_element(),
            MarkdownNode::TaskListMarker(checked) => div()
                .flex_none()
                .text_color(if *checked { green() } else { muted() })
                .child(if *checked { "☑" } else { "☐" })
                .into_any_element(),
            MarkdownNode::Rule => div().w_full().h(px(1.)).bg(line()).into_any_element(),
            MarkdownNode::Container { tag, children } => match tag {
                Tag::Emphasis => div()
                    .flex_initial()
                    .min_w_0()
                    .italic()
                    .child(self.render_inline_group(
                        message_index,
                        message,
                        children,
                        &format!("{id}-emphasis"),
                        cx,
                    ))
                    .into_any_element(),
                Tag::Strong => div()
                    .flex_initial()
                    .min_w_0()
                    .font_weight(FontWeight::BOLD)
                    .child(self.render_inline_group(
                        message_index,
                        message,
                        children,
                        &format!("{id}-strong"),
                        cx,
                    ))
                    .into_any_element(),
                Tag::Strikethrough => div()
                    .flex_initial()
                    .min_w_0()
                    .text_color(muted())
                    .child(self.render_inline_group(
                        message_index,
                        message,
                        children,
                        &format!("{id}-strike"),
                        cx,
                    ))
                    .into_any_element(),
                Tag::Superscript => div()
                    .flex_initial()
                    .min_w_0()
                    .text_size(px(10.))
                    .child(self.render_inline_group(
                        message_index,
                        message,
                        children,
                        &format!("{id}-sup"),
                        cx,
                    ))
                    .into_any_element(),
                Tag::Subscript => div()
                    .flex_initial()
                    .min_w_0()
                    .text_size(px(10.))
                    .child(self.render_inline_group(
                        message_index,
                        message,
                        children,
                        &format!("{id}-sub"),
                        cx,
                    ))
                    .into_any_element(),
                Tag::Link { dest_url, .. } => {
                    let safe = external_url_allowed(dest_url.as_ref());
                    let mut link = div()
                        .id(SharedString::from(id.to_string()))
                        .flex_initial()
                        .min_w_0()
                        .text_color(if safe { theme_link() } else { muted() })
                        .child(self.render_inline_group(
                            message_index,
                            message,
                            children,
                            &format!("{id}-link"),
                            cx,
                        ));
                    if safe {
                        let target = dest_url.to_string();
                        link = link
                            .cursor_pointer()
                            .underline()
                            .on_click(move |_, _, cx| cx.open_url(&target));
                    }
                    link.into_any_element()
                }
                Tag::Image { dest_url, .. } => div()
                    .flex()
                    .flex_wrap()
                    .text_color(muted())
                    .child(format!("[image: {}]", text_content(children)))
                    .child(if dest_url.is_empty() {
                        ""
                    } else {
                        " · preview unavailable"
                    })
                    .into_any_element(),
                _ => self
                    .render_markdown_blocks(message_index, message, children, id, cx)
                    .into_any_element(),
            },
        }
    }

    fn render_inline_fragment(
        &mut self,
        _message_index: usize,
        message: &ChatMessage,
        fragment: InlineFragment,
        id: &str,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        match fragment {
            InlineFragment::Text(value) => div()
                .flex_initial()
                .min_w_0()
                .whitespace_normal()
                .child(value)
                .into_any_element(),
            InlineFragment::Citation(marker) => {
                let sources = message.sources.clone();
                let citation = marker.clone();
                div()
                    .id(SharedString::from(id.to_string()))
                    .flex_none()
                    .px_1()
                    .py(px(1.))
                    .bg(panel_3())
                    .border_1()
                    .border_color(orange())
                    .rounded_sm()
                    .cursor_pointer()
                    .text_size(px(10.))
                    .text_color(orange_light())
                    .child(format!("[{marker}]"))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.open_source_citation(citation.clone(), sources.clone(), cx)
                    }))
                    .into_any_element()
            }
        }
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
            self.expanded_sources.insert(super::source_key(source));
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
        self.set_input_text(super::InputTarget::Composer, prompt, cx);
        self.regenerate_without_user = true;
        self.send_message(cx);
    }

    pub(super) fn render_chat(&mut self, cx: &mut Context<Self>) -> gpui::Div {
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
            if let Some(error_detail) = &message.error_detail {
                card = card.child(
                    div()
                        .mt_2()
                        .text_size(px(11.))
                        .text_color(red())
                        .child(format!("Backend save warning: {error_detail}")),
                );
            }
            if !message.sources.is_empty() {
                let sources = message.sources.clone();
                card = card.child(super::ui_button(
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
                card = card.child(super::ui_button(
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
                        .child(super::ui_button(
                            format!("message-copy-{index}"),
                            "Copy answer",
                            false,
                            cx.listener(move |_, _, _, cx| {
                                cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                                    visible_answer(&copy_message.raw_content),
                                ));
                            }),
                        ))
                        .child(super::ui_button(
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
        let composer = super::input_field(
            "composer",
            self.inputs.composer.clone(),
            cx.listener(|this, _, window, cx| {
                this.focus_input(super::InputTarget::Composer, window, cx)
            }),
        );
        div()
            .flex()
            .flex_col()
            .flex_1()
            .h_full()
            .min_h_0()
            .min_w_0()
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
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .text_size(px(16.))
                            .text_color(text())
                            .child(title),
                    )
                    .child(
                        div()
                            .flex_none()
                            .ml_2()
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
                            .child(super::ui_button(
                                "retrieval-scope",
                                format!("Scope: {}", self.retrieval_scope),
                                false,
                                cx.listener(|this, _, _, cx| this.cycle_retrieval_scope(cx)),
                            ))
                            .child(super::ui_button(
                                "response-effort",
                                format!("Effort: {}", self.response_effort),
                                false,
                                cx.listener(|this, _, _, cx| this.cycle_response_effort(cx)),
                            ))
                            .child(div().flex_1())
                            .child(if self.is_typing {
                                super::ui_button(
                                    "stop-query",
                                    "Stop",
                                    false,
                                    cx.listener(|this, _, _, cx| this.stop_query(cx)),
                                )
                            } else {
                                super::ui_button(
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
