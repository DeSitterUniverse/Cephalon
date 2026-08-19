//! Theme-aware chat content primitives.
//!
//! The stateful answer/citation interaction remains on `NativeApp`; this module
//! owns the reusable native surfaces for Markdown tables and fenced code blocks.

use gpui::prelude::*;
use gpui::{div, px};

use super::theme::{line, muted, orange_light, panel_2, panel_3, text};

pub(crate) fn render_markdown_table(rows: Vec<Vec<String>>) -> gpui::Div {
    let mut table = div()
        .w_full()
        .flex()
        .flex_col()
        .bg(panel_2())
        .border_1()
        .border_color(line())
        .rounded_sm();
    for (row_index, row) in rows.into_iter().enumerate() {
        let mut table_row = div()
            .w_full()
            .flex()
            .items_start()
            .border_b_1()
            .border_color(line())
            .px_1()
            .py(px(2.));
        for cell in row {
            table_row = table_row.child(
                div()
                    .flex_1()
                    .text_size(px(11.))
                    .font_weight(if row_index == 0 {
                        gpui::FontWeight::BOLD
                    } else {
                        gpui::FontWeight::NORMAL
                    })
                    .text_color(if row_index == 0 {
                        orange_light()
                    } else {
                        text()
                    })
                    .child(cell),
            );
        }
        table = table.child(table_row);
    }
    table
}

pub(crate) fn code_block(value: &str, language: &str) -> gpui::Div {
    let mut block = div()
        .w_full()
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
    block.child(value.to_string())
}
