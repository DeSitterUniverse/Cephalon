//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_library(&mut self, width: f32, cx: &mut Context<Self>) -> gpui::Div {
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
                .child(
                    div()
                        .flex_1()
                        .overflow_hidden()
                        .truncate()
                        .text_color(text())
                        .child(document.name.clone()),
                )
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
            .w(px(width))
            .h_full()
            .p_3()
            .flex()
            .flex_col()
            .gap_2()
            .bg(panel())
            .border_r_1()
            .border_color(line())
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_size(px(15.)).text_color(text()).child("Library"))
                    .child(ui_button(
                        "close-library",
                        "×",
                        false,
                        cx.listener(|this, _, _, cx| {
                            this.left_open = false;
                            cx.notify();
                        }),
                    )),
            )
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

    pub(super) fn render_nav(&mut self, compact: bool, cx: &mut Context<Self>) -> gpui::Div {
        let mut nav = div()
            .w(px(if compact { 96. } else { 112. }))
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
}
