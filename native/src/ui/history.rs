//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_history(&mut self, cx: &mut Context<Self>) -> gpui::Div {
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
                .child(
                    div()
                        .flex_1()
                        .overflow_hidden()
                        .truncate()
                        .text_color(text())
                        .child(title),
                );
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
            .child(list)
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
}
