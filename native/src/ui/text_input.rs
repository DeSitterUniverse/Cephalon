use gpui::prelude::*;
use gpui::{
    actions, div, px, App, Context, CursorStyle, Entity, EventEmitter, FocusHandle, Focusable,
    SharedString, Subscription, Window,
};
use gpui_elements::editable_text::{self, EditableTextState, StringStorage};

actions!([Submit]);

#[derive(Debug, Clone, Copy)]
pub struct TextChanged;

#[derive(Debug, Clone, Copy)]
pub struct TextSubmitted;

/// A small Cephalon-facing wrapper around GPUI-CE's maintained editable text element.
///
/// Keeping an entity wrapper lets the rest of the application continue to subscribe to
/// typed changes and mirror values into its draft state, while the actual editing behavior
/// (selection, IME, clipboard, wrapping, caret scrolling, and navigation) remains in GPUI-CE.
pub struct TextInput {
    id: SharedString,
    content: SharedString,
    placeholder: SharedString,
    multiline: bool,
    focus_handle: FocusHandle,
    state: Entity<EditableTextState>,
    _state_subscription: Subscription,
}

impl EventEmitter<TextChanged> for TextInput {}
impl EventEmitter<TextSubmitted> for TextInput {}

impl TextInput {
    pub fn new(
        cx: &mut Context<Self>,
        id: impl Into<SharedString>,
        content: impl Into<SharedString>,
        placeholder: impl Into<SharedString>,
        multiline: bool,
    ) -> Self {
        let id = id.into();
        let content = content.into();
        let state =
            cx.new(|cx| EditableTextState::new(StringStorage::from(content.to_string()), cx));
        let focus_handle = state.read(cx).focus_handle(cx);
        let state_subscription = cx.subscribe(
            &state,
            |this, state, _: &gpui_elements::editable_text::TextChanged, cx| {
                let content = state.read_with(cx, |state, _| state.as_str().to_owned());
                this.content = content.into();
                cx.emit(TextChanged);
                cx.notify();
            },
        );

        Self {
            id,
            content,
            placeholder: placeholder.into(),
            multiline,
            focus_handle,
            state,
            _state_subscription: state_subscription,
        }
    }

    pub fn text(&self) -> &str {
        &self.content
    }

    pub fn set_text(&mut self, value: impl Into<String>, cx: &mut Context<Self>) {
        let value = value.into();
        self.content = value.clone().into();
        self.state.update(cx, |state, cx| state.emplace(&value, cx));
        cx.notify();
    }

    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(TextSubmitted);
    }
}

impl Render for TextInput {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let id = self.id.to_string();
        let state = self.state.downgrade();
        let placeholder = self.placeholder.clone();

        let input = if self.multiline {
            editable_text::text_area(id)
                .state(state)
                .placeholder(placeholder)
                .w_full()
                .min_h(px(36.))
                .max_h(px(132.))
                .p_2()
                .whitespace_normal()
                .overflow_y_scroll()
        } else {
            editable_text::text_input(id)
                .state(state)
                .placeholder(placeholder)
                .w_full()
                .h(px(36.))
                .min_h(px(36.))
                .p_2()
                .whitespace_nowrap()
                .overflow_x_scroll()
        };

        div()
            .w_full()
            .key_context("TextInput")
            .track_focus(&self.focus_handle)
            .cursor(CursorStyle::IBeam)
            .text_size(px(12.))
            .on_action(cx.listener(Self::submit))
            .child(input)
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;

    #[gpui::test]
    fn editable_text_state_round_trips_unicode_and_newlines(cx: &mut TestAppContext) {
        let state =
            cx.update(|cx| cx.new(|cx| EditableTextState::new(StringStorage::from(""), cx)));
        cx.update(|cx| {
            state.update(cx, |state, cx| state.emplace("café ✅\n第二行", cx));
        });
        cx.read_entity(&state, |state, _| {
            assert_eq!(state.as_str(), "café ✅\n第二行");
        });
    }
}
