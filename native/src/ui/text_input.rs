use gpui::prelude::*;
use gpui::{
    actions, div, px, App, Context, CursorStyle, Entity, EntityInputHandler, EventEmitter,
    FocusHandle, Focusable, SharedString, Subscription, Window,
};
use gpui_elements::editable_text::{
    self,
    actions::{Cut, EditableTextActionHandler},
    EditableTextState, StringStorage,
};

use super::theme::{
    input_caret, input_marked, input_placeholder, input_selection, line, orange, panel_2, text,
};

actions!([Submit, FocusNextInput, FocusPreviousInput, CutSelectionOnly]);

#[derive(Debug, Clone, Copy)]
pub struct TextSubmitted;

/// A small Cephalon-facing wrapper around GPUI-CE's maintained editable text element.
///
/// GPUI-CE owns the text, selection, IME, clipboard, and caret state. This
/// wrapper keeps Cephalon's form styling and form-specific key semantics.
pub struct TextInput {
    id: SharedString,
    placeholder: SharedString,
    multiline: bool,
    focus_handle: FocusHandle,
    state: Entity<EditableTextState>,
    _state_subscription: Subscription,
}

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
        let focus_handle = state.read(cx).focus_handle(cx).tab_stop(true);
        // A supplied state does not get the observer that a keyed editor state gets.
        let state_subscription = cx.subscribe(
            &state,
            |_, _, _: &gpui_elements::editable_text::TextChanged, cx| cx.notify(),
        );
        Self {
            id,
            placeholder: placeholder.into(),
            multiline,
            focus_handle,
            state,
            _state_subscription: state_subscription,
        }
    }

    pub fn text(&self, cx: &App) -> String {
        self.state.read(cx).as_str().to_owned()
    }

    pub fn editor_state(&self) -> Entity<EditableTextState> {
        self.state.clone()
    }

    pub fn set_text(&mut self, value: impl Into<String>, cx: &mut Context<Self>) {
        let value = value.into();
        self.state.update(cx, |state, cx| state.emplace(&value, cx));
    }

    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(TextSubmitted);
    }

    fn focus_next_input(
        &mut self,
        _: &FocusNextInput,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus_next(cx);
    }

    fn focus_previous_input(
        &mut self,
        _: &FocusPreviousInput,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus_prev(cx);
    }

    /// Keep ordinary form fields from inheriting the editor convention of cutting
    /// the current line when there is no selection. GPUI-CE still owns selection,
    /// clipboard and mutation behavior; this adapter only declines the action when
    /// its maintained selection is empty.
    fn cut_selection_only(
        &mut self,
        _: &CutSelectionOnly,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.state.update(cx, |state, cx| {
            let Some(selection) = state.selected_text_range(false, window, cx) else {
                return;
            };
            if selection.range.start != selection.range.end {
                state.cut(&Cut, window, cx);
            }
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let id = self.id.to_string();
        let state = self.state.downgrade();
        let placeholder = self.placeholder.clone();
        let focused = self.focus_handle.is_focused(window);

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
        let input = input
            .text_color(text())
            .placeholder_color(input_placeholder())
            .selection_color(input_selection())
            .caret_color(input_caret())
            .marked_color(input_marked());

        div()
            .w_full()
            .key_context("TextInput")
            .track_focus(&self.focus_handle)
            .tab_stop(true)
            .cursor(CursorStyle::IBeam)
            .text_size(px(12.))
            .bg(panel_2())
            .border_1()
            .border_color(if focused { orange() } else { line() })
            .rounded_sm()
            .overflow_hidden()
            .on_action(cx.listener(Self::submit))
            .on_action(cx.listener(Self::focus_next_input))
            .on_action(cx.listener(Self::focus_previous_input))
            .on_action(cx.listener(Self::cut_selection_only))
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
    use gpui::{
        AnyWindowHandle, AppContext, InputEvent, KeyBinding, KeyDownEvent, KeyUpEvent, Keystroke,
        TestAppContext,
    };

    struct FormHarness {
        first: Entity<TextInput>,
        second: Entity<TextInput>,
    }

    impl Render for FormHarness {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .flex()
                .flex_col()
                .child(self.first.clone())
                .child(self.second.clone())
        }
    }

    fn dispatch_key(cx: &mut TestAppContext, window: AnyWindowHandle, key: &str) {
        let keystroke = Keystroke::parse(key).unwrap();
        cx.update_window(window, |_, window, cx| {
            window.dispatch_event(
                KeyDownEvent {
                    keystroke,
                    is_held: false,
                    prefer_character_input: false,
                }
                .to_platform_input(),
                cx,
            );
        })
        .unwrap();
        let keystroke = Keystroke::parse(key).unwrap();
        cx.update_window(window, |_, window, cx| {
            window.dispatch_event(KeyUpEvent { keystroke }.to_platform_input(), cx);
        })
        .unwrap();
    }

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

    #[gpui::test]
    fn wrapper_reads_the_editor_as_the_only_text_state(cx: &mut TestAppContext) {
        let input = cx.update(|cx| cx.new(|cx| TextInput::new(cx, "test", "", "", true)));
        let editor = cx.read_entity(&input, |input, _| input.editor_state());
        cx.update(|cx| {
            editor.update(cx, |editor, cx| editor.emplace("café ✅\n第二行", cx));
        });
        cx.read_entity(&input, |input, cx| {
            assert_eq!(input.text(cx), "café ✅\n第二行");
        });
    }

    #[gpui::test]
    fn form_cut_and_tab_navigation_keep_editor_semantics(cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.bind_keys([
                KeyBinding::new("ctrl-x", CutSelectionOnly, Some("EditableText")),
                KeyBinding::new("tab", FocusNextInput, Some("EditableText")),
                KeyBinding::new("shift-tab", FocusPreviousInput, Some("EditableText")),
            ]);
        });
        let first = cx.update(|cx| cx.new(|cx| TextInput::new(cx, "first", "café ✅", "", false)));
        let second = cx.update(|cx| cx.new(|cx| TextInput::new(cx, "second", "", "", false)));
        let first_focus = cx.read_entity(&first, |input, _| input.focus_handle());
        let second_focus = cx.read_entity(&second, |input, _| input.focus_handle());
        let window: AnyWindowHandle = cx
            .add_window({
                let first = first.clone();
                let second = second.clone();
                move |_, _| FormHarness { first, second }
            })
            .into();
        cx.update_window(window, |_, window, cx| {
            window.focus(&first_focus, cx);
            window.draw(cx).clear(cx);
        })
        .unwrap();

        dispatch_key(cx, window, "ctrl-x");
        cx.read_entity(&first, |input, cx| assert_eq!(input.text(cx), "café ✅"));

        dispatch_key(cx, window, "tab");
        cx.update_window(window, |_, window, _| {
            assert!(second_focus.is_focused(window))
        })
        .unwrap();
        dispatch_key(cx, window, "shift-tab");
        cx.update_window(window, |_, window, _| {
            assert!(first_focus.is_focused(window))
        })
        .unwrap();
    }
}
