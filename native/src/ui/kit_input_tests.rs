use super::{bind_form_keys, traverse_form_focus, FocusNext, FocusPrevious, Submit};
use gpui_kit::component::input::{Input, InputState, Textarea, TextareaState};
use gpui_kit::component::theme::Theme;
use gpui_kit::prelude::*;
use gpui_kit::test::TestWindowExt;
use gpui_kit::{div, AppContext, Context, Entity, Focusable, TestAppContext, Window, WindowHandle};

struct Form {
    first: Entity<InputState>,
    second: Entity<InputState>,
    composer: Entity<TextareaState>,
    submits: usize,
}

impl Render for Form {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .child(Input::new(&self.first).id("first"))
            .child(Input::new(&self.second).id("second"))
            .child(div().id("composer").child(Textarea::new(&self.composer)))
            .on_action(
                cx.listener(|_, _: &FocusNext, window, cx| traverse_form_focus(window, cx, false)),
            )
            .on_action(
                cx.listener(|_, _: &FocusPrevious, window, cx| {
                    traverse_form_focus(window, cx, true)
                }),
            )
            .on_action(cx.listener(|this, _: &Submit, _, _| this.submits += 1))
    }
}

fn form(cx: &mut TestAppContext) -> WindowHandle<Form> {
    cx.update(gpui_kit::init);
    cx.update(bind_form_keys);
    cx.add_window(|window, cx| Form {
        first: cx.new(|cx| InputState::new(window, cx)),
        second: cx.new(|cx| InputState::new(window, cx)),
        composer: cx.new(|cx| TextareaState::new(window, cx).auto_grow(1, 5)),
        submits: 0,
    })
}

#[gpui_kit::test]
fn cephalon_palette_reaches_kit_button_tokens(cx: &mut TestAppContext) {
    cx.update(gpui_kit::init);
    cx.update(super::theme::apply_to_kit);
    cx.update(|cx| {
        let theme = Theme::global(cx);
        assert_eq!(theme.tokens.button_primary.color, theme.button_primary);
        assert_eq!(
            theme.tokens.button_primary.color,
            super::theme::orange().into()
        );
    });
}

#[gpui_kit::test]
fn kit_input_keeps_unicode_in_one_editor(cx: &mut TestAppContext) {
    let handle = form(cx);
    cx.update_window(handle.into(), |_, window, cx| {
        window.click("first", cx);
        window.input("A🦀中文", cx);
        window.click("second", cx);
        window.input("é", cx);
        assert_eq!(window.find("first").value(), Some("A🦀中文"));
        assert_eq!(window.find("second").value(), Some("é"));
    })
    .unwrap();
}

#[gpui_kit::test]
fn cut_without_selection_keeps_form_value(cx: &mut TestAppContext) {
    let handle = form(cx);
    cx.update_window(handle.into(), |_, window, cx| {
        window.click("first", cx);
        window.input("keep this line", cx);
        window.press("ctrl-x", cx);
        assert_eq!(window.find("first").value(), Some("keep this line"));
    })
    .unwrap();
}

#[gpui_kit::test]
fn form_tab_traversal_and_submission(cx: &mut TestAppContext) {
    let handle = form(cx);
    let composer = handle
        .update(cx, |form, _, _| form.composer.clone())
        .unwrap();
    cx.update_window(handle.into(), |_, window, cx| {
        window.click("first", cx);
        window.press("tab", cx);
        assert_eq!(window.find("second").focused(), Some(true));
        window.press("shift-tab", cx);
        assert_eq!(window.find("first").focused(), Some(true));
        composer.read(cx).focus_handle(cx).focus(window, cx);
        window.input("line one", cx);
        window.press("secondary-enter", cx);
    })
    .unwrap();
    handle
        .update(cx, |form, _, cx| {
            assert_eq!(form.composer.read(cx).value(), "line one");
            assert_eq!(form.submits, 1);
        })
        .unwrap();
}
