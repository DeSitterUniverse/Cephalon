mod api;
mod backend;
mod ui;

use backend::BackendService;
use gpui::{px, size, App, AppContext, Bounds, KeyBinding, WindowBounds, WindowOptions};
use gpui_elements::editable_text::actions::{default_bindings, DEFAULT_INPUT_CONTEXT};
use gpui_platform::application;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use ui::{
    CutSelectionOnly, FocusNext, FocusNextInput, FocusPrevious, FocusPreviousInput, NativeApp,
    Submit,
};

fn main() {
    application().run(|cx: &mut App| {
        cx.bind_keys(default_bindings().as_keybindings(Some(DEFAULT_INPUT_CONTEXT)));
        cx.bind_keys([
            KeyBinding::new("cmd-enter", Submit, None),
            KeyBinding::new("ctrl-enter", Submit, None),
            KeyBinding::new("tab", FocusNext, None),
            KeyBinding::new("shift-tab", FocusPrevious, None),
            // GPUI-CE's editor defaults are retained for all normal editing
            // actions, while form semantics override only Tab and Cut.
            KeyBinding::new("tab", FocusNextInput, Some(DEFAULT_INPUT_CONTEXT)),
            KeyBinding::new("shift-tab", FocusPreviousInput, Some(DEFAULT_INPUT_CONTEXT)),
            KeyBinding::new("secondary-x", CutSelectionOnly, Some(DEFAULT_INPUT_CONTEXT)),
        ]);
        let api = api::ApiClient::configured();
        let backend = Arc::new(BackendService::new());
        let stop = Arc::new(AtomicBool::new(false));
        let quit_backend = backend.clone();
        let quit_stop = stop.clone();
        let _ = cx.on_app_quit(move |_cx| {
            quit_stop.store(true, std::sync::atomic::Ordering::Relaxed);
            quit_backend.shutdown();
            async {}
        });
        let bounds = Bounds::centered(None, size(px(1660.), px(1060.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(1180.), px(720.))),
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("Cephalon".into()),
                    ..Default::default()
                }),
                icon: application_icon(),
                ..Default::default()
            },
            move |_window, cx| cx.new(|cx| NativeApp::new(api, backend, stop, cx)),
        )
        .expect("failed to open Cephalon window");
    });
}

fn application_icon() -> Option<Arc<image::RgbaImage>> {
    image::load_from_memory(include_bytes!("../../assets/cephalon.png"))
        .ok()
        .map(|image| Arc::new(image.to_rgba8()))
}
