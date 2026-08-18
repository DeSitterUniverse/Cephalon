mod api;
mod backend;
mod ui;

use backend::BackendService;
use gpui::{
    px, size, App, AppContext, Application, Bounds, KeyBinding, WindowBounds, WindowOptions,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use ui::{
    Backspace, Copy, Cut, Delete, Down, End, Home, Left, NativeApp, Newline, Paste, Right,
    SelectAll, SelectDown, SelectLeft, SelectRight, SelectUp, Submit, Up,
};

fn main() {
    Application::new().run(|cx: &mut App| {
        cx.bind_keys([
            KeyBinding::new("backspace", Backspace, None),
            KeyBinding::new("delete", Delete, None),
            KeyBinding::new("left", Left, None),
            KeyBinding::new("right", Right, None),
            KeyBinding::new("up", Up, None),
            KeyBinding::new("down", Down, None),
            KeyBinding::new("shift-left", SelectLeft, None),
            KeyBinding::new("shift-right", SelectRight, None),
            KeyBinding::new("shift-up", SelectUp, None),
            KeyBinding::new("shift-down", SelectDown, None),
            KeyBinding::new("cmd-a", SelectAll, None),
            KeyBinding::new("ctrl-a", SelectAll, None),
            KeyBinding::new("home", Home, None),
            KeyBinding::new("end", End, None),
            KeyBinding::new("cmd-enter", Submit, None),
            KeyBinding::new("ctrl-enter", Submit, None),
            KeyBinding::new("enter", Newline, None),
            KeyBinding::new("cmd-v", Paste, None),
            KeyBinding::new("ctrl-v", Paste, None),
            KeyBinding::new("cmd-c", Copy, None),
            KeyBinding::new("ctrl-c", Copy, None),
            KeyBinding::new("cmd-x", Cut, None),
            KeyBinding::new("ctrl-x", Cut, None),
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
                ..Default::default()
            },
            move |_window, cx| cx.new(|cx| NativeApp::new(api, backend, stop, cx)),
        )
        .expect("failed to open Cephalon window");
    });
}
