mod api;
mod backend;
mod ui;

use backend::BackendService;
use gpui::{px, size, App, AppContext, Application, Bounds, WindowBounds, WindowOptions};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use ui::NativeApp;

fn main() {
    Application::new().run(|cx: &mut App| {
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
