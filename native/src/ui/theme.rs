use gpui_kit::component::theme::{Theme, ThemeMode};
use gpui_kit::App;
use std::sync::atomic::{AtomicBool, Ordering};

static GRAPHITE_THEME: AtomicBool = AtomicBool::new(false);

pub(super) fn set_graphite(graphite: bool) {
    GRAPHITE_THEME.store(graphite, Ordering::Relaxed);
}

pub(super) fn apply_to_kit(cx: &mut App) {
    Theme::change(ThemeMode::Dark, None, cx);
    let theme = Theme::global_mut(cx);
    theme.font_size = gpui_kit::px(13.);
    theme.background = bg().into();
    theme.foreground = text().into();
    theme.muted = panel_2().into();
    theme.muted_foreground = muted().into();
    theme.input = panel_2().into();
    theme.border = line().into();
    theme.primary = orange().into();
    theme.primary_hover = orange_light().into();
    theme.button_primary = orange().into();
    theme.button_primary_hover = orange_light().into();
    theme.link = link().into();
    theme.success = green().into();
    theme.warning = yellow().into();
    theme.danger = red().into();
    theme.tokens = theme.colors.into();
    Theme::sync_base(cx);
}

pub(super) fn graphite() -> bool {
    GRAPHITE_THEME.load(Ordering::Relaxed)
}

pub(super) fn bg() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x17191c)
    } else {
        gpui_kit::rgb(0x000000)
    }
}

pub(super) fn panel() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x1c1f23)
    } else {
        gpui_kit::rgb(0x020202)
    }
}

pub(super) fn panel_2() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x23272c)
    } else {
        gpui_kit::rgb(0x080808)
    }
}

pub(super) fn panel_3() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x2d3238)
    } else {
        gpui_kit::rgb(0x121212)
    }
}

pub(super) fn line() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x40474f)
    } else {
        gpui_kit::rgb(0x252525)
    }
}

pub(super) fn text() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0xe6edf3)
    } else {
        gpui_kit::rgb(0xffe5cc)
    }
}

pub(super) fn muted() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0xaab4bf)
    } else {
        gpui_kit::rgb(0x9b9b9b)
    }
}

pub(super) fn faint() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x7e8894)
    } else {
        gpui_kit::rgb(0x666666)
    }
}

pub(super) fn orange() -> gpui_kit::Rgba {
    gpui_kit::rgb(0xff9a2e)
}

pub(super) fn orange_light() -> gpui_kit::Rgba {
    gpui_kit::rgb(0xffbd6b)
}

pub(super) fn link() -> gpui_kit::Rgba {
    if graphite() {
        gpui_kit::rgb(0x79b8ff)
    } else {
        gpui_kit::rgb(0x8fc7ff)
    }
}

pub(super) fn green() -> gpui_kit::Rgba {
    gpui_kit::rgb(0x6ee7a8)
}

pub(super) fn yellow() -> gpui_kit::Rgba {
    gpui_kit::rgb(0xe9b949)
}

pub(super) fn red() -> gpui_kit::Rgba {
    gpui_kit::rgb(0xf87171)
}
