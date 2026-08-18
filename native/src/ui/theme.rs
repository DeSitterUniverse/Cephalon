use std::sync::atomic::{AtomicBool, Ordering};

static GRAPHITE_THEME: AtomicBool = AtomicBool::new(false);

pub(super) fn set_graphite(graphite: bool) {
    GRAPHITE_THEME.store(graphite, Ordering::Relaxed);
}

fn graphite() -> bool {
    GRAPHITE_THEME.load(Ordering::Relaxed)
}

pub(super) fn bg() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0x17191c)
    } else {
        gpui::rgb(0x000000)
    }
}

pub(super) fn panel() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0x1c1f23)
    } else {
        gpui::rgb(0x020202)
    }
}

pub(super) fn panel_2() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0x23272c)
    } else {
        gpui::rgb(0x080808)
    }
}

pub(super) fn panel_3() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0x2d3238)
    } else {
        gpui::rgb(0x121212)
    }
}

pub(super) fn line() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0x40474f)
    } else {
        gpui::rgb(0x252525)
    }
}

pub(super) fn text() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0xe6edf3)
    } else {
        gpui::rgb(0xffe5cc)
    }
}

pub(super) fn muted() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0xaab4bf)
    } else {
        gpui::rgb(0x9b9b9b)
    }
}

pub(super) fn faint() -> gpui::Rgba {
    if graphite() {
        gpui::rgb(0x7e8894)
    } else {
        gpui::rgb(0x666666)
    }
}

pub(super) fn orange() -> gpui::Rgba {
    gpui::rgb(0xff9a2e)
}

pub(super) fn orange_light() -> gpui::Rgba {
    gpui::rgb(0xffbd6b)
}

pub(super) fn green() -> gpui::Rgba {
    gpui::rgb(0x6ee7a8)
}

pub(super) fn yellow() -> gpui::Rgba {
    gpui::rgb(0xe9b949)
}

pub(super) fn red() -> gpui::Rgba {
    gpui::rgb(0xf87171)
}
