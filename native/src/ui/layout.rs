use gpui::{px, Pixels};

/// The workbench keeps the primary chat surface available at every supported width.
///
/// Wide windows can afford two docked secondary panels. Medium windows keep the
/// details panel docked and turn Library into a drawer. Narrow windows use drawers
/// for both secondary surfaces so a panel request never becomes a no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutMode {
    Wide,
    Medium,
    Narrow,
}

pub(crate) fn layout_mode(width: Pixels) -> LayoutMode {
    if width < px(1260.) {
        LayoutMode::Narrow
    } else if width < px(1660.) {
        LayoutMode::Medium
    } else {
        LayoutMode::Wide
    }
}

pub(crate) fn compact_navigation(width: Pixels) -> bool {
    width < px(1420.)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breakpoint_boundaries_keep_secondary_panels_reachable() {
        assert_eq!(layout_mode(px(1180.)), LayoutMode::Narrow);
        assert_eq!(layout_mode(px(1259.)), LayoutMode::Narrow);
        assert_eq!(layout_mode(px(1260.)), LayoutMode::Medium);
        assert_eq!(layout_mode(px(1366.)), LayoutMode::Medium);
        assert_eq!(layout_mode(px(1420.)), LayoutMode::Medium);
        assert_eq!(layout_mode(px(1499.)), LayoutMode::Medium);
        assert_eq!(layout_mode(px(1500.)), LayoutMode::Medium);
        assert_eq!(layout_mode(px(1659.)), LayoutMode::Medium);
        assert_eq!(layout_mode(px(1660.)), LayoutMode::Wide);
        assert_eq!(layout_mode(px(1750.)), LayoutMode::Wide);
        assert_eq!(layout_mode(px(1920.)), LayoutMode::Wide);
    }

    #[test]
    fn compact_navigation_has_a_single_boundary() {
        assert!(compact_navigation(px(1419.)));
        assert!(!compact_navigation(px(1420.)));
    }
}
