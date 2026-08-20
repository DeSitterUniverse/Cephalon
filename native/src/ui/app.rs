//! Native GPUI view rendering for this part of the workbench.

use gpui::prelude::*;

use super::*;

impl NativeApp {
    pub(super) fn render_right_panel(&mut self, width: f32, cx: &mut Context<Self>) -> gpui::Div {
        let content = match self.panel {
            Panel::History => self.render_history(cx),
            Panel::Document => self.render_document(cx),
            Panel::Sources => self.render_sources(cx),
            Panel::Settings => self.render_settings(cx),
            Panel::Trace => self.render_trace(cx),
            Panel::Health => self.render_health(cx),
            Panel::Evaluation => self.render_evaluation(cx),
            Panel::Support => self.render_support(cx),
        };
        div()
            .w(px(width))
            .h_full()
            .p_3()
            .flex()
            .flex_col()
            .gap_2()
            .bg(panel())
            .border_l_1()
            .border_color(line())
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_size(px(15.))
                            .text_color(text())
                            .child(self.panel.title()),
                    )
                    .child(ui_button(
                        "close-details",
                        "×",
                        false,
                        cx.listener(|this, _, _, cx| {
                            this.right_open = false;
                            cx.notify();
                        }),
                    )),
            )
            .child(
                div()
                    .id("right-panel-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(content),
            )
    }

    pub(super) fn render_overlays(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let mut notices = div().flex().flex_col().gap_1();
        for notice in &self.notices {
            notices = notices.child(
                div()
                    .id(SharedString::from(format!("notice-{}", notice.id)))
                    .p_2()
                    .bg(panel_3())
                    .border_1()
                    .border_color(notice.color)
                    .text_size(px(12.))
                    .text_color(notice.color)
                    .child(notice.message.clone()),
            );
        }
        let mut overlay = div().w_full().p_3().child(notices);
        if let Some(confirmation) = self.confirmation.clone() {
            overlay = overlay.child(self.render_confirmation_dialog(confirmation, cx));
        }
        overlay
    }

    fn render_confirmation_dialog(
        &mut self,
        confirmation: Confirmation,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        div()
            .id("confirmation-backdrop")
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui::hsla(0., 0., 0., 0.62))
            .block_mouse_except_scroll()
            .child(
                div()
                    .id("confirmation-dialog")
                    .track_focus(&self.confirmation_focus)
                    .tab_stop(false)
                    .role(gpui::Role::Dialog)
                    .aria_label(confirmation.title.clone())
                    .w(px(420.))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .bg(panel_3())
                    .border_1()
                    .border_color(orange())
                    .child(div().text_color(text()).child(confirmation.title))
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(muted())
                            .child(confirmation.message),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child(ui_button_with_focus(
                                "confirm-action",
                                "Confirm",
                                true,
                                &self.confirmation_confirm_focus,
                                cx.listener(|this, _, window, cx| this.confirm_action(window, cx)),
                            ))
                            .child(ui_button_with_focus(
                                "cancel-action",
                                "Cancel",
                                false,
                                &self.confirmation_cancel_focus,
                                cx.listener(|this, _, window, cx| {
                                    this.close_confirmation(window, cx)
                                }),
                            )),
                    ),
            )
    }

    pub(super) fn render_boot(&mut self, cx: &mut Context<Self>) -> gpui::Div {
        let retry = if self.boot == BootState::Failed {
            ui_button(
                "boot-retry",
                "Retry backend",
                true,
                cx.listener(|this, _, _, cx| this.retry_backend(cx)),
            )
        } else {
            div()
                .id("boot-loading-status")
                .flex_none()
                .px_2()
                .py_1()
                .text_size(px(12.))
                .text_color(faint())
                .child("Waiting for local backend…")
        };
        let mut card = div()
            .w(px(540.))
            .p_6()
            .bg(panel_2())
            .border_1()
            .border_color(line())
            .rounded_sm()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .text_size(px(24.))
                    .text_color(text())
                    .child("Cephalon"),
            )
            .child(
                div()
                    .text_color(if self.boot == BootState::Failed {
                        red()
                    } else {
                        yellow()
                    })
                    .child(self.boot_status.clone()),
            );
        if let Some(error) = &self.boot_error {
            card = card.child(
                div()
                    .p_3()
                    .bg(panel_3())
                    .border_1()
                    .border_color(line())
                    .text_color(red())
                    .child(error.clone()),
            );
        }
        card = card.child(retry);
        div()
            .size_full()
            .relative()
            .flex()
            .items_center()
            .justify_center()
            .bg(bg())
            .child(card)
    }

    pub(super) fn render_shell(&mut self, window: &Window, cx: &mut Context<Self>) -> gpui::Div {
        let viewport_width = window.viewport_size().width;
        let mode = layout_mode(viewport_width);
        let compact = compact_navigation(viewport_width);
        let show_library_docked = self.left_open && mode == LayoutMode::Wide;
        let show_right_panel_docked = self.right_open && mode != LayoutMode::Narrow;
        let show_library_drawer = self.left_open
            && mode != LayoutMode::Wide
            && (mode == LayoutMode::Medium || self.panel == Panel::History || !self.right_open);
        let show_right_panel_drawer = self.right_open
            && mode == LayoutMode::Narrow
            && (self.panel != Panel::History || !self.left_open);
        let library_width = if compact { 260. } else { 300. };
        let right_width = if viewport_width < px(1750.) {
            330.
        } else {
            390.
        };
        let model_name = self
            .data
            .models
            .active_model
            .clone()
            .or_else(|| {
                self.data
                    .server
                    .as_ref()
                    .map(|server| server.model_name.clone())
            })
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "No model connected".into());
        let model_color = if self.data.models.active_model.is_some() {
            green()
        } else {
            yellow()
        };
        let mut body = div().relative().flex().flex_1().min_h_0().min_w_0();
        if show_library_docked {
            body = body.child(self.render_library(library_width, cx));
        }
        body = body
            .child(self.render_nav(compact, cx))
            .child(self.render_chat(cx));
        if show_right_panel_docked {
            body = body.child(self.render_right_panel(right_width, cx));
        }
        if show_library_drawer || show_right_panel_drawer {
            let mut drawers = div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .flex()
                .justify_between();
            if show_library_drawer {
                drawers = drawers.child(self.render_library(library_width, cx));
            }
            if show_right_panel_drawer {
                drawers = drawers.child(self.render_right_panel(right_width, cx));
            }
            body = body.child(drawers);
        }
        let topbar = div()
            .h(px(58.))
            .w_full()
            .px_4()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .bg(panel_2())
            .border_b_1()
            .border_color(line())
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .flex_1()
                    .min_w_0()
                    .child(
                        div()
                            .text_size(px(18.))
                            .text_color(orange_light())
                            .child("CEPHALON"),
                    )
                    .child(ui_button(
                        "new-chat",
                        "+ New chat",
                        false,
                        cx.listener(|this, _, _, cx| this.new_conversation(cx)),
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .max_w(px(if compact { 180. } else { 300. }))
                            .overflow_hidden()
                            .truncate()
                            .text_color(model_color)
                            .child(model_name),
                    )
                    .child(ui_button(
                        "connect-model",
                        "Connect",
                        self.data.models.active_model.is_some(),
                        cx.listener(|this, _, _, cx| this.connect_model(cx)),
                    ))
                    .child(if compact {
                        div()
                    } else {
                        div()
                            .text_size(px(11.))
                            .text_color(self.event_status.color())
                            .child(format!("● {}", self.event_status.label()))
                    })
                    .child(ui_button(
                        "toggle-library",
                        if self.left_open {
                            "Hide library"
                        } else {
                            "Show library"
                        },
                        false,
                        cx.listener(|this, _, window, cx| {
                            if layout_mode(window.viewport_size().width) != LayoutMode::Wide {
                                this.right_open = false;
                            }
                            this.left_open = !this.left_open;
                            cx.notify();
                        }),
                    ))
                    .child(ui_button(
                        "toggle-details",
                        if self.right_open {
                            "Hide details"
                        } else {
                            "Show details"
                        },
                        false,
                        cx.listener(|this, _, window, cx| {
                            if layout_mode(window.viewport_size().width) == LayoutMode::Narrow {
                                this.left_open = false;
                            }
                            this.right_open = !this.right_open;
                            cx.notify();
                        }),
                    )),
            );
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(bg())
            .text_color(text())
            .child(topbar)
            .child(body)
            .child(self.render_overlays(cx))
    }
}
