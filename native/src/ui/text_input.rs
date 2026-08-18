use std::ops::Range;

use gpui::prelude::*;
use gpui::{
    actions, div, fill, point, px, relative, rgba, App, Bounds, ClipboardItem, Context,
    CursorStyle, Element, ElementId, ElementInputHandler, Entity, EntityInputHandler, EventEmitter,
    FocusHandle, Focusable, GlobalElementId, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, PaintQuad, Pixels, Point, ShapedLine, SharedString, Style, TextRun,
    UTF16Selection, Window,
};
use unicode_segmentation::UnicodeSegmentation;

actions!([
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    SelectLeft,
    SelectRight,
    SelectUp,
    SelectDown,
    SelectAll,
    Home,
    End,
    Paste,
    Cut,
    Copy,
    Newline,
    Submit,
]);

#[derive(Debug, Clone, Copy)]
pub struct TextChanged;

#[derive(Debug, Clone, Copy)]
pub struct TextSubmitted;

pub struct TextInput {
    pub id: SharedString,
    pub focus_handle: FocusHandle,
    pub content: SharedString,
    pub placeholder: SharedString,
    pub multiline: bool,
    pub selected_range: Range<usize>,
    pub selection_reversed: bool,
    pub marked_range: Option<Range<usize>>,
    pub last_layouts: Vec<(Range<usize>, ShapedLine)>,
    pub last_bounds: Option<Bounds<Pixels>>,
    is_selecting: bool,
    line_height: f32,
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
        let content = content.into();
        let end = content.len();
        Self {
            id: id.into(),
            focus_handle: cx.focus_handle(),
            content,
            placeholder: placeholder.into(),
            multiline,
            selected_range: end..end,
            selection_reversed: false,
            marked_range: None,
            last_layouts: Vec::new(),
            last_bounds: None,
            is_selecting: false,
            line_height: 18.0,
        }
    }

    pub fn text(&self) -> &str {
        &self.content
    }

    pub fn set_text(&mut self, value: impl Into<SharedString>, cx: &mut Context<Self>) {
        let value = value.into();
        if self.content == value {
            return;
        }
        let end = value.len();
        self.content = value;
        self.selected_range = end..end;
        self.selection_reversed = false;
        self.marked_range = None;
        self.emit_changed(cx);
    }

    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    fn emit_changed(&mut self, cx: &mut Context<Self>) {
        self.last_layouts.clear();
        self.last_bounds = None;
        cx.emit(TextChanged);
        cx.notify();
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else if self.selection_reversed {
            self.move_to(self.selected_range.end, cx);
        } else {
            self.move_to(self.selected_range.start, cx);
        }
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.cursor_offset()), cx);
        } else if self.selection_reversed {
            self.move_to(self.selected_range.start, cx);
        } else {
            self.move_to(self.selected_range.end, cx);
        }
    }

    fn up(&mut self, _: &Up, _: &mut Window, cx: &mut Context<Self>) {
        self.move_vertical(-1, false, cx);
    }

    fn down(&mut self, _: &Down, _: &mut Window, cx: &mut Context<Self>) {
        self.move_vertical(1, false, cx);
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_up(&mut self, _: &SelectUp, _: &mut Window, cx: &mut Context<Self>) {
        self.move_vertical(-1, true, cx);
    }

    fn select_down(&mut self, _: &SelectDown, _: &mut Window, cx: &mut Context<Self>) {
        self.move_vertical(1, true, cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
        self.select_to(self.content.len(), cx);
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        let offset = self.cursor_offset();
        let range = self
            .line_ranges()
            .into_iter()
            .find(|range| range.contains(&offset));
        self.move_to(range.map(|range| range.start).unwrap_or(0), cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        let offset = self.cursor_offset();
        let range = self
            .line_ranges()
            .into_iter()
            .find(|range| range.contains(&offset) || offset == range.end);
        self.move_to(
            range.map(|range| range.end).unwrap_or(self.content.len()),
            cx,
        );
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn newline(&mut self, _: &Newline, window: &mut Window, cx: &mut Context<Self>) {
        if self.multiline {
            self.replace_text_in_range(None, "\n", window, cx);
        }
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(TextSubmitted);
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle);
        self.is_selecting = true;
        let offset = self.index_for_mouse_position(event.position);
        if event.modifiers.shift {
            self.select_to(offset, cx);
        } else {
            self.move_to(offset, cx);
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        }
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            let text = if self.multiline {
                text
            } else {
                text.replace(['\r', '\n'], " ")
            };
            self.replace_text_in_range(None, &text, window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            self.replace_text_in_range(None, "", window, cx);
        }
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        self.selection_reversed = false;
        cx.notify();
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn move_vertical(&mut self, direction: i32, selecting: bool, cx: &mut Context<Self>) {
        let ranges = self.line_ranges();
        let offset = self.cursor_offset();
        let Some(current) = ranges
            .iter()
            .position(|range| range.contains(&offset) || offset == range.end)
        else {
            return;
        };
        let target = current as i32 + direction;
        if target < 0 || target >= ranges.len() as i32 {
            return;
        }
        let column = offset.saturating_sub(ranges[current].start);
        let target_range = &ranges[target as usize];
        let target_offset = target_range.start + column.min(target_range.len());
        if selecting {
            self.select_to(target_offset, cx);
        } else {
            self.move_to(target_offset, cx);
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        let Some(bounds) = self.last_bounds.as_ref() else {
            return self.content.len();
        };
        if position.y < bounds.top() {
            return 0;
        }
        let line_index = (f32::from(position.y - bounds.top()) / self.line_height).floor() as usize;
        let ranges = self.line_ranges();
        let Some((range, line)) = ranges
            .get(line_index.min(ranges.len().saturating_sub(1)))
            .zip(
                self.last_layouts
                    .get(line_index.min(self.last_layouts.len().saturating_sub(1))),
            )
        else {
            return self.content.len();
        };
        let x = (position.x - bounds.left()).max(px(0.));
        range.start + line.1.closest_index_for_x(x).min(range.len())
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        let offset = offset.min(self.content.len());
        if self.selection_reversed {
            self.selected_range.start = offset;
        } else {
            self.selected_range.end = offset;
        }
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify();
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        let mut utf8_offset = 0;
        let mut utf16_count = 0;
        for ch in self.content.chars() {
            if utf16_count >= offset {
                break;
            }
            utf16_count += ch.len_utf16();
            utf8_offset += ch.len_utf8();
        }
        utf8_offset.min(self.content.len())
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16_offset = 0;
        let mut utf8_count = 0;
        for ch in self.content.chars() {
            if utf8_count >= offset {
                break;
            }
            utf8_count += ch.len_utf8();
            utf16_offset += ch.len_utf16();
        }
        utf16_offset
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range.start)..self.offset_from_utf16(range.end)
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .rev()
            .find_map(|(index, _)| (index < offset).then_some(index))
            .unwrap_or(0)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .find_map(|(index, _)| (index > offset).then_some(index))
            .unwrap_or(self.content.len())
    }

    fn line_ranges(&self) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let mut start = 0;
        for (index, character) in self.content.char_indices() {
            if character == '\n' {
                ranges.push(start..index);
                start = index + character.len_utf8();
            }
        }
        ranges.push(start..self.content.len());
        ranges
    }

    fn line_height(&self) -> Pixels {
        px(self.line_height)
    }

    fn viewport_height(&self) -> Pixels {
        let line_count = self.line_ranges().len().max(1) as f32;
        px((line_count.min(6.0) * self.line_height + 16.0).max(36.0))
    }
}

impl EntityInputHandler for TextInput {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(
        &self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _window: &mut Window, _cx: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or_else(|| self.selected_range.clone());
        let replacement = if self.multiline {
            new_text.to_string()
        } else {
            new_text.replace(['\r', '\n'], " ")
        };
        self.content =
            (self.content[..range.start].to_owned() + &replacement + &self.content[range.end..])
                .into();
        let end = range.start + replacement.len();
        self.selected_range = end..end;
        self.selection_reversed = false;
        self.marked_range = None;
        self.emit_changed(cx);
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or_else(|| self.selected_range.clone());
        self.content =
            (self.content[..range.start].to_owned() + new_text + &self.content[range.end..]).into();
        self.marked_range =
            (!new_text.is_empty()).then_some(range.start..range.start + new_text.len());
        self.selected_range = new_selected_range_utf16
            .as_ref()
            .map(|new_range| self.range_from_utf16(new_range))
            .map(|new_range| new_range.start + range.start..new_range.end + range.start)
            .unwrap_or_else(|| range.start + new_text.len()..range.start + new_text.len());
        self.selection_reversed = false;
        self.emit_changed(cx);
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        bounds: Bounds<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let range = self.range_from_utf16(&range_utf16);
        let line_ranges = self.line_ranges();
        let line_index = line_ranges
            .iter()
            .position(|line| line.contains(&range.start) || range.start == line.end)
            .unwrap_or(0);
        let line_range = line_ranges.get(line_index)?;
        let line = self.last_layouts.get(line_index)?;
        let start = range.start.clamp(line_range.start, line_range.end) - line_range.start;
        let end = range.end.clamp(line_range.start, line_range.end) - line_range.start;
        Some(Bounds::from_corners(
            point(
                bounds.left() + line.1.x_for_index(start),
                bounds.top() + self.line_height() * line_index as f32,
            ),
            point(
                bounds.left() + line.1.x_for_index(end),
                bounds.top() + self.line_height() * (line_index + 1) as f32,
            ),
        ))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        let bounds = self.last_bounds?;
        let ranges = self.line_ranges();
        let line_index = (f32::from(point.y - bounds.top()) / self.line_height).floor() as usize;
        let range = ranges.get(line_index.min(ranges.len().saturating_sub(1)))?;
        let line = self
            .last_layouts
            .get(line_index.min(self.last_layouts.len().saturating_sub(1)))?;
        let index = line
            .1
            .index_for_x(point.x - bounds.left())
            .unwrap_or(range.len());
        Some(self.offset_to_utf16(range.start + index.min(range.len())))
    }
}

struct TextElement {
    input: Entity<TextInput>,
}

struct PrepaintState {
    lines: Vec<(Range<usize>, ShapedLine)>,
    cursor: Option<PaintQuad>,
    selections: Vec<PaintQuad>,
}

impl IntoElement for TextElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for TextElement {
    type RequestLayoutState = ();
    type PrepaintState = PrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let input = self.input.read(cx);
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = (input.line_height() * input.line_ranges().len().max(1) as f32).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let input = self.input.read(cx);
        let content = input.content.clone();
        let ranges = input.line_ranges();
        let style = window.text_style();
        let color = if content.is_empty() {
            muted_color()
        } else {
            style.color
        };
        let display = if content.is_empty() {
            vec![(0..0, input.placeholder.clone())]
        } else {
            ranges
                .iter()
                .map(|range| (range.clone(), content[range.clone()].to_string().into()))
                .collect()
        };
        let font_size = style.font_size.to_pixels(window.rem_size());
        let mut lines = Vec::with_capacity(display.len());
        for (range, text) in display {
            let run = TextRun {
                len: text.len(),
                font: style.font(),
                color,
                background_color: None,
                underline: None,
                strikethrough: None,
            };
            let shaped = window
                .text_system()
                .shape_line(text, font_size, &[run], None);
            lines.push((range, shaped));
        }

        let mut selections = Vec::new();
        let selection = input.selected_range.clone();
        if !selection.is_empty() && !content.is_empty() {
            for (line_index, (range, line)) in lines.iter().enumerate() {
                let start = selection.start.max(range.start);
                let end = selection.end.min(range.end);
                if start < end {
                    selections.push(fill(
                        Bounds::from_corners(
                            point(
                                bounds.left() + line.x_for_index(start - range.start),
                                bounds.top() + input.line_height() * line_index as f32,
                            ),
                            point(
                                bounds.left() + line.x_for_index(end - range.start),
                                bounds.top() + input.line_height() * (line_index + 1) as f32,
                            ),
                        ),
                        rgba(0x553f2a1f),
                    ));
                }
            }
        }

        let cursor_offset = input.cursor_offset();
        let cursor = if selection.is_empty() && !content.is_empty() {
            lines
                .iter()
                .enumerate()
                .find_map(|(line_index, (range, line))| {
                    if cursor_offset >= range.start && cursor_offset <= range.end {
                        Some(fill(
                            Bounds::new(
                                point(
                                    bounds.left() + line.x_for_index(cursor_offset - range.start),
                                    bounds.top() + input.line_height() * line_index as f32,
                                ),
                                gpui::size(px(2.), bounds.bottom() - bounds.top()),
                            ),
                            gpui::rgb(0xffe5cc),
                        ))
                    } else {
                        None
                    }
                })
        } else {
            None
        };

        PrepaintState {
            lines,
            cursor,
            selections,
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus_handle = self.input.read(cx).focus_handle.clone();
        window.handle_input(
            &focus_handle,
            ElementInputHandler::new(bounds, self.input.clone()),
            cx,
        );
        for selection in prepaint.selections.drain(..) {
            window.paint_quad(selection);
        }
        let line_height = self.input.read(cx).line_height();
        for (line_index, (_, line)) in prepaint.lines.iter().enumerate() {
            line.paint(
                point(
                    bounds.left(),
                    bounds.top() + line_height * line_index as f32,
                ),
                line_height,
                window,
                cx,
            )
            .ok();
        }
        if focus_handle.is_focused(window) {
            if let Some(cursor) = prepaint.cursor.take() {
                window.paint_quad(cursor);
            }
        }
        self.input.update(cx, |input, _cx| {
            input.last_layouts = prepaint
                .lines
                .iter()
                .map(|(range, line)| (range.clone(), line.clone()))
                .collect();
            input.last_bounds = Some(bounds);
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id(self.id.clone())
            .w_full()
            .h(self.viewport_height())
            .min_h(px(36.))
            .p_2()
            .flex()
            .key_context("TextInput")
            .track_focus(&self.focus_handle())
            .cursor(CursorStyle::IBeam)
            .border_1()
            .border_color(gpui::rgb(0x332b22))
            .rounded_sm()
            .overflow_y_scroll()
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::up))
            .on_action(cx.listener(Self::down))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_up))
            .on_action(cx.listener(Self::select_down))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::newline))
            .on_action(cx.listener(Self::submit))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .text_size(px(12.))
            .line_height(self.line_height())
            .child(TextElement { input: cx.entity() })
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

fn muted_color() -> gpui::Hsla {
    gpui::hsla(0.0, 0.0, 0.55, 1.0)
}
