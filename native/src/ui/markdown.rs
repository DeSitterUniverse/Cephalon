//! CommonMark/GFM parsing and safety helpers for the native chat renderer.

use pulldown_cmark::{Event, Options, Parser, Tag};

#[derive(Debug, Clone)]
pub(crate) enum InlineFragment {
    Text(String),
    Citation(String),
}

#[derive(Debug, Clone)]
pub(crate) enum MarkdownNode {
    Container {
        tag: Tag<'static>,
        children: Vec<MarkdownNode>,
    },
    Text(String),
    Code(String),
    Html(String),
    InlineHtml(String),
    InlineMath(String),
    DisplayMath(String),
    FootnoteReference(String),
    SoftBreak,
    HardBreak,
    Rule,
    TaskListMarker(bool),
}

pub(crate) fn parse_markdown(raw: &str) -> Vec<MarkdownNode> {
    let visible = visible_answer(raw);
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_GFM
        | Options::ENABLE_SMART_PUNCTUATION
        | Options::ENABLE_HEADING_ATTRIBUTES
        | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS
        | Options::ENABLE_PLUSES_DELIMITED_METADATA_BLOCKS
        | Options::ENABLE_MATH
        | Options::ENABLE_DEFINITION_LIST
        | Options::ENABLE_SUPERSCRIPT
        | Options::ENABLE_SUBSCRIPT;
    let parser = Parser::new_ext(&visible, options);
    let mut roots = Vec::new();
    let mut stack: Vec<(Tag<'static>, Vec<MarkdownNode>)> = Vec::new();

    for event in parser {
        match event.into_static() {
            Event::Start(tag) => stack.push((tag, Vec::new())),
            Event::End(_) => {
                if let Some((tag, children)) = stack.pop() {
                    push_node(
                        &mut stack,
                        &mut roots,
                        MarkdownNode::Container { tag, children },
                    );
                }
            }
            Event::Text(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::Text(value.into_string()),
            ),
            Event::Code(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::Code(value.into_string()),
            ),
            Event::Html(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::Html(value.into_string()),
            ),
            Event::InlineHtml(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::InlineHtml(value.into_string()),
            ),
            Event::InlineMath(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::InlineMath(value.into_string()),
            ),
            Event::DisplayMath(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::DisplayMath(value.into_string()),
            ),
            Event::FootnoteReference(value) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::FootnoteReference(value.into_string()),
            ),
            Event::SoftBreak => push_node(&mut stack, &mut roots, MarkdownNode::SoftBreak),
            Event::HardBreak => push_node(&mut stack, &mut roots, MarkdownNode::HardBreak),
            Event::Rule => push_node(&mut stack, &mut roots, MarkdownNode::Rule),
            Event::TaskListMarker(checked) => push_node(
                &mut stack,
                &mut roots,
                MarkdownNode::TaskListMarker(checked),
            ),
        }
    }

    while let Some((tag, children)) = stack.pop() {
        push_node(
            &mut stack,
            &mut roots,
            MarkdownNode::Container { tag, children },
        );
    }
    roots
}

fn push_node(
    stack: &mut [(Tag<'static>, Vec<MarkdownNode>)],
    roots: &mut Vec<MarkdownNode>,
    node: MarkdownNode,
) {
    if let Some((_, children)) = stack.last_mut() {
        if let MarkdownNode::Text(value) = node {
            if let Some(MarkdownNode::Text(previous)) = children.last_mut() {
                previous.push_str(&value);
            } else {
                children.push(MarkdownNode::Text(value));
            }
        } else {
            children.push(node);
        }
    } else {
        roots.push(node);
    }
}

pub(crate) fn text_content(nodes: &[MarkdownNode]) -> String {
    let mut output = String::new();
    for node in nodes {
        match node {
            MarkdownNode::Text(value)
            | MarkdownNode::Code(value)
            | MarkdownNode::Html(value)
            | MarkdownNode::InlineHtml(value)
            | MarkdownNode::InlineMath(value)
            | MarkdownNode::DisplayMath(value)
            | MarkdownNode::FootnoteReference(value) => output.push_str(value),
            MarkdownNode::Container { children, .. } => output.push_str(&text_content(children)),
            MarkdownNode::SoftBreak | MarkdownNode::HardBreak => output.push('\n'),
            MarkdownNode::Rule | MarkdownNode::TaskListMarker(_) => {}
        }
    }
    output
}

pub(crate) fn split_citations(text: &str) -> Vec<InlineFragment> {
    let mut fragments = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = text[cursor..].find("[[src:") {
        let start = cursor + relative_start;
        if start > cursor {
            fragments.push(InlineFragment::Text(text[cursor..start].to_string()));
        }
        let marker_start = start + 2;
        let Some(relative_end) = text[marker_start..].find("]]") else {
            fragments.push(InlineFragment::Text(text[start..].to_string()));
            cursor = text.len();
            break;
        };
        let end = marker_start + relative_end;
        let marker = text[marker_start..end].to_string();
        fragments.push(InlineFragment::Citation(
            marker.strip_prefix("src:").unwrap_or(&marker).to_string(),
        ));
        cursor = end + 2;
    }
    if cursor < text.len() {
        fragments.push(InlineFragment::Text(text[cursor..].to_string()));
    }
    if fragments.is_empty() && !text.is_empty() {
        fragments.push(InlineFragment::Text(text.to_string()));
    }
    fragments
}

pub(crate) fn visible_answer(raw: &str) -> String {
    let mut visible = String::with_capacity(raw.len());
    let mut cursor = 0;
    while let Some(relative_start) = raw[cursor..].find("<think>") {
        let start = cursor + relative_start;
        visible.push_str(&raw[cursor..start]);
        let content_start = start + "<think>".len();
        let Some(relative_end) = raw[content_start..].find("</think>") else {
            return visible;
        };
        cursor = content_start + relative_end + "</think>".len();
    }
    visible.push_str(&raw[cursor..]);
    visible
}

/// Only ordinary web links are eligible for OS/browser dispatch. Cephalon
/// citations are parsed into a separate fragment and never pass through this
/// policy.
pub(crate) fn external_url_allowed(url: &str) -> bool {
    let trimmed = url.trim();
    let Some((scheme, remainder)) = trimmed.split_once(':') else {
        return false;
    };
    let scheme = scheme.to_ascii_lowercase();
    let authority = remainder.strip_prefix("//").unwrap_or_default();
    matches!(scheme.as_str(), "http" | "https")
        && !authority.trim().is_empty()
        && !authority.chars().any(char::is_whitespace)
        && !authority.chars().any(char::is_control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulldown_cmark::{Alignment, Tag};

    fn contains_tag(nodes: &[MarkdownNode], predicate: &impl Fn(&Tag<'static>) -> bool) -> bool {
        nodes.iter().any(|node| match node {
            MarkdownNode::Container { tag, children } => {
                predicate(tag) || contains_tag(children, predicate)
            }
            _ => false,
        })
    }

    fn contains_task_marker(nodes: &[MarkdownNode], checked: bool) -> bool {
        nodes.iter().any(|node| match node {
            MarkdownNode::TaskListMarker(value) => *value == checked,
            MarkdownNode::Container { children, .. } => contains_task_marker(children, checked),
            _ => false,
        })
    }

    #[test]
    fn parses_commonmark_and_gfm_blocks() {
        let nodes = parse_markdown(
            "# Heading\n\n- [x] done\n- [ ] todo\n\n| A | B |\n| :- | -: |\n| 1 | 2 |\n\n~~old~~ and [^1]\n\n[^1]: note",
        );
        assert!(contains_tag(&nodes, &|tag| matches!(
            tag,
            Tag::Heading { .. }
        )));
        assert!(contains_tag(&nodes, &|tag| matches!(tag, Tag::List(_))));
        assert!(contains_tag(&nodes, &|tag| matches!(tag, Tag::Table(_))));
        assert!(contains_tag(&nodes, &|tag| matches!(
            tag,
            Tag::Strikethrough
        )));
        assert!(contains_tag(&nodes, &|tag| matches!(
            tag,
            Tag::FootnoteDefinition(_)
        )));
        assert!(contains_task_marker(&nodes, true));
        assert!(contains_task_marker(&nodes, false));
    }

    #[test]
    fn preserves_nested_formatting_and_citations() {
        let nodes = parse_markdown("**bold _nested_ [[src:S1]]**");
        let rendered_text = text_content(&nodes);
        assert!(rendered_text.contains("nested"));
        let original = "before [[src:S1]]  after [[src:S2]]";
        let fragments = split_citations(original);
        assert_eq!(
            fragments
                .iter()
                .filter(|fragment| matches!(fragment, InlineFragment::Citation(_)))
                .count(),
            2
        );
        let reconstructed = fragments
            .iter()
            .map(|fragment| match fragment {
                InlineFragment::Text(value) => value.clone(),
                InlineFragment::Citation(value) => format!("[[src:{value}]]"),
            })
            .collect::<String>();
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn external_urls_require_http_or_https_authority() {
        assert!(external_url_allowed("https://example.com/docs?q=1"));
        assert!(external_url_allowed("HTTP://example.com"));
        assert!(!external_url_allowed("file:///C:/secret.txt"));
        assert!(!external_url_allowed("javascript:alert(1)"));
        assert!(!external_url_allowed("data:text/html,hello"));
        assert!(!external_url_allowed("https://"));
        assert!(!external_url_allowed("https://example.com/a path"));
    }

    #[test]
    fn hidden_thinking_is_removed_before_markdown_parsing() {
        let nodes = parse_markdown("visible <think>secret **not shown**</think> answer");
        assert_eq!(text_content(&nodes), "visible  answer");
    }

    #[test]
    fn long_prose_preserves_native_text_layout_input() {
        let paragraph = "The quick brown fox crosses a deliberately long paragraph so that the native layout engine must wrap at a word boundary without any renderer-owned chunks or lost whitespace.";
        let nodes = parse_markdown(paragraph);

        assert_eq!(text_content(&nodes), paragraph);
        assert!(!text_content(&nodes).contains('\u{200b}'));
    }

    #[test]
    fn code_blocks_preserve_long_lines_and_whitespace() {
        let code = format!(
            "```json\n{{\"url\":\"https://example.com/{}\",\"items\":[1,2,3]}}\n```",
            "x".repeat(220)
        );
        let nodes = parse_markdown(&code);
        let rendered = text_content(&nodes);

        assert!(rendered.contains("https://example.com/"));
        assert!(rendered.contains(&"x".repeat(220)));
        assert!(!rendered.contains('\u{200b}'));
    }

    #[test]
    fn tables_keep_one_multi_cell_header_and_alignment() {
        let nodes = parse_markdown(
            "| Name | Score | Page |\n|:-----|------:|-----:|\n| **Alpha** | 0.91 | [[src:S1]] 4 |\n| Beta | 0.82 | 8 |",
        );
        let MarkdownNode::Container {
            tag: Tag::Table(alignments),
            children,
        } = nodes.first().expect("table root")
        else {
            panic!("expected a table root");
        };
        assert_eq!(
            alignments,
            &vec![Alignment::Left, Alignment::Right, Alignment::Right]
        );

        let MarkdownNode::Container {
            tag: Tag::TableHead,
            children: head_rows,
        } = children.first().expect("table head")
        else {
            panic!("expected a table head");
        };
        assert_eq!(head_rows.len(), 3);
        assert!(head_rows.iter().all(|node| matches!(
            node,
            MarkdownNode::Container {
                tag: Tag::TableCell,
                ..
            }
        )));

        assert!(text_content(&nodes).contains("Alpha"));
        assert!(split_citations("[[src:S1]]")
            .iter()
            .any(|fragment| matches!(fragment, InlineFragment::Citation(value) if value == "S1")));
    }
}
