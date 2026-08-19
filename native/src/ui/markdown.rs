//! Pure Markdown/citation parsing helpers used by the native chat renderer.
//!
//! Rendering adapters stay stateful in `ui.rs` so they can use the Cephalon theme
//! and native click handlers, while this module keeps syntax recognition and
//! safety policy independently testable.

#[derive(Debug, Clone)]
pub(crate) enum InlineFragment {
    Text(String),
    Strong(String),
    Emphasis(String),
    Code(String),
    Link { label: String, url: String },
    Citation(String),
}

pub(crate) fn parse_inline(line: &str) -> Vec<InlineFragment> {
    let mut fragments = Vec::new();
    let mut plain = String::new();
    let mut index = 0;
    let flush_plain = |fragments: &mut Vec<InlineFragment>, plain: &mut String| {
        if !plain.is_empty() {
            fragments.push(InlineFragment::Text(std::mem::take(plain)));
        }
    };
    while index < line.len() {
        let rest = &line[index..];
        if rest.starts_with("[[src:") {
            if let Some(end) = rest.find("]]") {
                flush_plain(&mut fragments, &mut plain);
                let marker = rest[2..end].to_string();
                fragments.push(InlineFragment::Citation(
                    marker.strip_prefix("src:").unwrap_or(&marker).to_string(),
                ));
                index += end + 2;
                continue;
            }
        }
        let marker = if rest.starts_with("**") || rest.starts_with("__") {
            Some((2, true))
        } else if rest.starts_with('*') || rest.starts_with('_') {
            Some((1, false))
        } else if rest.starts_with(char::from(96)) {
            Some((1, false))
        } else {
            None
        };
        if let Some((delimiter_len, strong)) = marker {
            let delimiter = &rest[..delimiter_len];
            if let Some(relative_end) = rest[delimiter_len..].find(delimiter) {
                flush_plain(&mut fragments, &mut plain);
                let start = delimiter_len;
                let end = start + relative_end;
                let value = rest[start..end].to_string();
                if delimiter == char::from(96).to_string() {
                    fragments.push(InlineFragment::Code(value));
                } else if strong {
                    fragments.push(InlineFragment::Strong(value));
                } else {
                    fragments.push(InlineFragment::Emphasis(value));
                }
                index += end + delimiter_len;
                continue;
            }
        }
        if rest.starts_with('[') {
            if let Some(label_end) = rest.find("](") {
                if let Some(url_end) = rest[label_end + 2..].find(')') {
                    flush_plain(&mut fragments, &mut plain);
                    let label = rest[1..label_end].to_string();
                    let url_start = label_end + 2;
                    let url = rest[url_start..url_start + url_end].to_string();
                    fragments.push(InlineFragment::Link { label, url });
                    index += url_start + url_end + 1;
                    continue;
                }
            }
        }
        let character = rest
            .chars()
            .next()
            .expect("index is on a character boundary");
        plain.push(character);
        index += character.len_utf8();
    }
    flush_plain(&mut fragments, &mut plain);
    fragments
}

pub(crate) fn markdown_prefix(line: &str) -> (Option<String>, &str) {
    let trimmed = line.trim_start();
    if let Some(body) = trimmed.strip_prefix("> ") {
        return (Some("›".into()), body);
    }
    if let Some(body) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        return (Some("•".into()), body);
    }
    let digits = trimmed
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .count();
    if digits > 0 && trimmed.as_bytes().get(digits..digits + 2) == Some(b". ") {
        return (Some(trimmed[..digits + 2].into()), &trimmed[digits + 2..]);
    }
    (None, line)
}

pub(crate) fn is_table_separator(line: &str) -> bool {
    let cells = table_cells(line);
    !cells.is_empty()
        && cells.iter().all(|cell| {
            cell.chars().filter(|character| *character == '-').count() >= 3
                && cell
                    .chars()
                    .all(|character| character == '-' || character == ':' || character == ' ')
        })
}

pub(crate) fn table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
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
}

#[cfg(test)]
mod tests {
    use super::external_url_allowed;

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
}
