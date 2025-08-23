"""
Markdown and Display Utilities

This module provides functions for formatting text, markdown, HTML,
and other display-related operations for UI components.
"""

import json
import html
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def format_markdown(text: str, escape_html: bool = True) -> str:
    """
    Format text with basic markdown support.
    
    Args:
        text: Input text to format
        escape_html: Whether to escape HTML characters
        
    Returns:
        Formatted markdown string
    """
    if escape_html:
        text = html.escape(text)
    
    # Bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    
    # Italic text
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Code spans
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    
    # Line breaks
    text = text.replace('\n', '<br>')
    
    return text


def create_code_block(code: str, language: str = "", title: Optional[str] = None) -> str:
    """
    Create a formatted code block with syntax highlighting support.
    
    Args:
        code: Code content
        language: Programming language for syntax highlighting
        title: Optional title for the code block
        
    Returns:
        HTML formatted code block
    """
    escaped_code = html.escape(code)
    
    title_html = f'<div class="code-title">{html.escape(title)}</div>' if title else ""
    
    return f"""
<div class="code-block">
    {title_html}
    <pre><code class="language-{language}">{escaped_code}</code></pre>
</div>
    """.strip()


def format_json(data: Any, indent: int = 2, sort_keys: bool = True) -> str:
    """
    Format data as pretty-printed JSON.
    
    Args:
        data: Data to format as JSON
        indent: Number of spaces for indentation
        sort_keys: Whether to sort dictionary keys
        
    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        return f"Error formatting JSON: {str(e)}"


def create_collapsible_section(title: str, content: str, collapsed: bool = True) -> str:
    """
    Create a collapsible section with title and content.
    
    Args:
        title: Section title
        content: Section content
        collapsed: Whether section starts collapsed
        
    Returns:
        HTML for collapsible section
    """
    state = "closed" if collapsed else "open"
    
    return f"""
<details {state}>
    <summary>{html.escape(title)}</summary>
    <div class="collapsible-content">
        {content}
    </div>
</details>
    """.strip()


def format_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> str:
    """
    Format list of dictionaries as HTML table.
    
    Args:
        data: List of dictionaries to display as table
        headers: Optional list of column headers
        
    Returns:
        HTML table string
    """
    if not data:
        return "<p>No data to display</p>"
    
    if headers is None:
        headers = list(data[0].keys()) if data else []
    
    table_html = ["<table>"]
    
    # Header row
    if headers:
        table_html.append("<thead><tr>")
        for header in headers:
            table_html.append(f"<th>{html.escape(str(header))}</th>")
        table_html.append("</tr></thead>")
    
    # Data rows
    table_html.append("<tbody>")
    for row in data:
        table_html.append("<tr>")
        for header in headers:
            value = row.get(header, "")
            table_html.append(f"<td>{html.escape(str(value))}</td>")
        table_html.append("</tr>")
    table_html.append("</tbody></table>")
    
    return "\n".join(table_html)


def sanitize_html(text: str, allowed_tags: Optional[List[str]] = None) -> str:
    """
    Sanitize HTML by removing or escaping potentially dangerous tags.
    
    Args:
        text: HTML text to sanitize
        allowed_tags: List of allowed HTML tags
        
    Returns:
        Sanitized HTML string
    """
    if allowed_tags is None:
        allowed_tags = ['p', 'br', 'strong', 'em', 'code', 'pre', 'a', 'ul', 'ol', 'li']
    
    # Simple regex-based sanitization (for production, use a proper library like bleach)
    # Remove script tags completely
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove dangerous attributes
    text = re.sub(r'\s(?:on\w+|javascript:|vbscript:|data:)=["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    
    # Keep only allowed tags
    if allowed_tags:
        allowed_pattern = '|'.join(allowed_tags)
        text = re.sub(r'<(?!/?)(?!' + allowed_pattern + r'\b)[^>]*>', '', text, flags=re.IGNORECASE)
    
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of text
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rstrip() + suffix


def format_timestamp(timestamp: Union[datetime, str, float], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp as readable string.
    
    Args:
        timestamp: Timestamp to format (datetime, ISO string, or Unix timestamp)
        format_str: Format string for output
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            # Try parsing ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, (int, float)):
            # Unix timestamp
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            return str(timestamp)
        
        return dt.strftime(format_str)
    except (ValueError, TypeError):
        return str(timestamp)


def create_badge(text: str, badge_type: str = "info", size: str = "small") -> str:
    """
    Create a styled badge/tag element.
    
    Args:
        text: Badge text
        badge_type: Badge type (info, success, warning, error)
        size: Badge size (small, medium, large)
        
    Returns:
        HTML badge element
    """
    return f'<span class="badge badge-{badge_type} badge-{size}">{html.escape(text)}</span>'


def format_error_message(error: Union[Exception, str], include_traceback: bool = False) -> str:
    """
    Format error message for display.
    
    Args:
        error: Error to format
        include_traceback: Whether to include traceback info
        
    Returns:
        Formatted error message
    """
    if isinstance(error, Exception):
        error_type = type(error).__name__
        error_msg = str(error)
        
        if include_traceback:
            import traceback
            tb = traceback.format_exc()
            return f"""
<div class="error-message">
    <strong>{error_type}:</strong> {html.escape(error_msg)}
    <details>
        <summary>Traceback</summary>
        <pre>{html.escape(tb)}</pre>
    </details>
</div>
            """.strip()
        else:
            return f'<div class="error-message"><strong>{error_type}:</strong> {html.escape(error_msg)}</div>'
    else:
        return f'<div class="error-message">{html.escape(str(error))}</div>'


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        current: Current progress value
        total: Total/maximum value
        width: Width of progress bar in characters
        
    Returns:
        Progress bar string
    """
    if total <= 0:
        return "[" + " " * width + "] 0%"
    
    percentage = min(100, max(0, (current / total) * 100))
    filled = int((percentage / 100) * width)
    bar = "=" * filled + " " * (width - filled)
    
    return f"[{bar}] {percentage:.1f}%"