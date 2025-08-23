"""
Text utilities - Common text processing and manipulation functions
"""

import re
import string
import unicodedata
from typing import List, Optional, Dict, Set, Union
from collections import Counter
import html


def clean_text(text: str, 
               remove_extra_spaces: bool = True,
               remove_newlines: bool = False,
               remove_punctuation: bool = False,
               lowercase: bool = False) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_extra_spaces: Remove multiple consecutive spaces
        remove_newlines: Remove all newline characters
        remove_punctuation: Remove all punctuation
        lowercase: Convert to lowercase
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Remove HTML entities
    text = html.unescape(text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove newlines if requested
    if remove_newlines:
        text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove punctuation if requested
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of output
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    if len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return text[:max_length]
    
    return text[:max_length - len(suffix)] + suffix

def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize a string for use as a filename.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid characters with
    
    Returns:
        Sanitized filename
    """
    if not isinstance(filename, str):
        filename = str(filename) if filename is not None else "unnamed"
    
    # Remove or replace invalid filename characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    filename = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    # Truncate if too long (255 is common filesystem limit)
    return truncate_text(filename, 255, "")

def extract_keywords(text: str, 
                    min_length: int = 3,
                    max_keywords: Optional[int] = None,
                    exclude_common: bool = True) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords to return
        exclude_common: Exclude common words
    
    Returns:
        List of keywords
    """
    if not isinstance(text, str):
        return []
    
    # Common words to exclude
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Clean and tokenize text
    clean_text_str = clean_text(text, lowercase=True, remove_punctuation=True)
    words = clean_text_str.split()
    
    # Filter words
    keywords = []
    for word in words:
        if (len(word) >= min_length and 
            (not exclude_common or word.lower() not in common_words) and
            word.isalpha()):
            keywords.append(word.lower())
    
    # Count occurrences and sort by frequency
    word_counts = Counter(keywords)
    sorted_keywords = [word for word, _ in word_counts.most_common()]
    
    # Limit number of keywords if specified
    if max_keywords:
        sorted_keywords = sorted_keywords[:max_keywords]
    
    return sorted_keywords

def count_words(text: str, exclude_empty: bool = True) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        exclude_empty: Whether to exclude empty strings from count
    
    Returns:
        Number of words
    """
    if not isinstance(text, str):
        return 0
    
    words = text.split()
    
    if exclude_empty:
        words = [word for word in words if word.strip()]
    
    return len(words)

def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Count characters in text.
    
    Args:
        text: Input text
        include_spaces: Whether to include spaces in count
    
    Returns:
        Number of characters
    """
    if not isinstance(text, str):
        return 0
    
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(' ', ''))

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Input text
    
    Returns:
        List of email addresses found
    """
    if not isinstance(text, str):
        return []
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
    
    Returns:
        List of URLs found
    """
    if not isinstance(text, str):
        return []
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text (US format).
    
    Args:
        text: Input text
    
    Returns:
        List of phone numbers found
    """
    if not isinstance(text, str):
        return []
    
    # Pattern for US phone numbers in various formats
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b',  # 1234567890
    ]
    
    phone_numbers = []
    for pattern in phone_patterns:
        phone_numbers.extend(re.findall(pattern, text))
    
    return list(set(phone_numbers))  # Remove duplicates

def replace_multiple(text: str, replacements: Dict[str, str]) -> str:
    """
    Replace multiple substrings in text.
    
    Args:
        text: Input text
        replacements: Dictionary of {old: new} replacements
    
    Returns:
        Text with replacements made
    """
    if not isinstance(text, str) or not replacements:
        return text if isinstance(text, str) else str(text)
    
    # Sort replacements by length (longest first) to avoid partial replacements
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    
    for old, new in sorted_replacements:
        text = text.replace(old, new)
    
    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace characters to single spaces.
    
    Args:
        text: Input text
    
    Returns:
        Text with normalized whitespace
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Replace all whitespace sequences with single space
    return re.sub(r'\s+', ' ', text).strip()

def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Input text with HTML tags
    
    Returns:
        Text with HTML tags removed
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def camel_to_snake(text: str) -> str:
    """
    Convert camelCase to snake_case.
    
    Args:
        text: Input text in camelCase
    
    Returns:
        Text in snake_case
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    # Insert underscore before uppercase letters and convert to lowercase
    snake_case = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_case).lower()

def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """
    Convert snake_case to camelCase.
    
    Args:
        text: Input text in snake_case
        capitalize_first: Whether to capitalize the first letter
    
    Returns:
        Text in camelCase
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    components = text.split('_')
    if not components:
        return text
    
    if capitalize_first:
        return ''.join(word.capitalize() for word in components)
    else:
        return components[0] + ''.join(word.capitalize() for word in components[1:])

def title_case(text: str, exceptions: Optional[Set[str]] = None) -> str:
    """
    Convert text to title case with optional exceptions.
    
    Args:
        text: Input text
        exceptions: Set of words to keep lowercase (articles, prepositions, etc.)
    
    Returns:
        Text in title case
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    if exceptions is None:
        exceptions = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 
                     'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet'}
    
    words = text.split()
    result = []
    
    for i, word in enumerate(words):
        if i == 0 or i == len(words) - 1:
            # Always capitalize first and last words
            result.append(word.capitalize())
        elif word.lower() in exceptions:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    
    return ' '.join(result)

def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two strings are similar based on character overlap.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        threshold: Similarity threshold (0.0 to 1.0)
    
    Returns:
        True if texts are similar enough
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        return False
    
    # Simple character-based similarity
    text1_clean = clean_text(text1, lowercase=True, remove_punctuation=True)
    text2_clean = clean_text(text2, lowercase=True, remove_punctuation=True)
    
    if not text1_clean or not text2_clean:
        return text1_clean == text2_clean
    
    # Calculate Jaccard similarity using character bigrams
    def get_bigrams(text: str) -> Set[str]:
        return set(text[i:i+2] for i in range(len(text)-1))
    
    bigrams1 = get_bigrams(text1_clean)
    bigrams2 = get_bigrams(text2_clean)
    
    if not bigrams1 and not bigrams2:
        return True
    
    intersection = len(bigrams1.intersection(bigrams2))
    union = len(bigrams1.union(bigrams2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity >= threshold

def wrap_text(text: str, width: int = 80, break_long_words: bool = True) -> List[str]:
    """
    Wrap text to specified width.
    
    Args:
        text: Input text to wrap
        width: Maximum line width
        break_long_words: Whether to break words longer than width
    
    Returns:
        List of wrapped lines
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    if width <= 0:
        return [text]
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        
        # If adding this word would exceed width, start new line
        if current_length + word_length + len(current_line) > width and current_line:
            lines.append(' '.join(current_line))
            current_line = []
            current_length = 0
        
        # If word is longer than width and break_long_words is True
        if word_length > width and break_long_words:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []
                current_length = 0
            
            # Break the long word
            while len(word) > width:
                lines.append(word[:width])
                word = word[width:]
            
            if word:
                current_line = [word]
                current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length
    
    # Add remaining words
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else ['']