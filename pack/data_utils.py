# -*- coding: utf-8 -*-

import hashlib


def filter_bad_ord(x):
    """
    Filter out invalid characters from input string
    Args:
        x: Input string to filter
    Returns:
        Filtered string containing only valid single characters within ASCII range 32-65532
    """
    tmp = ""
    if x is None:
        return x
    for i in x:
        # Skip multi-character sequences
        if len(i) > 1:
            continue
        # Only keep characters with ordinal values between 32 (space) and 65532
        if 32 <= ord(i) < 65533:
            tmp += i
    return tmp


def md5_encrypt(text):
    """
    Generate MD5 hash of input text
    Args:
        text: Input string/value to hash
    Returns:
        32-character hexadecimal MD5 hash string
    """
    # Convert input to string if it isn't already
    if not isinstance(text, str):
        text = str(text)
    # Encode string to UTF-8, generate MD5 hash and return hex representation
    return hashlib.md5(text.encode('utf-8')).hexdigest()