"""UTF-8 Encoding from First Principles — encode characters to bytes manually.

Code Block 2: Demonstrates UTF-8's variable-length encoding scheme.
"""

def utf8_encode_char(char):
    """Encode a single character to UTF-8 bytes from first principles."""
    cp = ord(char)  # Unicode code point
    if cp <= 0x7F:          # 1-byte: 0xxxxxxx
        return [cp]
    elif cp <= 0x7FF:       # 2-byte: 110xxxxx 10xxxxxx
        return [0xC0 | (cp >> 6),
                0x80 | (cp & 0x3F)]
    elif cp <= 0xFFFF:      # 3-byte: 1110xxxx 10xxxxxx 10xxxxxx
        return [0xE0 | (cp >> 12),
                0x80 | ((cp >> 6) & 0x3F),
                0x80 | (cp & 0x3F)]
    else:                    # 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        return [0xF0 | (cp >> 18),
                0x80 | ((cp >> 12) & 0x3F),
                0x80 | ((cp >> 6) & 0x3F),
                0x80 | (cp & 0x3F)]

examples = [('A', 'ASCII'), ('\u00e9', 'Latin'), ('\u4e2d', 'CJK'), ('\U0001f600', 'Emoji')]
for char, script in examples:
    encoded = utf8_encode_char(char)
    hex_str = ' '.join(f'0x{b:02X}' for b in encoded)
    binary = ' '.join(f'{b:08b}' for b in encoded)
    print(f"'{char}' ({script:<6}) U+{ord(char):04X} -> [{hex_str}]")
    print(f"  binary: {binary}")

# Blog claims:
# 'A'  (ASCII ) U+0041 -> [0x41]
# 'é'  (Latin ) U+00E9 -> [0xC3 0xA9]
# '中' (CJK   ) U+4E2D -> [0xE4 0xB8 0xAD]
# '😀' (Emoji ) U+1F600 -> [0xF0 0x9F 0x98 0x80]
