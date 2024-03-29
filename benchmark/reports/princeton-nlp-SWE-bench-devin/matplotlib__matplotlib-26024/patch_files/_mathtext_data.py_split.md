

## EpicSplitter

1 chunks

#### Split 1
159 tokens, line: 1 - 1236

```python
"""
font data tables for truetype and afm computer modern fonts
"""

latex_to_bakoma =
 # ... other code

# Automatically generated.

type12uni =
 # ... other code

uni2type1 = {v: k for k, v in type12uni.items()}

tex2uni =
 # ... other code

# Each element is a 4-tuple of the form:
#   src_start, src_end, dst_font, dst_start
#
stix_virtual_fonts =
 # ... other code


# Fix some incorrect glyphs.
stix_glyph_fixes = {
    # Cap and Cup glyphs are swapped.
    0x22d2: 0x22d3,
    0x22d3: 0x22d2,
}
```

