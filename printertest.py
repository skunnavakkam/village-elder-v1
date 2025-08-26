# markdown_printer_fixed.py
import re, textwrap, time

ESC = b"\x1b"
GS  = b"\x1d"

def _wrap(text, width):
    out = []
    for para in text.splitlines():
        if not para.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(para, width=width, replace_whitespace=False))
    return out

def _encode(s: str) -> bytes:
    return s.encode("ascii", "replace")

def _apply_inline(markup: str) -> bytes:
    s = markup
    # Bold: **...**
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: "\x1bE\x01" + m.group(1) + "\x1bE\x00", s)
    # Italic: *...* (render as underline)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: "\x1b-\x01" + m.group(1) + "\x1b-\x00", s)
    # Inline code: `...` (keep the backticks visible, no style applied)
    # So "`foo`" → "`foo`" literally
    return _encode(s)

def print_markdown(md: str, device: str="/dev/usb/lp0"):
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    
    with open(device, "wb") as f:
        f.write(ESC + b"@")                # init
        f.write(ESC + b"a" + b"\x00")      # align left
        f.write(GS + b"!" + b"\x00")       # normal size

        for raw in md.splitlines():
            line = raw.rstrip()

            # Headings
            if line.startswith("# "):
                text = line[2:].strip()
                f.write(ESC + b"a" + b"\x01")   # center
                f.write(GS + b"!" + b"\x11")    # double width+height
                f.write(_encode(text.upper()) + b"\n")
                f.write(GS + b"!" + b"\x00" + ESC + b"a" + b"\x00" + b"\n")
                continue
            elif line.startswith("## "):
                text = line[3:].strip()
                f.write(GS + b"!" + b"\x01" + ESC + b"E" + b"\x01")
                f.write(_encode(text) + b"\n")
                f.write(GS + b"!" + b"\x00" + ESC + b"E" + b"\x00" + b"\n")
                continue
            elif line.startswith("### "):
                text = line[4:].strip()
                f.write(ESC + b"-" + b"\x01")
                f.write(_encode(text) + b"\n")
                f.write(ESC + b"-" + b"\x00" + b"\n")
                continue

            # Bulleted list (minus + space)
            m = re.match(r"^\s*-\s+(.*)", line)
            if m:
                text = m.group(1)
                wrapped = _wrap(text, width=30)
                for i, w in enumerate(wrapped):
                    prefix = "- " if i == 0 else "  "
                    f.write(_apply_inline(prefix + w) + b"\n")
                continue

            # Numbered list
            m = re.match(r"^\s*(\d+)\.\s+(.*)", line)
            if m:
                num, text = m.groups()
                prefix = f"{num}. "
                wrapped = _wrap(text, width=30 - len(num))
                for i, w in enumerate(wrapped):
                    f.write(_apply_inline((prefix if i == 0 else " " * len(prefix)) + w) + b"\n")
                continue

            # Blank line
            if not line.strip():
                f.write(b"\n")
                continue

            # Normal paragraph
            for w in _wrap(line, width=32):
                f.write(_apply_inline(w) + b"\n")

        f.write(b"\n\n\n")  # feed

if __name__ == "__main__":
    demo = """# Receipt Demo

Hello **world** - this is *italic*, and here is `inline code`.

## List
- apples
- bananas
- **bold** inside list
"""
    print_markdown(demo)