from PIL import Image, ImageDraw, ImageFont, ImageChops

class TextBufferEPD:
    def __init__(self,
                 epd,
                 font_path=None,
                 font_size=32,
                 margin=8,
                 line_spacing=1.2,
                 align="left",
                 tail=True,
                 rotation=90,                 # keep 0 if you want text parallel to width; set 90/270 if needed
                 fast_mode=True,             # enables partial update path
                 full_refresh_every=30):     # do a full refresh every N partials to fight ghosting
        self.epd = epd
        self.font = (ImageFont.truetype(font_path, font_size)
                     if font_path else ImageFont.load_default())
        self.margin = int(margin)
        self.line_spacing = float(line_spacing)
        self.align = align
        self.tail = bool(tail)
        self.rotation = int(rotation) % 360
        self.fast_mode = bool(fast_mode)
        self.full_refresh_every = int(full_refresh_every)
        self._buffer = []
        self._last_img = None
        self._partials_since_full = 0

        # Introspect driver capabilities
        self._cap = {
            "FULL_UPDATE": getattr(self.epd, "FULL_UPDATE", None),
            "PART_UPDATE": getattr(self.epd, "PART_UPDATE", None),
            "displayPartial": hasattr(self.epd, "displayPartial"),
            "displayPartialWindow": hasattr(self.epd, "displayPartialWindow"),
            "displayPartBaseImage": hasattr(self.epd, "displayPartBaseImage"),
        }

        # Initialize panel with a clean full refresh first
        self._enter_full_mode()
        self.epd.Clear(0xFF)
        # If we can, move into partial mode for subsequent fast writes
        if self.fast_mode:
            self._enter_partial_mode()

    # ---------- Public API ----------
    def clear(self, hardware: bool = False):
        """Clear in-memory buffer; optionally clear panel to white and reset partial cycle."""
        self._buffer.clear()
        self._last_img = None
        if hardware:
            self._enter_full_mode()
            self.epd.Clear(0xFF)
            if self.fast_mode:
                self._enter_partial_mode()
        else:
            # Quick visual clear using partial if possible
            blank = self._render_frame()
            if self.rotation:
                blank = blank.rotate(self.rotation, expand=True)
            self._fast_push(blank)

    def set_text(self, text: str, push: bool = True):
        self.clear()
        self._buffer = [text]
        if push:
            self.push()

    def write(self, text: str, push: bool = True):
        self._buffer.append(text)
        if push:
            self.push()

    def writeline(self, text: str = "", push: bool = True):
        self._buffer.append(text + "\n")
        if push:
            self.push()

    def set_rotation(self, degrees: int):
        self.rotation = int(degrees) % 360

    def push(self):
        """Render and display. Uses partial update when available to avoid full-screen blink."""
        img = self._render_frame()
        if self.rotation:
            img = img.rotate(self.rotation, expand=True)

        if not self.fast_mode:
            # Slow, full flash path
            self._enter_full_mode()
            self.epd.display(self.epd.getbuffer(img))
            if self.fast_mode:
                self._enter_partial_mode()
            self._last_img = img
            self._partials_since_full = 0
            return

        # Fast path: partial update
        if (self.full_refresh_every > 0 and
            self._partials_since_full >= self.full_refresh_every):
            self._enter_full_mode()
            self.epd.display(self.epd.getbuffer(img))
            self._last_img = img
            self._partials_since_full = 0
            if self.fast_mode:
                self._enter_partial_mode()
            return

        self._fast_push(img)
        self._last_img = img
        self._partials_since_full += 1

    # ---------- Internals ----------
    def _enter_full_mode(self):
        if self._cap["FULL_UPDATE"] is not None:
            self.epd.init(self._cap["FULL_UPDATE"])
        else:
            # Some drivers just reuse init() for full
            self.epd.init()

    def _enter_partial_mode(self):
        # Many drivers require: full base image -> switch to PART -> partials
        if self._cap["PART_UPDATE"] is not None:
            self.epd.init(self._cap["PART_UPDATE"])
        # If a base image call exists, prime it (prevents first-partial ghosting)
        if self._cap["displayPartBaseImage"] and self._last_img is not None:
            self.epd.displayPartBaseImage(self.epd.getbuffer(self._last_img))

    def _render_frame(self):
        # Layout canvas dimensions so wrapping is correct before any rotation
        if self.rotation in (0, 180):
            W, H = int(self.epd.width), int(self.epd.height)
        else:
            W, H = int(self.epd.height), int(self.epd.width)

        img = Image.new("1", (W, H), 255)
        draw = ImageDraw.Draw(img)

        usable_w = max(W - 2 * self.margin, 1)
        usable_h = max(H - 2 * self.margin, 1)

        lines = self._wrap_lines(draw, usable_w, usable_h)
        y = self.margin
        line_h = max(int(self._line_height(draw) * self.line_spacing), 1)

        for line in lines:
            x = self._x_for_line(draw, line, W)
            draw.text((x, y), line, font=self.font, fill=0)
            y += line_h
        return img

    def _fast_push(self, img):
        """
        Try to do a no-blink partial update.
        Strategy:
          1) If driver has displayPartialWindow and we have a diff bbox, update just that window.
          2) Else if driver has displayPartial, send whole frame (still no blink).
          3) Else fall back to full display.
        """
        bbox = None
        if self._last_img is not None and self._last_img.size == img.size:
            diff = ImageChops.difference(self._last_img, img)
            bbox = diff.getbbox()  # (l, t, r, b) or None if no change

        if bbox and self._cap["displayPartialWindow"]:
            l, t, r, b = bbox
            # Many controllers want x aligned to 8px and width multiples of 8
            l_aligned = (l // 8) * 8
            r_aligned = ((r + 7) // 8) * 8
            r_aligned = min(r_aligned, img.width)
            w = max(r_aligned - l_aligned, 8)
            h = max(b - t, 1)

            region = img.crop((l_aligned, t, l_aligned + w, t + h))
            self.epd.displayPartialWindow(self.epd.getbuffer(region), l_aligned, t, w, h)
            return

        if self._cap["displayPartial"]:
            self.epd.displayPartial(self.epd.getbuffer(img))
            return

        # Fallback: full (will blink)
        self._enter_full_mode()
        self.epd.display(self.epd.getbuffer(img))
        if self.fast_mode:
            self._enter_partial_mode()

    # --- text layout helpers ---
    def _wrap_lines(self, draw, max_w, max_h):
        text = "".join(self._buffer)
        paragraphs = text.splitlines()

        lines = []
        for p in paragraphs:
            if p == "":
                lines.append("")
            else:
                lines.extend(self._wrap_paragraph(draw, p, max_w))

        line_h = max(int(self._line_height(draw) * self.line_spacing), 1)
        if line_h <= 0:
            return []
        max_lines = max_h // line_h
        if max_lines <= 0:
            return []
        return lines[-max_lines:] if self.tail and len(lines) > max_lines else lines[:max_lines]

    def _wrap_paragraph(self, draw, text, max_w):
        words = text.split()
        if not words:
            return [""]

        out, cur = [], ""
        for w in words:
            cand = f"{cur} {w}".strip() if cur else w
            if self._text_width(draw, cand) <= max_w:
                cur = cand
            else:
                if cur:
                    out.append(cur)
                if self._text_width(draw, w) > max_w:
                    out.extend(self._hard_wrap_word(draw, w, max_w))
                    cur = ""
                else:
                    cur = w
        if cur:
            out.append(cur)
        return out

    def _hard_wrap_word(self, draw, word, max_w):
        chunks, chunk = [], ""
        for ch in word:
            cand = chunk + ch
            if self._text_width(draw, cand) <= max_w:
                chunk = cand
            else:
                if chunk:
                    chunks.append(chunk)
                chunk = ch
        if chunk:
            chunks.append(chunk)
        return chunks

    def _line_height(self, draw):
        bbox = draw.textbbox((0, 0), "Ag", font=self.font)
        return bbox[3] - bbox[1]

    def _text_width(self, draw, s):
        try:
            return int(draw.textlength(s, font=self.font))
        except Exception:
            bbox = draw.textbbox((0, 0), s, font=self.font)
            return int(bbox[2] - bbox[0])

    def _x_for_line(self, draw, s, canvas_w):
        if self.align == "left":
            return self.margin
        w = self._text_width(draw, s)
        if self.align == "center":
            return max((canvas_w - w) // 2, self.margin)
        if self.align == "right":
            return max(canvas_w - self.margin - w, self.margin)
        return self.margin