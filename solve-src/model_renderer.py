import random
from typing import List
from PIL import Image, ImageDraw, ImageFilter

from config import Config
from symbol_mapping import value_to_model_symbol

cfg = Config()


def _pitch_to_y(pitch: int, staff_top: int, spacing: int) -> int:
    bottom = staff_top + 4 * spacing
    return int(bottom - pitch * (spacing / 2.0))


def _draw_staff(draw: ImageDraw.ImageDraw, width: int, top: int, spacing: int):
    for i in range(5):
        y = top + i * spacing + random.randint(-1, 1)
        draw.line((24, y, width - 24, y), fill=0, width=random.choice([1, 1, 2]))


def _draw_ledger(draw: ImageDraw.ImageDraw, x: int, y: int):
    draw.line((x - 8, y, x + 8, y), fill=0, width=1)


def _draw_symbol(draw: ImageDraw.ImageDraw, x: int, value: int, staff_top: int, spacing: int):
    sym = value_to_model_symbol(value)
    y = _pitch_to_y(sym.pitch, staff_top, spacing)

    head_w = random.randint(9, 11)
    head_h = random.randint(6, 8)
    filled = sym.style in (0, 1)
    bbox = (x - head_w // 2, y - head_h // 2, x + head_w // 2, y + head_h // 2)

    if filled:
        draw.ellipse(bbox, outline=0, fill=0)
    else:
        draw.ellipse(bbox, outline=0, width=1)

    stem_up = sym.pitch <= 3
    stem_len = random.randint(18, 24)

    if stem_up:
        sx = x + head_w // 2 - 1
        draw.line((sx, y, sx, y - stem_len), fill=0, width=2)
        if sym.style >= 1:
            draw.line((sx, y - stem_len, sx + 9, y - stem_len + 4), fill=0, width=2)
        if sym.style >= 2:
            draw.line((sx, y - stem_len + 5, sx + 9, y - stem_len + 9), fill=0, width=2)
        if sym.style >= 3:
            draw.arc((sx + 2, y - stem_len + 8, sx + 14, y - stem_len + 20), 200, 360, fill=0, width=1)
    else:
        sx = x - head_w // 2 + 1
        draw.line((sx, y, sx, y + stem_len), fill=0, width=2)
        if sym.style >= 1:
            draw.line((sx, y + stem_len, sx - 9, y + stem_len - 4), fill=0, width=2)
        if sym.style >= 2:
            draw.line((sx, y + stem_len - 5, sx - 9, y + stem_len - 9), fill=0, width=2)
        if sym.style >= 3:
            draw.arc((sx - 14, y + stem_len - 20, sx - 2, y + stem_len - 8), 0, 160, fill=0, width=1)

    top_line = staff_top
    bottom_line = staff_top + 4 * spacing
    if y < top_line - spacing // 2:
        yy = top_line - spacing
        while yy >= y - 1:
            _draw_ledger(draw, x, yy)
            yy -= spacing
    elif y > bottom_line + spacing // 2:
        yy = bottom_line + spacing
        while yy <= y + 1:
            _draw_ledger(draw, x, yy)
            yy += spacing


def _draw_bar(draw: ImageDraw.ImageDraw, x: int, staff_top: int, spacing: int):
    draw.line((x, staff_top - 4, x, staff_top + 4 * spacing + 4), fill=0, width=random.choice([1, 2]))


def estimate_required_width(n_values: int, note_spacing: int = 19, margin: int = 40) -> int:
    n_bars = max(0, (n_values - 1) // 8)
    return margin + n_values * note_spacing + n_bars * max(8, note_spacing // 2) + margin


def render_model_image(values, seed=None, force_width=None):
    if seed is not None:
        random.seed(seed)

    staff_top = random.randint(42, 60)
    spacing = random.randint(9, 11)
    note_spacing = random.randint(16, 22)

    width = force_width if force_width is not None else max(
        cfg.image_width,
        estimate_required_width(len(values), note_spacing)
    )

    img = Image.new("L", (width, cfg.image_height), 255)
    draw = ImageDraw.Draw(img)

    _draw_staff(draw, width, staff_top, spacing)

    x = 40
    for i, v in enumerate(values):
        _draw_symbol(draw, x + random.randint(-1, 1), v, staff_top, spacing)
        x += note_spacing

        if (i + 1) % 8 == 0 and i + 1 < len(values):
            _draw_bar(draw, x, staff_top, spacing)
            x += max(8, note_spacing // 2)

    if random.random() < 0.55:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.6)))

    px = img.load()
    for _ in range(random.randint(50, 140)):
        xx = random.randint(0, width - 1)
        yy = random.randint(0, cfg.image_height - 1)
        px[xx, yy] = max(0, px[xx, yy] - random.randint(10, 70))

    return img
