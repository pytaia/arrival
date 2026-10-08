import random
from typing import List
from PIL import Image, ImageDraw, ImageFilter

from config import Config

cfg = Config()


def _value_to_carrier_symbol(v: int):
    row = v // 4
    kind = v % 4
    return row, kind


def _draw_marker(draw: ImageDraw.ImageDraw, cx: int, cy: int, kind: int):
    if kind == 0:
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=0, fill=0)
    elif kind == 1:
        draw.rectangle((cx - 4, cy - 4, cx + 4, cy + 4), outline=0, fill=0)
    elif kind == 2:
        draw.polygon([(cx, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)], outline=0, fill=0)
    elif kind == 3:
        draw.line((cx - 5, cy - 5, cx + 5, cy + 5), fill=0, width=2)
        draw.line((cx - 5, cy + 5, cx + 5, cy - 5), fill=0, width=2)


def render_carrier(values: List[int], seed: int | None = None) -> Image.Image:
    if seed is not None:
        random.seed(seed)

    img = Image.new("L", (cfg.carrier_width, cfg.carrier_height), 255)
    draw = ImageDraw.Draw(img)

    left = 30
    top = 20
    usable_h = 120
    col_w = 18

    draw.rectangle((left - 8, top, cfg.carrier_width - 20, top + usable_h), outline=180, width=1)
    for i in range(8):
        y = top + int(i * (usable_h / 7.0))
        draw.line((left - 8, y, cfg.carrier_width - 20, y), fill=200, width=1)

    x = left + 5
    for i, v in enumerate(values):
        row, kind = _value_to_carrier_symbol(v)
        y = top + int(row * (usable_h / 7.0)) + random.randint(-1, 1)
        _draw_marker(draw, x, y, kind)
        x += col_w
        if (i + 1) % 8 == 0 and i + 1 < len(values):
            draw.line((x, top - 4, x, top + usable_h + 4), fill=100, width=2)
            x += 8
        if x > cfg.carrier_width - 24:
            break

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.5)))

    return img
