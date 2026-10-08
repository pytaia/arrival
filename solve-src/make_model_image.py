import argparse
import random

from payload_codec import text_to_canonical_values
from model_renderer import render_model_image


def main():
    parser = argparse.ArgumentParser(
        description="Generate a model-readable image in the note-like language from text."
    )
    parser.add_argument("text", help="Arbitrary text to encode")
    parser.add_argument("output", help="Output PNG path")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument("--width", type=int, default=None, help="Optional forced width")
    args = parser.parse_args()

    text = args.text.strip()

    if args.seed is None:
        args.seed = random.randint(0, 10_000_000)

    values = text_to_canonical_values(text)
    img = render_model_image(values, seed=args.seed, force_width=args.width)
    img.save(args.output)

    print("saved:", args.output)
    print("text:", text)
    print("seed:", args.seed)
    print("num_values:", len(values))
    print("image_size:", img.size)
    print("values:", values)


if __name__ == "__main__":
    main()