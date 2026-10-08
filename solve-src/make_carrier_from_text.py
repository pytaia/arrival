import argparse
import random

from payload_codec import text_to_canonical_values
from carrier_renderer import render_carrier


def main():
    parser = argparse.ArgumentParser(
        description="Generate a public carrier image from text."
    )
    parser.add_argument("text", help="Arbitrary text to encode")
    parser.add_argument("output", help="Output PNG path")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    args = parser.parse_args()

    text = args.text.strip()

    if args.seed is None:
        args.seed = random.randint(0, 10_000_000)

    values = text_to_canonical_values(text)
    img = render_carrier(values, seed=args.seed)
    img.save(args.output)

    print("saved:", args.output)
    print("text:", text)
    print("seed:", args.seed)
    print("num_values:", len(values))
    print("values:", values)


if __name__ == "__main__":
    main()
