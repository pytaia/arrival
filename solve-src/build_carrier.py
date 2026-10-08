import hashlib

from organizer_secret import ORGANIZER_SECRET_TEXT
from payload_codec import text_to_canonical_values
from carrier_renderer import render_carrier


def main():
    values = text_to_canonical_values(ORGANIZER_SECRET_TEXT)
    img = render_carrier(values, seed=1337)
    img.save("carrier.png")

    print("saved carrier.png")
    print("num_values =", len(values))
    print("values =", values)


if __name__ == "__main__":
    main()
