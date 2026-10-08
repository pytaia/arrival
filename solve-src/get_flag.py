import sys
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from vocab import decode_ctc_greedy, token_to_value, BAR_TOKEN
from payload_codec import canonical_values_to_text


def get_model_hw(session: ort.InferenceSession) -> Tuple[int, int]:
    inp = session.get_inputs()[0]
    shape = inp.shape
    h = shape[2]
    w = shape[3]
    if not isinstance(h, int) or not isinstance(w, int):
        raise RuntimeError(f"Model input H/W must be static, got {shape}")
    return h, w


def load_image(path: str, expected_h: int, expected_w: int):
    img = Image.open(path).convert("L")
    if img.size != (expected_w, expected_h):
        img = img.resize((expected_w, expected_h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    return arr[None, None, :, :]


def tokens_to_values(tokens: List[str]) -> List[int]:
    values = []
    for t in tokens:
        if t == BAR_TOKEN:
            continue
        values.append(token_to_value(t))
    return values


def main():
    if len(sys.argv) != 3:
        print("usage: python get_flag.py <model.onnx> <image.png>")
        raise SystemExit(1)

    model_path, image_path = sys.argv[1], sys.argv[2]

    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        h, w = get_model_hw(sess)
        x = load_image(image_path, h, w)

        logits = sess.run(None, {"image": x})[0]
        pred = logits.argmax(axis=-1)[:, 0].tolist()
        tokens = decode_ctc_greedy(pred)
        values = tokens_to_values(tokens)

        text = canonical_values_to_text(values)

        print(text)

    except Exception:
        print("invalid")


if __name__ == "__main__":
    main()
