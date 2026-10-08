import json
import os
import sys
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from vocab import decode_ctc_greedy, token_to_value, BAR_TOKEN


def get_model_hw(session: ort.InferenceSession) -> Tuple[int, int]:
    inp = session.get_inputs()[0]
    shape = inp.shape
    h = shape[2]
    w = shape[3]
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
    if len(sys.argv) != 4:
        print("usage: python check_sample.py <model.onnx> <labels.jsonl> <image_dir>")
        raise SystemExit(1)

    model_path, labels_path, image_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(labels_path, "r", encoding="utf-8") as f:
        row = json.loads(next(f))

    image_path = os.path.join(image_dir, row["image"])
    gt_values = row["values"]

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    h, w = get_model_hw(sess)
    x = load_image(image_path, h, w)

    logits = sess.run(None, {"image": x})[0]
    pred = logits.argmax(axis=-1)[:, 0].tolist()
    tokens = decode_ctc_greedy(pred)
    pred_values = tokens_to_values(tokens)

    print("image:", row["image"])
    print("gt values:  ", gt_values)
    print("pred tokens:", tokens)
    print("pred values:", pred_values)
    print("match:", gt_values == pred_values)


if __name__ == "__main__":
    main()
