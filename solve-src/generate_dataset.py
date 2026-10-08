import json
import random

from config import Config
from model_renderer import render_model_image
from utils import ensure_dir

cfg = Config()


def random_values():
    n = random.randint(cfg.min_values, cfg.max_values)
    return [random.randint(0, 31) for _ in range(n)]


def save_split(out_dir: str, labels_path: str, count: int):
    ensure_dir(out_dir)

    with open(labels_path, "w", encoding="utf-8") as f:
        for i in range(count):
            values = random_values()
            img = render_model_image(values, seed=random.randint(0, 10_000_000))
            name = f"{i:06d}.png"
            img.save(f"{out_dir}/{name}")
            f.write(json.dumps({"image": name, "values": values}) + "\n")

            if (i + 1) % 1000 == 0:
                print(f"{out_dir}: {i+1}/{count}")


def main():
    random.seed(cfg.seed)
    save_split(cfg.train_dir, cfg.train_labels, cfg.train_size)
    save_split(cfg.val_dir, cfg.val_labels, cfg.val_size)


if __name__ == "__main__":
    main()
    