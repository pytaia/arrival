import torch
import onnx
from config import Config
from model import CRNN

cfg = Config()


def main():
    ckpt = torch.load(f"{cfg.checkpoint_dir}/best.pt", map_location="cpu")

    model = CRNN()
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 1, cfg.image_height, cfg.image_width)

    out_path = "model.onnx"

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {1: "batch"}},
        opset_version=13,
    )

    # Перепаковать в один файл без external data
    m = onnx.load(out_path, load_external_data=True)
    onnx.save_model(
        m,
        out_path,
        save_as_external_data=False,
    )

    print("saved single-file model.onnx")


if __name__ == "__main__":
    main()
