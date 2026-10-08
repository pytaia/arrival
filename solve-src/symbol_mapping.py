from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSymbol:
    pitch: int  # 0..7
    style: int  # 0..3


def value_to_model_symbol(v: int) -> ModelSymbol:
    if not (0 <= v < 32):
        raise ValueError("bad value")
    return ModelSymbol(pitch=v // 4, style=v % 4)
