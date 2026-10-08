from typing import List, Dict

TOKENS: List[str] = ["<blank>"] + [f"K{i:02d}" for i in range(32)] + ["B"]

TOKEN_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(TOKENS)}
ID_TO_TOKEN: Dict[int, str] = {i: t for t, i in TOKEN_TO_ID.items()}

BLANK_ID = TOKEN_TO_ID["<blank>"]
BAR_TOKEN = "B"


def value_to_token(v: int) -> str:
    if not (0 <= v < 32):
        raise ValueError(f"value out of range: {v}")
    return f"K{v:02d}"


def token_to_value(t: str) -> int:
    if t == BAR_TOKEN:
        raise ValueError("bar token has no value")
    if not t.startswith("K"):
        raise ValueError(f"bad token: {t}")
    v = int(t[1:])
    if not (0 <= v < 32):
        raise ValueError(f"token out of range: {t}")
    return v


def encode_tokens(tokens: List[str]) -> List[int]:
    return [TOKEN_TO_ID[t] for t in tokens]


def decode_ctc_greedy(ids: List[int]) -> List[str]:
    out = []
    prev = None
    for x in ids:
        if x != prev and x != BLANK_ID:
            out.append(ID_TO_TOKEN[x])
        prev = x
    return out
