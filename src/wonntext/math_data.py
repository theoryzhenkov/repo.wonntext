"""Online +/- arithmetic data for the WONNText study.

The equation space (2-4 operands, 1-3 digit, addition/subtraction) is far larger
than any 2-10M model's memorisation capacity, so training samples fresh examples
every step (:class:`OnlineMathDataset`) and overfitting is structurally
impossible. Two fixed, RNG-disjoint held-out sets are built once
(:func:`build_fixed_set`):

  * in-distribution  - same 2-4 operands / 1-3 digit -> clean generalisation
  * extrapolation    - 5 operands -> rule-vs-shortcut generalisation

Character-level vocabulary (15 symbols) keeps the param budget in the layers.
The answer (right of ``=``) is the masked span the model predicts.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

# Fixed character vocabulary. PAD/MASK first so ids are stable.
PAD, MASK = "<pad>", "<mask>"
CHARS = list("0123456789+-=")
STOI = {PAD: 0, MASK: 1, **{c: i + 2 for i, c in enumerate(CHARS)}}
ITOS = {i: c for c, i in STOI.items()}
PAD_ID, MASK_ID = STOI[PAD], STOI[MASK]
VOCAB_SIZE = len(STOI)

# Default max sequence length: 5 operands x 3 digits + 4 ops + '=' + sign + 4
# answer digits = 25; 32 leaves margin and a clean power-of-two-ish width.
DEFAULT_SEQ_LEN = 32


def _operand(rng: random.Random, min_digits: int, max_digits: int) -> int:
    d = rng.randint(min_digits, max_digits)
    low = 0 if d == 1 else 10 ** (d - 1)
    return rng.randint(low, 10**d - 1)


def sample_equation(
    rng: random.Random,
    min_operands: int,
    max_operands: int,
    min_digits: int,
    max_digits: int,
) -> tuple[str, str]:
    """Return (question_with_equals, answer_str) for a fresh +/- equation."""
    n = rng.randint(min_operands, max_operands)
    operands = [_operand(rng, min_digits, max_digits) for _ in range(n)]
    ops = [rng.choice("+-") for _ in range(n - 1)]
    acc = operands[0]
    q = str(operands[0])
    for op, x in zip(ops, operands[1:], strict=True):
        acc = acc + x if op == "+" else acc - x
        q += f"{op}{x}"
    return q + "=", str(acc)


def encode(q: str, a: str, seq_len: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Encode 'q'+'a' into padded ids + an answer-position mask, or None if too long."""
    s = q + a
    if len(s) > seq_len:
        return None
    ids = [STOI[c] for c in s] + [PAD_ID] * (seq_len - len(s))
    mask = [False] * seq_len
    for i in range(len(q), len(s)):
        mask[i] = True
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class OnlineMathDataset(IterableDataset):
    """Infinite stream of fresh encoded equations; skips any held-out question."""

    def __init__(
        self,
        min_operands: int,
        max_operands: int,
        min_digits: int,
        max_digits: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        exclude: set[str] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = (min_operands, max_operands, min_digits, max_digits)
        self.seq_len = int(seq_len)
        self.exclude = exclude or set()
        self.seed = int(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = random.Random((self.seed << 16) ^ wid)
        while True:
            q, a = sample_equation(rng, *self.cfg)
            if q in self.exclude:
                continue
            enc = encode(q, a, self.seq_len)
            if enc is None:
                continue
            yield {"input_ids": enc[0], "answer_mask": enc[1]}


def build_fixed_set(
    n: int,
    min_operands: int,
    max_operands: int,
    min_digits: int,
    max_digits: int,
    seq_len: int,
    seed: int,
    exclude: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, set[str]]:
    """Build a deduplicated fixed set; returns (ids, answer_mask, question_set)."""
    rng = random.Random(seed)
    seen: set[str] = set(exclude or ())
    questions: set[str] = set()
    ids_list, mask_list = [], []
    attempts = 0
    while len(ids_list) < n and attempts < n * 50:
        attempts += 1
        q, a = sample_equation(rng, min_operands, max_operands, min_digits, max_digits)
        if q in seen:
            continue
        enc = encode(q, a, seq_len)
        if enc is None:
            continue
        seen.add(q)
        questions.add(q)
        ids_list.append(enc[0])
        mask_list.append(enc[1])
    return torch.stack(ids_list), torch.stack(mask_list), questions
