#set document(
  title: "WONNText: A Winfree Oscillatory Network for Masked Language Modeling and Arithmetic Reasoning",
  author: ("theoryzhenkov",),
  date: datetime(year: 2026, month: 6, day: 29),
)

#set page(paper: "us-letter", margin: (x: 1.1in, y: 1.1in))
#set text(size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#show heading.where(level: 1): it => v(0.6em) + text(weight: "bold", it.body) + v(0.2em)
#show heading.where(level: 2): it => v(0.4em) + text(weight: "bold", it.body)

#align(center)[
  #text(size: 16pt, weight: "bold")[
    WONNText: A Winfree Oscillatory Network for LM
  ]
  #v(0.4em)
  #text(size: 10pt)[theoryzhenkov \
  #emph[Draft preprint, 2026-06-29]]
]
#v(0.8em)

*Abstract.* We adapt the Winfree Oscillatory Neural Network (WONN), originally
proposed for 9×9 Sudoku constraint satisfaction, into a denoising masked-language
model. The resulting model, WONNText, keeps the oscillator dynamics of the
original architecture but replaces the 2-D grid coupling with 1-D bidirectional
attention over token positions, ties the input embedding to the output
projection, and is trained with masked cross-entropy. On WikiText-2, a
parameter-matched WONNText outperforms a bidirectional Transformer baseline in
perplexity and accuracy. Ablations isolate the contribution of the learned
oscillator-frequency embedding and of bidirectional coupling. Fine-tuning on a
two-digit addition task shows that WONNText reaches near-perfect whole-answer
accuracy while the Transformer baseline plateaus near chance. Preliminary
scaling experiments to 10× parameters are ongoing.

= Introduction

The Winfree model of coupled biological oscillators has inspired a family of
neural architectures in which units carry both a phase and a frequency and
interact through phase-dependent coupling. Recent work (Jiawen-Dai/WONN)
demonstrated that a single Winfree layer with attention coupling can solve
9×9 Sudoku as a constraint-satisfaction problem. This raises a natural
question: can the same oscillator substrate serve as a general sequence model?

We investigate this by converting the Sudoku WONN into a masked-language model,
WONNText, and benchmarking it against a parameter-matched bidirectional
Transformer on WikiText-2 and on a small arithmetic reasoning task.

= Method

WONNText preserves the core Winfree dynamics but makes four changes relative to
the Sudoku configuration:

+ *Input as frequency.* Token ids are embedded, and the embedding vector is
  used directly as the initial oscillator frequency $Omega_("init")$. Initial
  phases $Theta_("init")$ are drawn from $cal N(0, sigma^2)$. The mask token
  is an ordinary vocabulary entry.

+ *1-D bidirectional coupling.* The 2-D grid is replaced by a 1-D token
  sequence with full bidirectional attention and 1-D rotary position embeddings
  (RoPE). Sequence length is configurable in the 128–256 range.

+ *Tied output head.* The final phase at each position is mapped to
  $(sin theta, cos theta)$ features, projected to the embedding dimension, and
  a linear layer produces vocabulary logits. The output projection is tied to
  the input embedding.

+ *Masked cross-entropy objective.* The Sudoku constraint loss is replaced by
  standard masked-language-modeling loss.

Each Winfree layer unrolls for $T$ recurrent steps. At every step, $cos theta$
(the sensitivity) modulates an attention field computed from $sin theta$
(the influence), and phases are updated and wrapped to $[-pi, pi)$.

= Experiments

All models share a 10k BPE tokenizer trained on WikiText-2. The base
WONNText uses $"ch"=256$, heads $8$, $L=1$, $T=16$; the matched Transformer uses
$d_("model")=232$, one layer, $d_("ff")=464$. Both have approximately 2.8M
parameters. Training uses AdamW with a linear warmup and cosine schedule,
gradient clipping at $1.0$, and bf16 mixed precision.

== WikiText-2 language modeling

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Model*], [*Eval loss*], [*PPL*], [*Accuracy*],
  [WONNText bidirectional], [3.7614], [43.01], [35.34%],
  [Transformer baseline], [4.0758], [58.90], [30.29%],
  [WONNText random $Omega$], [4.0603], [57.99], [31.77%],
  [WONNText causal], [5.1900], [179.48], [20.60%],
)

WONNText achieves the lowest perplexity and highest accuracy. Removing the
learned frequency embedding (random $Omega$) collapses performance to the
Transformer baseline, indicating that the token-conditioned $Omega_("init")$
is the dominant factor. Replacing bidirectional attention with a causal mask
is catastrophic, consistent with the denoising (non-autoregressive) objective.

== Two-digit addition fine-tuning

We fine-tune the pretrained checkpoints on two-digit addition
($a + b = c$, operands 10–99). The answer digits are masked and the model must
predict them. The train/valid/test split is a random partition of all 8,100
equations with zero overlap between splits.

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  [*Model*], [*Token acc.*], [*Whole-answer acc.*],
  [WONNText bidirectional], [99.71%], [99.38%],
  [Transformer baseline], [77.29%], [43.95%],
)

The Transformer plateaus near 44% whole-answer accuracy, while WONNText
generalizes to near-perfect addition on unseen operand pairs.

== Scaling to 10× parameters (preliminary)

We scale both architectures to roughly 27–30M parameters (WONNText:
$"ch"=1024$, $L=4$, $T=16$; Transformer: $d_("model")=768$, $L=4$,
$d_("ff")=2048$). The Transformer pre-training is complete; WONNText training
is ongoing.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, left),
  [*Model*], [*Eval loss*], [*PPL*], [*Accuracy*], [*Status*],
  [Transformer 10×], [4.02], [55.8], [31.2%], [complete],
  [WONNText 10×], [—], [—], [—], [training],
)

A two-stage arithmetic curriculum (stage 1: two-digit addition; stage 2:
three-number expressions with $+,-,times,div$ and standard precedence) will be
applied to both 10× checkpoints once pre-training completes.

= Discussion

The results suggest that the oscillator substrate is not merely a constraint
solver: with a token-conditioned frequency embedding and bidirectional
coupling, it becomes a competitive sequence model that also transfers better
to a symbolic reasoning task. The dominant architectural factor is the learned
$Omega_("init")$; the attention directionality is secondary but necessary for
the denoising objective.

A practical limitation is compute: WONNText's $T$-step recurrence is
substantially slower per epoch than a depth-matched Transformer. We discuss
gradient checkpointing and fixed-point (Deep Equilibrium) reformulations as
future work to mitigate this.

= References

- Jiawen-Dai/WONN. Original Winfree Oscillatory Neural Network for Sudoku.
- Merity, S. et al. WikiText-2 language modeling dataset.
- Vaswani, A. et al. Attention Is All You Need (Transformer baseline).
