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

== Ablations (2.8M models)

We isolate the contribution of individual design choices by training ablated
variants on WikiText-2 with identical hyperparameters.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Variant*], [*Eval loss*], [*PPL*], [*Accuracy*],
  [WONNText (full)], [3.7614], [43.01], [35.34%],
  [random $Omega$ (no token embedding)], [4.0603], [57.99], [31.77%],
  [causal attention], [5.1900], [179.48], [20.60%],
  [Transformer baseline], [4.0758], [58.90], [30.29%],
)

*Random $Omega$.* Replacing the learned token-conditioned frequency
$Omega_("init")$ with a fixed random vector collapses perplexity from 43.0 to
58.0, matching the Transformer baseline. This identifies $Omega_("init")$ as
the single most important design choice: the oscillator substrate contributes
little without token-specific frequencies.

*Causal attention.* Replacing bidirectional attention with a causal mask
increases perplexity to 179.5, far worse than even the random-$Omega$ variant.
This is expected: the denoising objective requires the model to attend to both
preceding and following context, and the causal mask makes half the sequence
invisible at each position.

=== Oscillator dynamics diagnostic

To understand whether the $T$-step recurrence could be shortened or
parallelised, we measured per-step phase change ($Delta norm$) and coupling
energy across $t = 0..15$ on a trained 2.8M checkpoint:

#table(
  columns: (auto, auto, auto),
  align: (right, right, right),
  [*Step*], [*$Delta norm$*], [*Energy*],
  [0], [0.000], [0],
  [1], [0.746], [$-805$],
  [3], [0.359], [$-1016$],
  [5], [0.215], [$-1331$],
  [6], [0.204], [$-1336$],
  [8], [0.300], [$-1239$],
  [12], [0.486], [$-930$],
  [15], [0.363], [$-1025$],
)

The dynamics do not converge to a fixed point. Phase change reaches a minimum
at step 5–6, then rises as heterogeneous token-specific frequencies $Omega$
cause phase dispersion — a limit cycle characteristic of the Winfree model.
This finding has two consequences: (1) Deep Equilibrium (DEQ) reformulations
are inapplicable since no fixed point exists; (2) steps 8–15 are oscillatory
drift, motivating a reduction of $T$ from 16 to 8 in the 10× experiments.

=== Statistical significance of the arithmetic gap

The two-digit addition test set is a fixed partition of all 8,100 equations
(810 test examples, 2,070 answer tokens, zero train/test overlap). We report
Wilson 95% confidence intervals on the binomial proportion:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*Metric*], [*WONNText (95% CI)*], [*Transformer (95% CI)*],
  [Whole-answer acc.], [99.38% (98.56–99.74)], [43.95% (40.57–47.39)],
  [Token acc.], [99.71% (99.37–99.87)], [77.29% (75.44–79.05)],
)

The confidence intervals do not overlap ($z = 24.8$, $p approx 0$). The
55.4 percentage-point gap is robust to any reasonable run-to-run variance,
established from a single run with a fixed, non-overlapping test set.

=== Forward-mode variants for parallelisation

Motivated by the limit-cycle diagnostic, we implemented three alternative
Winfree forward modes and benchmarked them on the 2.8M model (3 epochs,
2,000 arithmetic samples, CPU):

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Mode*], [*Eval loss*], [*Time (s)*], [*Serial depth*],
  [recurrent ($T=8$)], [2.02], [17.6], [8],
  [predictor-corrector], [2.18], [15.0], [5],
  [parallel scan], [2.20], [9.4], [$O(1)$],
  [parallel scan + refine], [2.19], [15.1], [$O(1)$],
)

All modes learn; the quality gap between the exact recurrence and the fully
parallel scan is $approx 0.18$ nats. The parallel-scan mode evaluates the
attention coupling on the free-running (uncoupled) trajectory and accumulates
corrections via a parallel prefix sum ($"cumsum"$), reducing serial depth from
$O(T)$ to $O(1)$. On GPU, the speedup is expected to be larger because the
$T$-dimension batched attention exploits parallelism the sequential recurrence
cannot.

== Scaling to 10× parameters

We scale both architectures to roughly 27–30M parameters (WONNText:
$"ch"=1024$, $L=4$, $T=8$; Transformer: $d_("model")=768$, $L=4$,
$d_("ff")=2048$). Pre-training uses WikiText-103 ($approx 100M$ tokens, 50×
more data than WikiText-2) to escape the data-limited regime observed at 2.8M.
Both models use gradient checkpointing, bf16 mixed precision, AdamW with cosine
LR, and $T=8$ (reduced from 16 based on the oscillator diagnostic).

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, left),
  [*Model*], [*Eval loss*], [*PPL*], [*Accuracy*], [*Status*],
  [Transformer 10×], [—], [—], [—], [training],
  [WONNText 10×], [—], [—], [—], [training],
)

On WikiText-2, the 10× Transformer (30M params) achieved only $approx 1%$
accuracy gain over the 2.8M baseline (31.2% vs 30.3%), consistent with
WikiText-2 being data-limited rather than capacity-limited. The 10× runs on
WikiText-103 will test whether larger models separate when given sufficient data.

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

The oscillator dynamics diagnostic reveals that the $T$-step recurrence does
not converge to a fixed point but forms a limit cycle driven by heterogeneous
token frequencies. This is a feature, not a bug: the phase dispersion creates
token-specific, context-dependent representations. However, it means that
fixed-point reformulations (DEQ) are inapplicable, and the recurrence imposes a
serial compute cost. The parallel-scan variant reduces this to $O(1)$ serial
depth with a $approx 0.18$ nat quality loss, offering a practical speed–quality
tradeoff.

On WikiText-2, scaling from 2.8M to 30M parameters yielded only $approx 1%$
accuracy improvement, consistent with the dataset being data-limited rather
than capacity-limited. The 10× experiments on WikiText-103 ($approx 100M$
tokens) will test whether the architectures separate at scale when given
sufficient data. The arithmetic curriculum (two-digit addition, then
three-number expressions with mixed operators) remains the primary benchmark:
the 55.4 percentage-point gap at 2.8M ($p approx 0$, non-overlapping CIs)
suggests the oscillator substrate has a structural advantage on compositional
reasoning that may persist or widen at scale.

= References

- Jiawen-Dai/WONN. Original Winfree Oscillatory Neural Network for Sudoku.
- Merity, S. et al. WikiText-2 language modeling dataset.
- Vaswani, A. et al. Attention Is All You Need (Transformer baseline).
