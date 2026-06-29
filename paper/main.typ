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
proposed for constraint satisfaction tasks, into a general denoising masked-language
model. We call this model WONNText. Our results show that a
parameter-matched WONNText outperforms a bidirectional Transformer baseline in
accuracy on simple mathematical reasoning tasks, such as arithmetic.

= Introduction

TODO

= Method

TODO

= Experiments

All models share a 10k BPE tokenizer trained on WikiText-2. The base
WONNText uses $"ch"=256$, heads $8$, $L=1$, $T=16$; the matched Transformer uses
$d_("model")=232$, one layer, $d_("ff")=464$. Both have approximately 2.8M
parameters. Training uses AdamW with a linear warmup and cosine schedule,
gradient clipping at $1.0$, and bf16 mixed precision.

== Two-digit addition fine-tuning

We fine-tune the pretrained checkpoints on two-digit addition
($a + b = c$, operands 10–99).

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  [*Model*], [*Token acc.*], [*Whole-answer acc.*],
  [WONNText bidirectional], [99.71%], [99.38%],
  [Transformer baseline], [77.29%], [43.95%],
)

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

*Causal attention.* Replacing bidirectional attention with a causal mask dramatically
increases perplexity to 179.48.

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
55.4 percentage-point gap is robust to any reasonable run-to-run variance.

=== Forward-mode variants for parallelisation

Motivated by the limit-cycle diagnostic, we implemented three alternative
Winfree forward modes and benchmarked them on the 2.8M model:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Mode*], [*Eval loss*], [*Time (s)*], [*Serial depth*],
  [recurrent ($T=8$)], [2.02], [17.6], [8],
  [predictor-corrector], [2.18], [15.0], [5],
  [parallel scan], [2.20], [9.4], [$O(1)$],
  [parallel scan + refine], [2.19], [15.1], [$O(1)$],
)

+ *Parallel scan.* The parallel-scan mode evaluates the
attention coupling on the free-running (uncoupled) trajectory and accumulates
corrections via a parallel prefix sum ($"cumsum"$), reducing serial depth from
$O(T)$ to $O(1)$. Its *refine* variant repeats this cycle N times,
taking as an input previous approximation.

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
