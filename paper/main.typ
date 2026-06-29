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
2.8M WONNText drastically outperforms a parameter-matched bidirectional Transformer baseline in
accuracy on simple mathematical reasoning tasks, such as arithmetic.

= Introduction

The Winfree Oscillatory Neural Network (WONN) was originally proposed for
constraint satisfaction tasks such as Sudoku, where each cell is modelled as a
phase oscillator whose natural frequency encodes prior constraints and whose
coupling to neighbours drives the system toward a consistent solution. We adapt
this framework to masked language modeling (MLM) with four key changes: (1) the
token embedding serves as the oscillator's natural frequency, so token
identity is carried by frequency content rather than a static vector; (2) the
2-D spatial convolution coupling is replaced by 1-D bidirectional self-attention
with RoPE over token positions; (3) the output projection is tied to the input
embedding for honest parameter counts; and (4) the constraint-satisfaction loss
is replaced by masked cross-entropy.

We call the resulting model WONNText. Our contributions are: (i) the
architecture adaptation from constraint satisfaction to general-purpose
denoising; (ii) an ablation study isolating the importance of
token-conditioned frequencies and bidirectional coupling; (iii) a
demonstration that WONNText dramatically outperforms a parameter-matched
Transformer baseline on arithmetic reasoning after fine-tuning; and (iv)
parallelisation strategies for the oscillator recurrence that reduce serial
depth from $O(T)$ to $O(1)$.

= Method

Each token position in layer $ell$ carries a phase $theta in [-pi, pi)$. At
each of $T$ discrete steps within a layer, the phase evolves as:

$ theta_(t+1) = "wrap"_pi (theta_t + gamma (omega + cos(theta_t) dot I(sin(theta_t)))) $

where $omega in RR^(C times N)$ is the natural frequency, set to the token
embedding so that token identity is encoded as frequency content;
$cos(theta_t)$ is the phase-dependent sensitivity that gates the influence
field; $I(sin(theta_t))$ is the influence field, computed as $sin theta$
passed through a per-channel grouped MLP, then bidirectional multi-head
attention with RoPE, then LayerNorm and ReLU; and $gamma$ is a learnable
coupling strength initialised to $0.1$. The wrap operation maps to $[-pi, pi)$.

The model stacks $L$ such layers. The initial phase $theta_0$ is drawn from
$cal N(0, sigma^2)$ with $sigma = 0.1$. The final phase is decoded to
vocabulary logits via $[cos theta, sin theta]$ through a grouped $1 times 1$
convolution and a linear projection whose weight is tied to the token
embedding. We apply random masking at rate $0.15$ and compute cross-entropy
only on masked positions.

The Transformer baseline is a standard encoder with learned positional
embeddings, tied input/output embeddings, and the same masked cross-entropy
loss. Its dimension ($d_"model" = 232$) is chosen to match the WONNText
parameter count ($~2.75M$ vs $~2.96M$).

#figure(
  grid(
    columns: (1fr,),
    align: center,
    row-gutter: 4pt,
    rect(fill: luma(240), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[Token IDs #h(0.5em) $arrow.r$ #h(0.5em) Embedding (tied to output)]
    ],
    [#text(luma(120))[↓]],
    rect(fill: luma(240), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[$omega$ — natural frequency (token embedding)]
    ],
    [#text(luma(120))[↓]],
    rect(fill: luma(240), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[$theta_0 tilde cal N(0, sigma^2), quad sigma = 0.1$]
    ],
    [#text(luma(120))[↓]],
    rect(fill: luma(230), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[
        *WinfreeTextLayer* (repeated $L$ times) \
        #set text(size: 9pt)
        $theta_(t+1) = "wrap"_pi(theta_t + gamma(omega + cos theta_t dot I(sin theta_t)))$ \
        $I = "ReLU"("LN"("Attn"("MLP"(sin theta))))$ — unrolled $T$ times
      ]
    ],
    [#text(luma(120))[↓]],
    rect(fill: luma(240), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[Final phase $theta arrow.r [cos theta, sin theta]$]
    ],
    [#text(luma(120))[↓]],
    rect(fill: luma(240), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[Grouped $1 times 1$ conv $arrow.r$ Tied projection $arrow.r$ Logits]
    ],
    [#text(luma(120))[↓]],
    rect(fill: luma(220), inset: 7pt, radius: 3pt, width: 80%)[
      #align(center)[Masked cross-entropy (mask rate $0.15$)]
    ],
  ),
  caption: [WONNText architecture. Token embeddings serve as oscillator frequencies $omega$. A Winfree layer unrolls $T$ phase-update steps with bidirectional attention coupling. The final phase is decoded back to vocabulary logits via a tied projection.],
)

*Parallelisation.* The $T$-step recurrence is the main serial bottleneck. We
implement three alternative forward modes that trade quality for parallelism:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Mode*], [*Loss*], [*Speedup*], [*Serial depth*],
  [Recurrent ($T = 8$)], [2.02], [$1.0 times$], [$O(T)$],
  [Predictor-corrector], [2.18], [$1.2 times$], [$5$],
  [Parallel scan], [2.20], [$1.9 times$], [$O(1)$],
  [Parallel scan + refine], [2.19], [$1.2 times$], [$O(1)$],
)

Benchmarked on the 2.8M model (2000 arithmetic samples, 3 epochs, CPU). The
parallel-scan mode evaluates the coupling on the free-running (uncoupled)
trajectory $theta_"free"(t) = theta_0 + gamma t dot omega$, then accumulates
corrections via a prefix sum. This reduces serial depth to $O(1)$: one batched
attention call plus one $"cumsum"$. The predictor-corrector mode uses
$t_"pred"$ large-$gamma$ predictor steps followed by $t_"corr"$ normal
corrector steps, reducing serial depth from $T$ to $t_"pred" + t_"corr"$.

== Tokeniser

We train a byte-pair encoding (BPE) tokenizer with a vocabulary of 10\,000
tokens on the WikiText-2 training split. The same tokenizer is reused for all
downstream tasks (including arithmetic) to ensure vocabulary consistency.

== Optimiser

All experiments use Adam with a learning rate of $1 times 10^-3$ for
pretraining and $1 times 10^-4$ for fine-tuning, with gradient clipping at
$1.0$. No weight decay, warmup, or mixed precision is used.

= Experiments

We pretrain both WONNText and the Transformer baseline on WikiText-2, then
run two ablations (random $Omega$, causal attention). We then fine-tune both
pretrained checkpoints on two-digit addition and evaluate on a held-out test
set.

== Pretrain

WikiText-2 (Merity et al.) contains $~2M$ tokens of English text. After BPE
tokenisation ($10\,000$ vocab), we chunk into sequences of length 256 and train
for 50 epochs ($~100M$ tokens seen). At batch size 32, each epoch is $~7\,800$
sequences ($~244$ steps).

Training compute, estimated as $6 times N times D$ where $N$ is parameter
count and $D$ is tokens seen:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Model*], [*Params*], [*Tokens seen*], [*FLOPS*],
  [WONNText], [$2.96M$], [$100M$], [$~1.8 times 10^15$],
  [Transformer], [$2.75M$], [$100M$], [$~1.7 times 10^15$],
)

Both amounts to $~1.7$-$1.8$ PFLOPS — well within the budget of a single L4 GPU
($~30$ TFLOPS bf16) in under an hour of wall-clock time.

== Finetuning

We fine-tune the pretrained checkpoints on two-digit addition
($a + b = c$, operands 10–99), yielding all $90 times 90 = 8\,100$ equations.
We split into 80% train / 10% validation / 10% test (6\,480 / 810 / 810).

The answer digits are replaced with the mask token; the model must predict
them. Both models are initialised from their respective WikiText-2
checkpoints and trained for 100 epochs at batch size 64, learning rate
$1 times 10^-4$. The same BPE tokenizer is used (digits are single tokens).

== Results

The two-digit arithmetic test set is a fixed partition of all 8,100 equations
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

= Discussion

[TODO]

= Future work

+ Investigate scaling laws for WONNText, training a complete matrix of 10M and 25M models. 
+ Look into internal model representations for interpretable circuits.
+ Applying COBRA architecture to WONNText
+ Gather traning dynamics to analyse the model for grokking and behavior emergence

= References

- Jiawen-Dai/WONN. Original Winfree Oscillatory Neural Network for Sudoku.
- Merity, S. et al. WikiText-2 language modeling dataset.
- Vaswani, A. et al. Attention Is All You Need (Transformer baseline).

= Appendix A 

== Hyperparameters

#table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  [*Hyperparameter*], [*WONNText*], [*Transformer*],
  [Channels / $d_"model"$], [256], [232],
  [Layers ($L$)], [1], [1],
  [Attention heads], [8], [8],
  [FFN / hidden ratio], [$2 times$ ch], [$2 times d_"model"$],
  [Unroll steps ($T$)], [16], [---],
  [Coupling $gamma$], [0.1], [---],
  [$theta_"init"$ $sigma$], [0.1], [---],
  [RoPE], [yes], [no],
  [Positional encoding], [---], [learned],
  [Parameters], [$~2.96M$], [$~2.75M$],
  [Vocabulary], [10\,000], [10\,000],
  [Sequence length], [256], [256],
  [Batch size], [32], [32],
  [Epochs], [50], [50],
  [Learning rate], [$1 times 10^-3$], [$1 times 10^-3$],
  [Mask probability], [0.15], [0.15],
)

== Ablations

We isolate the contribution of individual design choices by training ablated
variants with identical hyperparameters, confirming effect size of
bi-directional attention and random $Omega$ initialisation. 

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Variant*], [*Eval loss*], [*PPL*], [*Accuracy*],
  [WONNText (full)], [3.7614], [43.01], [35.34%],
  [random $Omega$], [4.0603], [57.99], [31.77%],
  [causal attention], [5.1900], [179.48], [20.60%],
  [Transformer baseline], [4.0758], [58.90], [30.29%],
)