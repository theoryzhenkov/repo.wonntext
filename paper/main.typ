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

[::TODO This section is intentionally empty]

= Method

[::TODO Describe the architecture of the model, including our optimisations here. Try to be concise, but don't leave out important optimisation details.
Ideally, generate an accompanying picture showcasing model's architecture]

== Tokeniser

[::TODO Describe our tokeniser]

== Optimiser

[::TODO Describe our optimiser]

== Hyperparameters

[::TODO Provide hyperparameters with which the model was trained]

= Experiments

[::TODO Give a conscise overview of our experiments]

== Pretrain

[::TODO Describe the dataset for pretrain, give a FLOPS estimate, etc.]

== Finetuning

[::TODO Describe the finetuning regime]

We fine-tune the pretrained checkpoints on two-digit arithmetics
($a {+,-} b = c$, operands 10–99).

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

== Ablations

We isolate the contribution of individual design choices by training ablated
variants on WikiText-2 with identical hyperparameters, confirming effect size of
bi-directional attention and random initialisation. 

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Variant*], [*Eval loss*], [*PPL*], [*Accuracy*],
  [WONNText (full)], [3.7614], [43.01], [35.34%],
  [random $Omega$], [4.0603], [57.99], [31.77%],
  [causal attention], [5.1900], [179.48], [20.60%],
  [Transformer baseline], [4.0758], [58.90], [30.29%],
)

= Discussion

[::TODO give our most important results, don't make this section too long or verbose]

= References

- Jiawen-Dai/WONN. Original Winfree Oscillatory Neural Network for Sudoku.
- Merity, S. et al. WikiText-2 language modeling dataset.
- Vaswani, A. et al. Attention Is All You Need (Transformer baseline).
