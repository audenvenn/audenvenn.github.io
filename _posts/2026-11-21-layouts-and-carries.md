---
layout: post
title: "A CuTe layout is additive up to carries"
date: 2026-11-21 14:00:00 +0800
---

*About the preprint [Layouts and Carries: A Mathematical Foundation for the CuTe Layout Algebra](https://github.com/audenvenn/layouts-and-carries) (Auden Venn, 2026), which has not been peer-reviewed. The repository has the paper, a Lean 4 formalisation of its core results, and the programs that check the theory and compare it with three CuTe implementations.*

## Short version

CuTe, the core of NVIDIA's CUTLASS, describes how tensors, threads and hardware instructions index memory with *layouts*, nested (shape, stride) pairs. It manipulates them with an algebra of partial operations: coalesce, composition, complement, logical division and product. The operations are specified by post-conditions and guarded by divisibility checks. The paper shows that the algebra is governed by one identity: a layout is additive up to carries, and each carry is charged a jump.

![Left: the padded layout (4,8):(1,5), whose function is x + ⌊x/4⌋, sends the indices 0 to 11 to the offsets 0 to 13 and skips the offsets 4 and 9, one for each carry into the second digit. Right: A = (3,4):(1,5) composed with B = (2,2):(1,2). The true offsets are 0, 1, 2, 5; the mode-by-mode result gives 0, 1, 2, 3, and offset 3 is padding.](/assets/cute-carries.png)

## The identity

A flat layout (s<sub>0</sub>, …, s<sub>m−1</sub>) : (d<sub>0</sub>, …, d<sub>m−1</sub>) is a function on 0 ≤ x < N, where N = s<sub>0</sub>⋯s<sub>m−1</sub>. To evaluate it, write x in the mixed radix given by the shape, and take the inner product of its digits with the strides. Two quantities describe it:
- the prefix products P<sub>i</sub> = s<sub>0</sub>⋯s<sub>i−1</sub>, the weights of the digits;
- the *jumps* γ<sub>i</sub> = d<sub>i</sub> − s<sub>i−1</sub>·d<sub>i−1</sub>, how far the stride of digit i departs from continuing the previous digit linearly.

Then

```
f(x)     = x·d0 + Σ_i ⌊x / P_i⌋·γ_i
f(x + y) = f(x) + f(y) + Σ_i c_i(x, y)·γ_i
```

where c<sub>i</sub>(x, y), 0 or 1, is the carry into digit i when x and y are added in that radix. Here are three layouts of shape (4, 8), for 0 ≤ x < 32:
- row-major (4,8):(8,1) is 8x − 31⌊x/4⌋;
- padded (4,8):(1,5) is x + ⌊x/4⌋: every carry into the second digit skips one offset;
- column-major (4,8):(1,4) has no jump, and is just x.

## What follows from it

**Which functions are layouts.** Every function that vanishes at 0 has a unique expansion Σ<sub>Q</sub> γ<sub>Q</sub>⌊x/Q⌋. It is a layout function exactly when the scales Q ≥ 2 that occur form a divisibility chain and divide N. Coalescing drops the zero jumps. That gives the normal form, and an algorithm that recognises layouts.

**Composition.** If A∘B exists, nothing is left to choose: its strides are A evaluated at the strides of B. Whether that *naive composite* is right is decided by a *carry defect*. Write B's value at the coordinate (k<sub>0</sub>, k<sub>1</sub>, …) as Σ<sub>j</sub> k<sub>j</sub>·r<sub>j</sub> and add it up in A's radix by the schoolbook method; the defect counts the carries that occur, each weighted by A's jump at that scale. A∘B exists exactly when the defect vanishes for some subdivision of B's modes.

For some A this becomes a simple inequality. Take B = (t<sub>0</sub>, …):(r<sub>0</sub>, …), and suppose A's non-zero jumps all have the same sign, as for row-major, column-major and padded layouts. Then the test is one inequality per scale P of A that carries a non-zero jump, possibly after B's modes are subdivided:

```
Σ_j (t_j − 1)·(r_j mod P) < P
```

CuTe's stride and shape divisibility conditions cover a special case: strides with a single non-zero digit that divides the size of its mode.

**Complements and division.** A layout and its complement form a direct-sum decomposition of an interval, a classical subject (Krasner factorisations). The paper uses Long's 1967 formulation; de Bruijn (1956) treated the analogous question for all natural numbers. So the divisibility condition for a complement is not a limitation of the layout form: any set has to meet it. Logical division has an exact criterion as well. Dividing A by a tile B means composing A with B and B's complement, which together form a compact layout: a bijection onto A's index range. Provided B has such a complement, the composite exists exactly when every scale at which A has a non-zero jump is comparable, under divisibility, with every scale of this tiling.

**Other codomains.** The results on normal forms and composition hold for strides in any abelian group. That covers negative strides, coordinate-valued layouts, and XOR-based layouts like Triton's linear layouts. Triton's composition is always defined, because intermediate values are combined by XOR, which has no carries. CuTe adds integers, which do carry.

**Swizzles.** Offsets have two additions, integer + and XOR, and they differ exactly by the carries of the binary radix. A swizzle, such as CuTe's Swizzle&lt;B,M,S&gt; that XORs one bit field into another, is a layout for XOR. The paper does three things here:
- it characterises when a swizzled layout with offset 0 is XOR-linear;
- it gives a closed-form criterion for a slice of a swizzled layout whose inner layout is bit-disjoint to decay to an ordinary layout;
- it states conditions under which CuTe's operations on swizzled layouts are valid.

For example, take Swizzle&lt;1,1,1&gt; ∘ (4,2):(1,4) and fix the second coordinate to 1. The offsets are 6, 7, 4, 5: the ordinary layout (2,2):(1,−2) plus 6. C++ CuTe returns exactly that.

## Two small examples

Both were checked against the implementations.
- **A composite that comes out silently wrong.**
  - A = (3,4):(1,5) is a 3×4 column-major matrix with two padding slots after each column: f(x) = x + 2⌊x/3⌋.
  - B = (2,2):(1,2) views its first four elements as 2×2. The true offsets are 0, 1, 2, 5.
  - C++ CuTe (with static integers) and PyCuTe return (2,2):(1,2), whose offsets are 0, 1, 2, 3. CUTLASS's pycute rejects the call.
  - At the last element, 1 + 2 = 3 crosses A's scale 3, a carry that the mode-by-mode rule does not see. The true offset is 5; the returned one, 3, is a padding slot.
- **A composable pair that is rejected.**
  - A = (10,10,10):(d<sub>0</sub>, d<sub>1</sub>, d<sub>2</sub>) composed with B = 10:11 is 10:(d<sub>0</sub> + d<sub>1</sub>), whatever the strides. 11 has the base-10 digits (1, 1), and no multiple of it up to 9·11 = 99 carries.
  - Since 11 and 10 do not divide each other, the three implementations reject the call. This was checked with the strides (3,50,700), where the answer is 10:53.
  - With the strides (1,10,100), A coalesces to 1000:1 and the call is accepted.

## What the tests say about implementations

The repository compares the theory, on small synthetic layouts, with three implementations: C++ CuTe, CUTLASS's Python implementation pycute, and the PyCuTe reference implementation. The criteria explain two kinds of discrepancy:
- composites that come out silently wrong. This happens when modes of the inner layout interfere. In the C++ headers it also happens for some single-mode inner layouts, because the static stride check there is weaker than documented.
- composable pairs that are rejected.

These tests show that the behaviours exist. They say nothing about how often the behaviours occur in real kernels.

## What is machine-checked

About 1,500 lines of Lean 4 with Mathlib, with no `sorry`, check the core results:
- the floor-sum form;
- the carry identity;
- layouts as the carry-free additive maps;
- the existence and uniqueness of the ⌊x/Q⌋ expansion;
- the divisibility-chain criterion;
- the composition formula with its defect;
- the carry-free inequality for a fixed shape, in both directions;
- that a swizzle (one bit field, right shift) is an involution and XOR-linear.

Everything else, including the exact composition theorem, complements, division and the rest of the swizzle section, has paper proofs, validated by exhaustive and randomised programs. Section 9 of the paper lists which statement rests on what.

## What is new

The normal-form theorem for coalesce and the existence of complements were known and are cited in the paper. New are the carry identity as an organising principle, the chain criterion, the exact criteria for composition and division, and the treatment of swizzles.

---

*Disclosure, from the paper's acknowledgments: This work was carried out with substantial assistance from an AI system (Claude, developed by Anthropic), which, under the author's direction, surveyed the literature, proposed and developed the framework, wrote the proofs, the Lean formalisation and the test programs, and drafted the manuscript. Drafts were independently reviewed by a second AI system (Codex, developed by OpenAI), which found a number of errors that have been corrected and contributed several examples and counterexamples. The author takes responsibility for the content. The AI systems are not authors.*

*This post was also written with an AI assistant.*
