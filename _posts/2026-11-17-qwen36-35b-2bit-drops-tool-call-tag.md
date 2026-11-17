---
layout: post
title: "On a 24 GB Mac, ~2-bit Qwen3.6-35B drops its tool-call tag"
date: 2026-11-17 09:10:00 +0800
---

*Measured with [solvetime](https://github.com/audenvenn/solvetime), an open benchmark that scores local LLM setups by coding-agent tasks solved and time to solution, not tokens per second.*

## Short version

On a Mac with 24 GB, the builds of Qwen3.6-35B-A3B that fit in memory are about 2-bit. At 2 bits the model often writes a tool call without its opening `<tool_call>` token, and a server that parses strictly hands the agent plain text instead of a command. Over 32 coding-agent attempts per build:

| Build | Bits | Tasks solved (95% CI) | Attempts lost to malformed replies | Core set, 16 attempts |
|---|---|---|---|---|
| kapitsa 0.2.36 (release) | 2.48 | 19/32 (42-74%) | 13/32 | 6.6 min (failures end early) |
| kapitsa main 7445a6c, with a parser fallback | 2.48 | 28/32 (72-95%) | 0/32 | 13.5 min |
| llama.cpp 0.9.0, unsloth UD-IQ2_XXS | 2.42 | 23/32 (55-84%) | 5/32 | 12.7 min |
| raceline, 4-bit, experts read from the SSD (the author's engine) | 4.06 | 30/32 (80-98%) | 0/32 | 19.6 min |

## What the failure looks like

After its thinking, the model writes the call without the opener:

```
<function=bash>
<parameter=command>
find / -name "main.py" 2>/dev/null | head -20
</parameter>
</function>
</tool_call>
```

The server finds no tool call and returns this as `content` with `finish_reason: "stop"`. The agent (mini-swe-agent 2.4.6 here) tells the model its reply had no command and gives up after three such replies in a row. It comes from the weights, not from one server. With a seeded request, llama.cpp drops the opener on 8 of 10 seeds, deterministically for a given seed. kapitsa's developers logged every token the model sampled on these replies, and `<tool_call>` was never among them.

## The fix is on the parsing side

The model's intent is unambiguous: the function name, its parameters and the closing tag are all there. A server can accept a reply that is one well-formed `<function=…>…</function></tool_call>` call to a tool the request declared, and leave everything else as text. We reported it to kapitsa with a repro ([nace-ai/kapitsa#1277](https://github.com/nace-ai/kapitsa/issues/1277)), and they merged such a fallback the same day. On the same weights, tasks solved went from 19 to 28 of 32 (p = 0.022), with no attempt lost to the missing tag. For llama.cpp the related issue is [ggml-org/llama.cpp#30118](https://github.com/ggml-org/llama.cpp/issues/30118).

## A synthetic speed test pointed the wrong way

raceline keeps the model at 4 bits on 24 GB by keeping some of its experts on the SSD. In a synthetic test (a 4K-token prompt, then 512 tokens of code) it read prompts faster than the 2-bit builds (1268-1333 vs 1091 tok/s) and wrote faster too. In the agent it solved about as many tasks as the fixed kapitsa build (30 vs 28 of 32, which this sample can't tell apart), but it took 1.45 times as long end to end.

The likely reason: each tool output adds a few hundred new tokens, and those route to most of the model's 256 experts, so the 29% of experts raceline keeps on the SSD are read again for every such chunk. Its prompt reading during the runs was 420 tok/s (306 with a smaller expert cache) against kapitsa's 925. The synthetic prompt spread that cost over 4,000 tokens. Measure the workload you will actually run.

## How it was measured

- 16 small coding tasks (8 single-file, 8 closer to a working day), each with a hidden check the agent never sees; 2 attempts each, round-robin.
- mini-swe-agent 2.4.6 with one bash tool, the model card's sampling, thinking on; 50 steps and 15 minutes per attempt.
- Each engine runs the build it picks for 24 GB with its default settings. llama.cpp, which picks none, gets the GGUF that kapitsa picks, as a control.
- The agent runs in a sandbox: it writes only in its task directory and has no network.
- All 128 attempts were replayed from their recorded commands and re-scored. All match.

## Limits

One Mac (an M6 Mac mini), 32 attempts per build, one model so far. MTPLX's 35B packs don't fit 24 GB by its own catalog, and mlx-lm's 4-bit weights exceed the memory Metal gives the GPU, so neither ran.

## Results from your Mac

```sh
pip install git+https://github.com/audenvenn/solvetime
solvetime doctor
solvetime run --engine kapitsa --model qwen3.6-35b-a3b --set core --attempts 2
solvetime submit results/
```

Results from M1 to M5 Macs and other memory sizes are what would make this useful. Every table, build hash, excluded run and raw trajectory is on the [results page](https://github.com/audenvenn/solvetime/blob/main/published/2026-11-qwen3.6-35b-a3b-m6-24gb/README.md).

---

*Disclosure: I maintain raceline, one of the engines measured. This article was written with an AI assistant (Claude); every number comes from the linked results page.*
