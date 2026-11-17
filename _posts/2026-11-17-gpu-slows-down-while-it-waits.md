---
layout: post
title: "A Mac's GPU slows down while it waits for the CPU"
date: 2026-11-17 09:00:00 +0800
---


*Every number below is from one Mac mini (Apple M6, 12-core GPU, 24 GB, macOS 27.2), measured back to
back with the setting under test alternated. Sources: raceline issue #47, pull request #50,
`bench/hw/gpu_wait.mm`, `bench/results/gpu_keepalive_m6.txt`.*

**Short version.** When a GPU command buffer pauses to wait for the CPU, the GPU's power management takes
the pause for idleness and lowers the GPU's performance, so the work after each wait runs slower. An engine
that waits on the CPU many times per pass pays this every time. On the M6, a light kernel running on a
second command queue keeps the GPU up. In raceline it made single-token decoding of Qwen3.6-35B-A3B with its
experts on the SSD 27% faster (44.3 to 56.4 tokens/s). A 181-line program reproduces the effect without a
model: `bench/hw/gpu_wait.mm`.

**Who this may affect.** Metal code that makes the GPU wait for the CPU, inside a command buffer (on an `MTLSharedEvent` the CPU signals) or between command buffers: engines that load mixture-of-experts weights on demand, samplers that run on the CPU between passes, and anything that hands work back and forth within a frame. We measured the first; the others follow the same pattern but were not measured.

## The symptom: the same work, 32 to 53 tokens/s

raceline runs Qwen3.6-35B-A3B and Qwen3.5-35B-A3B, mixtures of experts too large for a 24 GB Mac's GPU
memory, by keeping most experts in memory and reading the rest from the SSD as each layer needs them. In
every mixture-of-experts layer, the GPU signals a shared event, a CPU thread reads the missing experts, and
the GPU waits for the CPU's answer before it goes on. A decoding pass waits 40 times.

Single-token decoding, without drafts, varied from one session to the next on the same Mac, build and
settings. On a quiet evening Qwen3.5-35B-A3B decoded at 52.9 tokens/s (18.9 ms a pass); the next morning at
31.7 to 40.3 (25 to 31 ms). Several things were ruled out:

- **The SSD.** Random reads of an expert's size (1.6 MB) ran at 4.2 GB/s one at a time and 7.4 to 7.8 GB/s
  eight at a time, as on the first day.
- **The amount of work.** The engine loaded exactly as many experts on both days (15,745).
- **The file cache.** After the model's pages were evicted (30 GB of other files read), four runs in a row
  were equally slow, and the disk delivered every expert load each time.

The cost appeared per pass, not per byte. Passes with several rows cost the same on both days, while
single-row passes cost 12 ms more.

## Narrowing it down

On a tiny test model, one handoff cost about 600 µs per layer inside the engine, against 52 µs for the bare
`MTLSharedEvent` round trip in a standalone program. Timestamps in the expert-cache thread put its own work
at about 5 µs. The time went into the GPU resuming after each wait, and most of all into the gaps between
passes: 2.5 ms of gap where a whole pass, with every expert in memory, takes 1.0 ms. Keeping the CPU thread
spinning instead of sleeping changed nothing. What changed everything was giving the GPU something to do
during the waits:

| Qwen3.6-35B-A3B, experts on disk, no drafts | decoding |
|---|---|
| alone | 32.5 tokens/s |
| with a light kernel running alongside on another queue (1024 threads) | 51.7 tokens/s |
| the same with 4096 threads | 48.8 tokens/s |

On the tiny model the same light kernel made the passes 2.6 times faster.

## A repro without a model

`bench/hw/gpu_wait.mm` runs the same pattern with no model. It builds a command buffer of "layers" of real
work, a number of small dispatches each, and after every layer the GPU hands over to the CPU. The CPU
answers from a spinning thread, either at once or after a set delay, and the program times three versions:

- **work only:** the same dispatches, with no handoffs;
- **handoffs:** a GPU to CPU to GPU round trip after every layer;
- **+ keep-alive:** the same, with a light kernel looping on a second queue the whole time.

It prints the time per layer beyond the work and the CPU's delay. On the M6, over two runs:

| The CPU answers | lost per layer beyond the work and the delay | with the keep-alive |
|---|---|---|
| at once | 43 to 68 µs | 38 to 64 µs |
| after 100 to 300 µs (about an SSD read) | 79 to 102 µs | 50 to 67 µs |

```
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation \
  -o build/hw/gpu_wait bench/hw/gpu_wait.mm
build/hw/gpu_wait            # delays of 0, 100 and 300 µs
```

The longer the CPU takes, the more the GPU loses on top of the wait itself. That is the signature of a GPU
that clocks down while it is idle, and it hits hardest exactly when the CPU is doing real work, such as
reading from a disk.

## The fix

raceline's expert cache now keeps such a kernel running back to back, on a command queue of its own, while
requests come in, and lets it stop 50 ms after the last one (`RACELINE_GPU_KEEPALIVE=0` turns it off). The
kernel is `kernels/keepalive.metal`: 1024 threads, a few hundred multiply-adds each, eight dispatches per
command buffer. It shares no memory with the model, so what the model computes does not change; the tests
that require bit-identical output in every mode still pass.

Measured with the keep-alive on and off alternated:

| Qwen3.6-35B-A3B, experts on disk | keep-alive | without |
|---|---|---|
| 24 GB Mac, no drafts | 56.1 / 56.6 tokens/s | 44.2 / 44.4 |
| the same, with its MTP drafts | 121.8 / 122.0 | 116.0 / 116.4 |
| as a 16 GB Mac would run it (5.79 GB of experts in memory), no drafts | 24.1 | 15.6 |
| the same, with drafts | 30.1 | 28.2 |

Drafts gain less: a drafted pass waits as often, 40 times, but yields several tokens, and the 256 tokens
took 57 passes instead of 256. Prompt processing barely moved: 1,096 to 1,105 tokens/s on a 1,024-token
prompt with the keep-alive, 1,057 to 1,089 without.

Dense models do not wait on the CPU inside a pass, and for them the extra kernel only competes for the GPU:
Qwen3.5-9B read a prompt at 997 tokens/s with it against 1,119 without, and decoded at 30.9 against 31.9.
raceline leaves them alone.

## What we do not know

- **One machine.** Whether other Apple GPUs (M1 to M5, Pro, Max) behave the same, and by how much, is open.
  `bench/hw/gpu_wait.mm` needs only Xcode's command-line tools; results from other Macs are welcome on the
  repository.
- **Power.** A GPU kept from idling draws more power. We did not measure how much, and for an engine on
  battery, that trade may not pay.
- **The mechanism.** We see its effect from outside. We have not read anything from Apple that describes
  this behaviour, and the keep-alive is a workaround that works, not a documented interface.

Code, raw output and the full investigation are in raceline (issue #47 and pull request #50). raceline is
now in maintenance mode, and this is one of the things it found along the way.


---

*Written with an AI assistant (Claude); the measurements and code are in the linked repository, and every number above comes from the files named at the top.*
