# smolvla-listens

**Does a 450M VLA actually read its instructions?** In LIBERO-Goal every task uses the same kitchen, so the instruction is the main thing that tells the robot which task to do. If you swap it for another task's instruction, does SmolVLA do the new task, or does it do the old one anyway? If it does the old one, it is ignoring language. LIBERO-Plus reports exactly this failure in bigger VLAs, but SmolVLA wasn't tested there.

*Work in progress (26–31 Oct 2026).*

## Why it matters

Recent LIBERO robustness studies suggest that VLA policies can score well while largely ignoring their instructions. LIBERO-Plus (Fei et al., arXiv 2510.13626) finds models "largely insensitive to language variations", and LIBERO-PRO (Zhou et al., arXiv 2510.03827) finds that outputs "remain unchanged even when given corrupted instructions". Neither study tested SmolVLA, so it is an open question whether a compact 450M policy actually listens.

## Setup (draft)

Five instruction conditions, each on the same 10 LIBERO-Goal tasks × 10 initial states:

| Condition | What the policy hears in task *i*'s scene |
|---|---|
| `original` | task *i*'s real instruction |
| `paraphrase_a` | light rewording, same objects (`instructions/paraphrase_a.json`) |
| `paraphrase_b` | heavier rewording, verbs and word order changed |
| `empty` | `""` |
| `swapped` | task (*i*+5) mod 10's instruction |

**Can the robot tell the task from the image alone?** Probably not. Episode *k* of every task starts from very similar scenes: the same objects in nearly the same places. A crude pixel check on the first frames found that two different tasks starting from the same init state differ in ~3.8% of pixels, while two episodes of the *same* task differ in ~7.3%. So the starting image gives no task-specific clue, and the instruction is the only reliable signal.
