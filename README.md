# LBD-Q-SAT

### Teaching a computer to solve logic puzzles smarter, using reinforcement learning

## What problem are we solving?

Imagine a giant logic puzzle made up of thousands of true/false statements, all linked together, where you need to find an assignment of "true" or "false" to each piece so that every rule is satisfied. This is called a **SAT problem** (short for "satisfiability"), and it shows up everywhere — chip design, scheduling, software verification, even AI planning.

Computers solve these puzzles using a method called **CDCL** (don't worry about the acronym). At every step, the solver has to make a guess: "let's try setting this particular piece to true." A good guess can solve the puzzle quickly. A bad guess wastes time backtracking. For decades, this guessing strategy has been a fixed, hand-crafted rule that experts designed and fine-tuned.

## What are we doing differently?

Instead of using a fixed rule, this project trains a **neural network** to make that guess — and it learns to do this the same way a person learns a video game: by playing it over and over, getting better with practice. This is called **reinforcement learning**.

Specifically:

- The puzzle is shown to the neural network as a **graph** — a kind of map showing how all the pieces and rules connect to each other.
- The network looks at this graph and predicts which guess is likely to be good.
- It tries a guess, sees what happens, and gets a "reward" — a score telling it whether that was a smart move or not.
- Over thousands of attempts, it gradually learns a smarter guessing strategy than the old hand-crafted rule.

## Our twist

Most prior work simply rewards the network for solving puzzles in fewer steps. We go further: we reward it based on **how useful the information it picks up along the way actually is** — not just how fast it finishes, but how *smart* each step was. This is measured using a score called **LBD**, commonly used by solver experts to judge the quality of what the solver has learned mid-search.

The idea: if the network gets credit for genuinely useful guesses (not just speed), it might learn a better overall strategy.

## Why this matters

If this works well, it points toward a future where computers don't just follow rules written by humans — they discover their own strategies for solving hard logic problems, potentially better than what any person has hand-designed.

## Inspired by

This project builds on and extends ideas from **Graph-Q-SAT** (Kurin, Godil, Whiteson & Catanzaro, NeurIPS 2020), a research paper that first showed a neural network could learn this kind of guessing strategy from scratch.
