# LBD-Q-SAT

Teaching a SAT solver where to branch with reinforcement learning, and testing whether a reward based on learned clause quality helps.

Deep Learning course project, MSc Artificial Intelligence, University of Bologna.

## What this is

A SAT problem asks whether a set of true/false variables can be assigned so that every clause in a logical formula holds. Modern solvers use an algorithm called CDCL (conflict-driven clause learning). At each step the solver must pick a variable and a value to try next. This is the branching decision, and it is normally made by a hand-written rule such as VSIDS.

This project replaces that rule with a neural network trained by reinforcement learning. It is a from-scratch reimplementation of Graph-Q-SAT (Kurin et al., NeurIPS 2020) with one change to the reward.

- **Graph-Q-SAT (baseline):** the agent gets a small flat penalty for every decision, so it learns to finish in as few decisions as possible.
- **LBD-Q-SAT (this project):** the agent gets the same penalty plus an extra penalty that depends on the quality of the clauses the solver learns after each decision.

Clause quality is measured with the literal block distance (LBD): the number of different decision levels among the literals of a learned clause. Low LBD clauses, especially "glue" clauses with LBD of 2 or less, are known to be the most useful ones for the solver.

The network, the training loop and all hyperparameters are identical for the two rewards, so any difference in the results comes from the reward.

## How it works

1. **State.** The current solver state is turned into a graph. Unassigned variables and unsatisfied original clauses are the nodes. An edge joins a variable to each clause it appears in, with a feature saying whether it appears positive or negated.
2. **Network.** A graph neural network runs 4 rounds of message passing (hidden size 64) between variables and clauses, then outputs two Q-values per variable: one for setting it to false and one for setting it to true.
3. **Action.** The agent picks the literal with the highest Q-value (epsilon-greedy during training).
4. **Environment.** The solver applies the decision, runs unit propagation, resolves any conflicts with clause learning and backjumping, and reports the LBD of every clause it learned.
5. **Reward.**
   - Flat: `-0.1` per decision.
   - LBD: `-0.1 - 0.02 * sum(max(LBD - 2, 0))` over the clauses learned in that step. Glue clauses cost nothing extra.
6. **Training.** Deep Q-learning (DQN) with a replay buffer of 20000 transitions, a target network updated every 500 steps, Huber loss, epsilon from 1.0 to 0.05 over 10000 steps, discount 0.99, learning rate 1e-4, 1500 episodes per reward, at most 300 decisions per episode.

## An important simplification

Graph-Q-SAT controls MiniSat, a fast C++ solver. To keep this project feasible, the agent here controls a small CDCL solver written from scratch in Python (`lbdq/cdcl.py`). It has unit propagation, first-UIP clause learning, backjumping and VSIDS. It has no restarts and no clause deletion.

MiniSat (through PySAT) is only used to generate and label the data.

This means the results are about decision counts on small formulas inside a simple solver. They say nothing about wall-clock speed against a production solver.

## Repository layout

```
notebooks/      the eight notebooks, run in order
lbdq/           cdcl.py, qnet.py, neurosat.py (written by the notebooks, copied here for reading)
figures/        result plots
extras/         retired notebooks from an earlier direction, not part of the final project
```

| Notebook | What it does |
| --- | --- |
| `01_data_generation` | Generates SR(n) formula pairs (one satisfiable, one not, differing by one literal) and labels them with MiniSat |
| `02_model_and_training` | Warm-up: reimplements and trains NeuroSAT, a supervised satisfiability classifier |
| `03_evaluation` | Warm-up: NeuroSAT accuracy, generalization to larger formulas, decoding assignments |
| `04_cdcl_solver_environment` | The Python CDCL solver, checked against MiniSat labels |
| `05_branching_baselines` | Builds the reinforcement learning datasets and runs VSIDS, DLIS and random branching |
| `06_graph_q_network` | State graph, batching and the Q-network |
| `07_dqn_training` | Trains one agent with the flat reward and one with the LBD reward |
| `08_lbd_q_sat_evaluation` | Compares both agents with the baselines |

Notebooks 02 and 03 are a separate supervised warm-up and only depend on 01.

## How to run

The notebooks are written for Google Colab with Google Drive mounted.

1. Copy the `notebooks/` folder to your Drive.
2. Set `PROJECT_DIR` in the first cell of each notebook to your own Drive folder. All data, models and plots are saved there.
3. Run the main part in order: `01 -> 04 -> 05 -> 06 -> 07 -> 08`.
4. Optionally run the warm-up: `01 -> 02 -> 03`.

A GPU runtime is recommended for 02 and 07. To run outside Colab, install `requirements.txt` and start the notebooks from the folder where you want the files saved.

## Results

> Results are pending the final training run (notebooks 07 and 08). The tables and plots below will be filled in from `final_results.pkl`.

Evaluation sets: 400 test formulas with 10 to 40 variables, 100 formulas each with 40 and 60 variables, and 50 formulas with 80 variables for generalization. The agents are trained only on formulas with at most 20 variables.

**Median decisions per formula**

| Variables | VSIDS | DLIS | Random | Q flat | Q LBD |
| --- | --- | --- | --- | --- | --- |
| 10–40 | | | | | |
| 40 | | | | | |
| 60 | | | | | |
| 80 | | n/a | n/a | | |

**Learned clause quality (mean LBD, share of glue clauses)**

| Variables | VSIDS | Q flat | Q LBD |
| --- | --- | --- | --- |
| 10–40 | | | |
| 40 | | | |
| 60 | | | |
| 80 | | | |

Plots, once copied into `figures/`:

- `dqn_validation.png`: validation decisions during training, both rewards against VSIDS and random
- `final_decisions.png`: median decisions for every method and formula size
- `final_flat_vs_lbd.png`: flat reward against LBD reward, formula by formula
- `final_lbd_distribution.png`: LBD of learned clauses for VSIDS and both agents

## Limitations

- The solver is a simple Python CDCL without restarts or clause deletion, not MiniSat.
- The agent sees only the original clauses, not the learned ones.
- Each reward is trained once, with one random seed, for a short budget (1500 episodes).
- Formulas are small random SR(n) instances. Nothing here has been tested on structured or industrial problems.
- The comparison counts decisions. Each neural decision is far slower than a VSIDS decision.

## References

- V. Kurin, S. Godil, S. Whiteson, B. Catanzaro. *Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?* NeurIPS 2020.
- D. Selsam, M. Lamm, B. Bünz, P. Liang, L. de Moura, D. Dill. *Learning a SAT Solver from Single-Bit Supervision.* ICLR 2019.
- G. Audemard, L. Simon. *Predicting Learnt Clauses Quality in Modern SAT Solvers.* IJCAI 2009.
- V. Mnih et al. *Human-level control through deep reinforcement learning.* Nature, 2015.
