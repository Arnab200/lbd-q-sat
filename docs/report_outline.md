# LBD-Q-SAT: report outline

## 1. Introduction
- SAT and why branching decisions matter in CDCL solvers.
- Graph-Q-SAT: learning the branching rule with a graph neural network and deep Q-learning.
- Research question: does a reward that depends on learned clause quality (LBD) train a better branching agent than a flat per-decision penalty?
- Contributions: from-scratch reimplementation, the LBD reward, a controlled comparison, a NeuroSAT warm-up.
- Stated up front: the agent controls a Python CDCL solver, not MiniSat.

## 2. Background
- 2.1 SAT and CNF.
- 2.2 CDCL: unit propagation, conflict analysis, first-UIP clause learning, backjumping.
- 2.3 Branching heuristics: VSIDS, DLIS, random.
- 2.4 Literal block distance and glue clauses (Audemard and Simon, 2009).
- 2.5 Graph neural networks on the variable-clause graph; NeuroSAT.
- 2.6 Deep Q-learning: replay buffer, target network, epsilon-greedy.
- 2.7 Graph-Q-SAT.

## 3. Method
- 3.1 Data: SR(n) pairs labelled with MiniSat; training, validation and evaluation splits.
- 3.2 Environment: the Python CDCL solver, its `start` and `step` interface, what it leaves out (restarts, clause deletion).
- 3.3 State: graph of unassigned variables and open original clauses, edge sign feature.
- 3.4 Q-network: 4 rounds of message passing, hidden size 64, LayerNorm, two Q-values per variable.
- 3.5 Rewards: flat `-0.1`; LBD `-0.1 - 0.02 * sum(max(LBD - 2, 0))`.
- 3.6 Training: DQN hyperparameters, episode cap, model selection on validation decisions.
- 3.7 Differences from Graph-Q-SAT, in one table (solver, state, training budget, problem sizes).

## 4. Experiments
- 4.1 Warm-up: NeuroSAT accuracy, accuracy against message-passing rounds, generalization, decoding.
- 4.2 Solver correctness against MiniSat labels.
- 4.3 Baselines: VSIDS, DLIS, random on each evaluation set.
- 4.4 Training curves for both rewards.
- 4.5 Evaluation protocol: sets, decision cap of 5000, metrics (mean and median decisions, ratio to VSIDS, mean LBD, glue share, win rate, Wilcoxon signed-rank test).

## 5. Results
- 5.1 Do the agents learn? Validation decisions against the random and VSIDS references.
- 5.2 Flat against LBD reward: decisions, per-formula win rate, significance.
- 5.3 Both agents against VSIDS.
- 5.4 Learned clause quality: does the LBD reward actually lower LBD?
- 5.5 SAT against UNSAT formulas.
- 5.6 Generalization to 80 variables.

## 6. Limitations
- Python solver without restarts or clause deletion.
- State omits learned clauses, so it is not fully Markov.
- One seed per reward; differences may be within seed noise.
- The LBD reward also penalises the number of non-glue conflicts, not only their quality.
- Short training budget and small random formulas.
- Decisions, not wall-clock time.
- Runs that hit the decision cap bias the means; medians are the main metric.

## 7. Conclusion
- Answer to the research question, stated plainly whichever way it falls.
- Future work: several seeds, learned clauses in the state, restarts, reward weight sweep, MiniSat integration.

## References
Kurin et al. 2020; Selsam et al. 2019; Audemard and Simon 2009; Mnih et al. 2015; Eén and Sörensson 2003 (MiniSat); Moskewicz et al. 2001 (VSIDS).
