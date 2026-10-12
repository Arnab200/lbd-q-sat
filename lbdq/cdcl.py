import random


class Solver:
    def __init__(self, n_vars, clauses):
        self.n_vars = n_vars
        self.clauses = [list(c) for c in clauses]
        self.n_original = len(clauses)
        self.assignment = {}
        self.level = {}
        self.reason = {}
        self.trail = []
        self.trail_lim = []
        self.activity = {v: 0.0 for v in range(1, n_vars + 1)}
        self.bump = 1.0
        self.decisions = 0
        self.conflicts = 0
        self.lbds = []
        self.result = None

    def value(self, lit):
        v = abs(lit)
        if v not in self.assignment:
            return None
        return self.assignment[v] == (lit > 0)

    def assign(self, lit, reason):
        v = abs(lit)
        self.assignment[v] = lit > 0
        self.level[v] = len(self.trail_lim)
        self.reason[v] = reason
        self.trail.append(lit)

    def decide(self, lit):
        self.trail_lim.append(len(self.trail))
        self.assign(lit, None)

    def propagate(self):
        changed = True
        while changed:
            changed = False
            for clause in self.clauses:
                satisfied = False
                free = 0
                last_free = None
                for lit in clause:
                    val = self.value(lit)
                    if val is True:
                        satisfied = True
                        break
                    if val is None:
                        free += 1
                        last_free = lit
                if satisfied:
                    continue
                if free == 0:
                    return clause
                if free == 1:
                    self.assign(last_free, clause)
                    changed = True
        return None

    def analyze(self, conflict):
        current = len(self.trail_lim)
        seen = set()
        learned = []
        counter = 0
        clause = conflict
        index = len(self.trail) - 1

        while True:
            for lit in clause:
                v = abs(lit)
                if v in seen or self.level[v] == 0:
                    continue
                seen.add(v)
                if self.level[v] == current:
                    counter += 1
                else:
                    learned.append(lit)

            while abs(self.trail[index]) not in seen:
                index -= 1
            lit = self.trail[index]
            index -= 1
            counter -= 1

            if counter == 0:
                learned.append(-lit)
                break

            v = abs(lit)
            clause = [l for l in self.reason[v] if abs(l) != v]

        lower_levels = [self.level[abs(l)] for l in learned[:-1]]
        back_level = max(lower_levels) if lower_levels else 0
        lbd = len(set(self.level[abs(l)] for l in learned))
        return learned, back_level, lbd

    def backtrack(self, level):
        if len(self.trail_lim) <= level:
            return
        start = self.trail_lim[level]
        for lit in self.trail[start:]:
            v = abs(lit)
            del self.assignment[v]
            del self.level[v]
            del self.reason[v]
        self.trail = self.trail[:start]
        self.trail_lim = self.trail_lim[:level]

    def bump_activity(self, clause):
        for lit in clause:
            self.activity[abs(lit)] += self.bump
        self.bump /= 0.95
        if self.bump > 1e100:
            for v in self.activity:
                self.activity[v] *= 1e-100
            self.bump *= 1e-100

    def settle(self):
        while True:
            conflict = self.propagate()
            if conflict is None:
                if len(self.assignment) == self.n_vars:
                    self.result = True
                return
            self.conflicts += 1
            if len(self.trail_lim) == 0:
                self.result = False
                return
            learned, back_level, lbd = self.analyze(conflict)
            self.lbds.append(lbd)
            self.backtrack(back_level)
            self.clauses.append(learned)
            self.bump_activity(learned)

    def start(self):
        self.settle()
        return self.result

    def step(self, lit):
        n_before = len(self.lbds)
        self.decisions += 1
        self.decide(lit)
        self.settle()
        return self.lbds[n_before:]

    def unassigned(self):
        return [v for v in range(1, self.n_vars + 1) if v not in self.assignment]


def vsids_literal(solver):
    v = max(solver.unassigned(), key=lambda x: solver.activity[x])
    return -v


def random_literal(solver):
    v = random.choice(solver.unassigned())
    return v if random.random() < 0.5 else -v


def solve(n_vars, clauses, choose, max_decisions=100000):
    solver = Solver(n_vars, clauses)
    solver.start()
    while solver.result is None and solver.decisions < max_decisions:
        solver.step(choose(solver))
    return solver
