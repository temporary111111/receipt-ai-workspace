# Elective workspace

This repository coordinates the independent receipt-processing programs used
for Elective 3. Clone the program repositories separately, then use the
shared data location configured in `repos.yaml`.

The current pipeline is:

```text
raw receipts -> text detector/crops -> visual review -> Elective features -> CSV -> quality filter -> 900+900 subset
```

See `AGENTS.md` for handoff rules and `STATUS.md` for current progress.
