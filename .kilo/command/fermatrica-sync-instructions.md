---
description: Review and sync AGENTS.md with current project structure
---

# Sync FERMATRICA Instructions

Review AGENTS.md and `.kilo/command/*.md` against the actual project structure and code conventions. Detect and fix any drift.

## Checklist for this review

1. **Directory structure** — read `fermatrica/` recursively. Does the module tree description in AGENTS.md match reality? Are there new modules or deleted modules?
2. **Pipeline steps** — read `fermatrica/model/predict.py`, `fermatrica/model/transform.py`. Is the canonical 6-step pipeline still accurate?
3. **Public API** — read `fermatrica/__init__.py`. Are all public symbols still present, or were any added/removed?
4. **model_conf.xlsx format** — read `fermatrica/model/model_conf.py`. Are all sheets/columns still the same?
5. **Optimizers** — read `fermatrica/optim/locals.py`, `globals.py`, `globals_deap.py`. Are the optimizer function signatures still correct?
6. **Transformation functions** — read `fermatrica/model/transform_fun.py`. Are all listed functions (`adstock`, `softmax`, `lag`, etc.) still present?
7. **Tests** — read `tests/` listing. Are the test module descriptions and what-to-test tables up to date?
8. **Sample structure** — if an external samples directory is configured, read one or two samples. Is the described sample structure accurate?
9. **Conventions** — read sample code (e.g., `code_py/scripts/run_test.py`). Are the import ordering, error handling, and type annotation descriptions still correct?
10. **Command files** — read `.kilo/command/` files. Do they still reference correct paths, function signatures, and workflow steps?

For each discrepancy found, update the relevant file. Do not delete information about deprecated features — instead, add a note marking them as deprecated and noting what replaced them.
