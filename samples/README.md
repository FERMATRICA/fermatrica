# FERMATRICA Samples

Working example models progressing from simple to complex. Each sample contains a complete model pipeline: data, configuration, transformation, optimization, fitting, and reporting.

## Overview

| # | Sample | Model Type | Target | Grouping | Period | Builds Upon |
|---|--------|-----------|--------|----------|--------|-------------|
| p00 | Time Series | OLS | units | — | weekly | — |
| p01 | Panel per Brand | LME | units | superbrand | weekly | p00 |
| p02 | Panel per SKU | LME | units | bs_key | weekly | p01 |
| p03 | Brand Awareness | LME | aba | superbrand | monthly | p01 |
| p04 | Panel per Region | LME | units | master | weekly | p01 |
| p05 | Two-Level Panel | LME × 2 | aba → units | superbrand → bs_key | monthly → weekly | p03, p02 |

## Progression

```
p00 (time series, OLS)
 └─► p01 (panel per brand, LME)
      ├─► p02 (panel per SKU, decomposition)
      ├─► p03 (panel per brand, ABA target)
      └─► p04 (panel per region, regional pricing)
            p03 + p02 ──► p05 (two-level: BHT → Sales)
```

## File Convention

Each sample has a `SPEC.md` with full specification. New samples should copy `SPEC_TEMPLATE.md` and fill in all sections.
