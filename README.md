[README_high_note.md](https://github.com/user-attachments/files/26226014/README_high_note.md)
# High Note: Driving Free-to-Premium Conversion

A causal analytics study on freemium music streaming conversion behavior.

---

## Overview

This project investigates what drives free users to upgrade to paid subscriptions on High Note, a freemium music streaming platform. Using a dataset of 43,827 users, we go beyond correlation to establish **causal evidence** that peer influence meaningfully increases conversion probability — and build a predictive model that identifies who is most likely to upgrade.

---

## Key Findings

- **Peer influence is causal:** Having at least one premium subscriber friend causes a **+9.12 percentage point** increase in conversion probability (95% CI: [8.18%, 10.06%], p < 0.001), after removing 3.39 pp of selection bias via Propensity Score Matching.
- **Social behavior predicts conversion more than demographics:** The social-only model (AUC ~0.77) outperforms demographics alone (AUC ~0.63).
- **Top predictors (full model):** subscriber friend count (OR = 1.82×), loved tracks (1.61×), shouts (1.44×), songs listened (1.38×).
- **Full model performance:** AUC = 0.812, McFadden pseudo R² = 0.198.

---

## Methods

| Stage | Technique |
|---|---|
| Descriptive analysis | Welch's two-sample t-tests |
| Behavioral framework | Participation Ladder (passive → social) |
| Causal inference | Propensity Score Matching (PSM), nearest-neighbor 1:1 |
| Predictive modeling | Logistic regression (5 model specifications) |

---

## Tech Stack

Python · pandas · statsmodels · scikit-learn · seaborn · matplotlib

---

## Repository Structure

```
├── HighNote_Full_Project.py       # Full analysis pipeline
├── High_Note_Paper.pdf            # Written report
└── index.html                     # GitHub Pages project site
```

---
