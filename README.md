# RSNA Knee Abnormality Detection

## Project Goal

Build a serious, reproducible ML/DL project for detecting twelve clinically
important knee abnormalities from multimodal knee MRI data (images plus
radiology report text) — not a copy-pasted public notebook. The purpose is
both to compete seriously and to produce a strong portfolio project covering
computer vision, medical imaging, and applied deep learning.

This project is deliberately scoped to ML/CV/medical-imaging work. It should
not accumulate unrelated infrastructure (web apps, chatbots, microservices)
unless there is a genuine technical reason.

## Task Summary (confirmed from the official competition overview, 27 Aug 2026)

- Task: multimodal (image + radiology report text) prediction of twelve
  clinically important knee abnormality targets, one row per
  `StudyInstanceUID`.
- Targets (verbatim, in submission-file order): ACL, MCL, Medial Meniscus,
  Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis,
  Baker's, Contusion, Fracture.
- Metric: macro-averaged ROC-AUC across the twelve targets —
  Final Score = (1/12) * sum(AUC_i).
- Submission file: CSV with header, `StudyInstanceUID` + 12 confidence-score
  columns (values in [0, 1]).
- Prize structure includes a separate efficiency track (own evaluation
  criteria, not yet pulled into this doc).
- Official framing: the dataset pairs every imaging study with its original
  radiology report, enabling models to learn from both visual scans and
  written diagnostic text. Goal is decision-support-grade accuracy,
  consistency, and speed, especially for settings with limited access to
  musculoskeletal radiologists.
- Timeline, exact rules, and dataset structure: still pending — see Open
  Questions.

**Open sub-question on task framing:** three targets (Medial OA, Lateral OA,
PF OA) are laterality-specific by name; the other nine (ACL, MCL, Medial
Meniscus, Lateral Meniscus, Effusion, Synovitis, Baker's, Contusion,
Fracture) are not. Need to confirm from the data description whether each
study represents a single knee (removing the ambiguity), or whether non-OA
findings are pooled across both knees within a study.

**Label coverage caveat (unverified):** early informal reports from other
participants suggest most training studies may only have a radiology report
(not a direct expert label), with a much smaller subset carrying
gold-standard expert annotations, and that report-derived labels do not
perfectly agree with expert labels. This has not been independently
verified for this project and must be checked directly against the actual
data before it drives any modeling decisions. If true, this is closer to a
weak-supervision problem than a plain multi-label image classification
problem.

## Development Philosophy

1. Understand the task and data before writing any modeling code.
2. Local validation strategy is the source of truth, not the public
   leaderboard.
3. Baselines first, complexity later, and only when justified by a stated
   hypothesis.
4. Core logic lives in `src/` (added once there is real logic to put there),
   notebooks orchestrate it rather than containing it.
5. Every meaningful experiment is recorded (config, seed, data split, score).

## Environments

- **Antigravity** — main development environment: architecture, source
  code, refactoring, tests, docs, git workflow. Notebooks (`.ipynb`) may
  also be created and maintained here for exploration — Antigravity is the
  environment, the notebook format itself is not exclusive to any one tool.
- **Notebooks** (`notebooks/`, `.ipynb`) — exploration and analysis only:
  DICOM/data inspection, MRI visualization, metadata/label analysis, data
  quality checks, small experiments, error analysis, result visualization.
  Notebooks orchestrate reusable code from `src/`; they do not contain the
  core implementation.
- **Hosted GPU notebooks** (platform-provided) — primary environment for
  large-scale GPU training, inference, and submission generation, since the
  data lives there and doesn't need to be downloaded locally. Repo code is
  committed to GitHub first, then imported or copied into the hosted
  notebook, which acts mainly as an execution layer.
- **Google Colab** — secondary GPU environment for prototyping and
  temporary experiments when the primary hosted GPUs aren't available. Not
  the primary home of the project; useful code moves back into the repo.
- **GitHub** — source of truth for code, config, and documentation.

## Git Workflow

- `main` — stable branch. Only updated from `dev`, and only after a real milestone (not after every merge).
- `dev` — integration branch. All work branches off `dev`.
- Work branches: `feature/...`, `chore/...`, `fix/...` or `bug/...`, `hotfix/...`, branched from `dev`, merged back into `dev`.
- Commit format: Conventional Commits, e.g. `feat(data): add initial DICOM metadata inspection`.

## Current Status

Project setup stage. No modeling has started. Evaluation metric, target
list, and submission format are confirmed (see above). See
`notebooks/01_competition_analysis.ipynb` for the full reconnaissance and
its remaining open questions.

## Open Questions (must resolve before Stage 2 / baseline work)

- [x] Exact evaluation metric definition — confirmed: macro ROC-AUC, equal
      weight across the 12 targets listed above.
- [x] Task definition and submission format — confirmed: one row per study,
      12 confidence scores, targets listed above.
- [ ] Exact final-submission deadline and rules (code-competition
      constraints, internet access, runtime limits, external pretrained
      model policy)
- [ ] Efficiency-prize evaluation criteria (separate from main track)
- [ ] Dataset file structure (studies/series/DICOM hierarchy, metadata
      fields)
- [ ] Whether a study = one knee, or can contain bilateral information for
      the non-OA targets
- [ ] Label source and coverage: how many studies have direct expert labels
      vs. report-derived labels only, and how reliable is the report-derived
      signal
- [ ] Whether patient- or site-level leakage is possible and how the
      official train/test split is structured

These should be answered directly from the official competition pages and
the actual downloaded data — not from secondary sources.