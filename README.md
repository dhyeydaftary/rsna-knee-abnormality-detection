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

- Task: predict the per-study probability of twelve clinically important
  knee abnormality findings, one row per `StudyInstanceUID`.
- Targets (verbatim, in submission-file order): ACL, MCL, Medial Meniscus,
  Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis,
  Baker's, Contusion, Fracture.
- Metric: macro-averaged ROC-AUC across the twelve targets —
  Final Score = (1/12) * sum(AUC_i).
- Submission file: CSV with header, `StudyInstanceUID` + 12 confidence-score
  columns (values in [0, 1]); must be named `submission.csv`.
- **Training is multimodal (image + report text); inference is image-only.**
  The `Report` field is explicitly withheld at test time, so any use of
  report text is limited to training-time label derivation/weak
  supervision — it cannot be a runtime model input.

## Dataset Structure (confirmed from the official data description, 27 Aug 2026)

- **`train.csv`** — one row per study: `StudyInstanceUID`, `Report`
  (free-text radiology report, multiple languages depending on reporting
  institution), and the 12 binary (0/1) labels.
- **`train_series.csv`** — one row per series: `StudyInstanceUID`,
  `SeriesInstanceUID`, `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1, not
  always equivalent to `Fluid_Sensitive`), `Anatomical_Plane` (Sagittal /
  Coronal / Axial).
- **`train_series/`** — DICOMs at
  `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`,
  one slice per file. Series typically have 20–45 slices (median 30), long
  tail out to a few hundred.
- **`test.csv`** — ~1,300 studies at scoring time; example file has 3.
  `StudyInstanceUID` only — **no `Report` field.**
- **`test_series.csv` / `test_series/`** — same schema as train, swapped
  for real data during scoring.
- **`sample_submission.csv`** — all label columns set to 0.5. Verified
  against an actual copy: 13 columns, header and order match exactly,
  `StudyInstanceUID` values are genuine DICOM UIDs.
- **Label coverage:** only a small subset of training studies carry direct
  per-condition labels; the rest have only a report, from which labels may
  need to be derived. This is a weak-supervision problem, not plain
  supervised multilabel classification. Exact counts still to be measured
  from our own copy of `train.csv`.
- **Distribution shift risk:** abnormality prevalence is not guaranteed to
  be the same across train, public leaderboard, and final evaluation sets —
  relevant to validation-strategy design.
- **DICOM notes:** intensities, orientations, and resolutions vary across
  series/studies. Transfer syntaxes vary (uncompressed Explicit VR LE, JPEG
  Lossless, JPEG 2000, Implicit VR LE) — the DICOM reader needs
  `pylibjpeg`/`gdcm`-level decompression support, not just plain `pydicom`.
  Metadata has been stripped to an allowlisted set of 86 tags.

**Open sub-question on task framing:** three targets (Medial OA, Lateral OA,
PF OA) are laterality-specific by name; the other nine are not, and there is
no laterality field in `train_series.csv`. The absence of a laterality
column is circumstantial evidence that each study represents a single knee,
but this has not been explicitly confirmed — check by inspecting a handful
of actual studies.

## Rules and Constraints (confirmed, 27 Aug 2026)

- **Timeline:** start Jul 30, 2026 · entry/team-merge deadline Oct 15, 2026
  · final submission deadline Oct 22, 2026 · winners' requirement deadline
  Nov 5, 2026. All 11:59 PM UTC; organizers may adjust.
- **Code Requirements (submissions are code competitions):**
  - Submitted via Notebook, ≤9 hours runtime (CPU or GPU).
  - Internet access disabled at submission time.
  - Freely & publicly available external data and pretrained models are
    allowed (must meet a "Reasonableness Standard" — equally accessible to
    all participants, minimal/no cost).
  - Output must be named `submission.csv`.
- **Team & submission limits:** max team size 5; max 5 submissions/day; up
  to 2 Final Submissions selected for judging.
- **Data Security (Official Rules, Section 2.4.b) — important, affects
  architecture:** Competition Data (including report text) must not be
  transmitted or made available to any party not participating in the
  competition. This plausibly prohibits sending report text to third-party
  hosted LLM APIs (OpenAI, Anthropic, Google, etc.), since the API provider
  is not a competition participant. **Working assumption: do not send
  report text to hosted LLM APIs; use local/open-weight models (run
  in-notebook or self-hosted) for any report-derived label or weak-
  supervision work instead.** This has not been explicitly ruled on by the
  host — revisit if clarified via the discussion forum.
- **Licensing (two separate things — don't conflate):**
  - Competition *data* may be used for commercial or non-commercial
    purposes, including academic research, subject to RSNA's MIRA license.
  - If you place as a winner, your *solution's* code/model must be released
    under CC-BY-NC 4.0 (non-commercial), plus full training/inference code,
    weights (as a public dataset), and reproducibility documentation.
- **Code sharing:** no private sharing of competition code outside your
  team; public sharing only via Kaggle's own forums/notebooks.
- **Prizes:** Main Leaderboard — 10 places, $5,000–$9,000 each. Efficiency
  Track — 3 places, $5,000–$7,000 each. A submission can win both tracks.
- **Efficiency Prize eligibility:** must be a selected (or auto-selected)
  submission, and must beat the `sample_submission.csv` benchmark on the
  Private Leaderboard.
- **Efficiency score** (lower is better):
  `Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400`
  where `Benchmark` = sample_submission.csv's AUC, `max AUC` = best Private
  LB AUC across all submissions, `RuntimeSeconds` = this submission's own
  eval runtime, `32400` = 9h runtime cap in seconds. A public Efficiency
  Leaderboard (rank only, updated daily on a public notebook) is visible
  during the competition; full scores appear on the private leaderboard
  after the competition ends.
- **Winners' Obligations (beyond standard terms):** a short video of the
  approach, publishing code + model weights publicly on the competition
  forum, and making the final model publicly available for open
  distribution/validation. Relevant to keep in mind for code hygiene from
  the start (no leaked credentials/PII, license-compatible dependencies),
  in case of placing.

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
  notebook, which acts mainly as an execution layer. Final submission
  itself must run as a code-competition Notebook per the Code Requirements
  above (≤9h, no internet).
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

Project setup stage. No modeling has started. Task definition, evaluation
metric, submission format, dataset structure, timeline, code requirements,
and efficiency-prize criteria are all confirmed (see above). See
`notebooks/01_competition_analysis.ipynb` for the full reconnaissance and
its remaining open questions.

## Open Questions (must resolve before Stage 3 / baseline work)

- [x] Exact evaluation metric definition
- [x] Task definition and submission format
- [x] Dataset file structure (studies/series/DICOM hierarchy, metadata
      fields)
- [x] Label source and coverage (confirmed as weak-supervision shaped;
      exact counts still to be measured)
- [x] Whether report text is available at inference — confirmed: no,
      training-time only
- [x] Final-submission deadline and rules (code-competition constraints,
      internet access, runtime limits, external pretrained model policy)
- [x] Efficiency-prize evaluation criteria
- [x] Data-security constraints on report text — confirmed present (Section
      2.4.b), interpretation not host-clarified; treating conservatively
      (no hosted LLM APIs on report text)
- [ ] Whether a study = one knee (circumstantial evidence, not explicit)
- [ ] Whether patient- or site-level leakage is possible and how the
      official train/test split is structured

These should be answered directly from the official competition pages and
the actual downloaded data — not from secondary sources.