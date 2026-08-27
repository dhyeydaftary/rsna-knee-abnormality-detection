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

## Dataset Structure (confirmed from official documentation, verified against actual data via `notebooks/02_data_exploration.ipynb`, 27 Aug 2026)

- **`train.csv`** — 4,407 rows: `StudyInstanceUID`, `Report` (free-text
  radiology report, multiple languages depending on reporting institution),
  and the 12 binary (0/1) labels. Zero duplicate `StudyInstanceUID`.
- **`train_series.csv`** — 24,371 rows: `StudyInstanceUID`,
  `SeriesInstanceUID`, `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1),
  `Anatomical_Plane` (Sagittal / Coronal / Axial). Zero duplicate
  `SeriesInstanceUID`, zero orphan records against `train.csv` in either
  direction.
- **`train_series/`** — DICOMs at
  `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`,
  one slice per file. Series typically have 20–45 slices (median 30), long
  tail out to a few hundred. **Not yet inspected directly** — pending
  `03_dicom_exploration.ipynb`.
- **`test.csv`** — ~1,300 studies at scoring time; example file currently
  has 3, no `Report` field. Zero overlap with training `StudyInstanceUID`s.
- **`test_series.csv` / `test_series/`** — same schema as train (schema
  match confirmed), swapped for real data during scoring.
- **`sample_submission.csv`** — verified: 13 columns, header/order match
  exactly, all values 0.5, `StudyInstanceUID` values are genuine DICOM UIDs.
- **Label coverage — exact numbers confirmed:** 58 of 4,407 studies (1.3%)
  have all 12 labels populated; the remaining 4,349 (98.7%) have all 12
  labels null. **This split is entirely all-or-nothing — zero partially
  labeled studies exist.** This is a weak-supervision problem, not plain
  supervised multilabel classification.
- **Per-target prevalence within the 58-study labeled subset** (small-n,
  treat as a rough shape indicator, not a reliable population statistic):
  Effusion 60.3%, Synovitis 46.6%, Medial Meniscus 44.8%, ACL 41.4%, Lateral
  Meniscus 39.7%, PF OA 36.2%, Contusion 32.8%, Fracture 31.0%, Medial OA
  25.9%, Baker's 20.7%, Lateral OA 19.0%, MCL 15.5%. Median 4 positive
  findings per labeled study (range 1–9) — genuinely multilabel.
- **Series structure confirmed:** 3–14 series per study (median 5, mean
  5.53). All 4,407 training studies contain all three anatomical planes
  (Sagittal, Coronal, Axial) — zero exceptions.
- **`Fluid_Sensitive` / `Fat_Suppression` — documentation discrepancy
  noted:** the data description states these are "not necessarily
  equivalent for every case," but in this copy of the training data they
  are perfectly correlated across all 24,371 series (zero mixed cases).
  Not a contradiction — the documentation only says they're not
  *guaranteed* equivalent — but no divergent case has been observed yet.
  Keeping both columns; not dropping either.
- **131 exact-duplicate reports** identified in `train.csv`. Not yet
  investigated further; flagged as relevant to the future weak-label
  generation stage.
- **Distribution shift risk:** abnormality prevalence is not guaranteed to
  be the same across train, public leaderboard, and final evaluation sets —
  relevant to validation-strategy design.
- **DICOM notes (from documentation, not yet directly verified):**
  intensities, orientations, and resolutions vary across series/studies.
  Transfer syntaxes vary (uncompressed Explicit VR LE, JPEG Lossless, JPEG
  2000, Implicit VR LE) — the DICOM reader needs `pylibjpeg`/`gdcm`-level
  decompression support, not just plain `pydicom`. Metadata has been
  stripped to an allowlisted set of 86 tags.

**Open sub-question on task framing:** three targets (Medial OA, Lateral OA,
PF OA) are laterality-specific by name; the other nine are not, and there is
no laterality field in `train_series.csv`. Report-text spot-checks show
explicit single-knee language (e.g. "MRI of left Knee," German "Rechts" =
right) in a minority of reports, and only 15 of 4,407 reports mention both
"left" and "right" — weak circumstantial evidence toward "each study = one
knee," but not confirmed. Requires DICOM-level inspection.

## Validation Feasibility (confirmed via `notebooks/02_data_exploration.ipynb`, 27 Aug 2026)

**No patient, site, institution, or scanner identifier column exists in any
of the five tabular files** — confirmed by an exhaustive column-name search
across `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`, and
`sample_submission.csv`. `StudyInstanceUID` is the only available grouping
key in the tabular data. Whether the same patient can appear across
multiple studies cannot currently be determined.

**This is a hard dependency for `03_dicom_exploration.ipynb`, not an
optional nice-to-have.** If DICOM headers also carry no patient/site
identifier (plausible, given the 86-tag metadata allowlist), the project
will need to explicitly accept an unresolved leakage risk or fall back to
non-grouped validation — and that limitation should be stated openly in any
writeup rather than assumed away.

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

## Data Location (not committed — see `.gitignore`)

Competition CSVs live in `data/` at the repo root (`data/train.csv`,
`data/train_series.csv`, `data/test.csv`, `data/test_series.csv`,
`data/sample_submission.csv`). DICOM folders (`data/train_series/`,
`data/test_series/`) go in the same location once downloaded. `data/` is
gitignored and must stay that way — Section 2.4.b of the official rules
prohibits making Competition Data available to anyone not participating in
the competition, so it must never be committed to GitHub even in a private
repo.

## Current Status

`01_competition_analysis.ipynb` (documentation reconnaissance) and
`02_data_exploration.ipynb` (actual tabular data verification) are both
complete and executed against real data. No modeling has started, no `src/`
code exists yet. Next milestone: `03_dicom_exploration.ipynb`.

## Open Questions

- [x] Exact evaluation metric definition
- [x] Task definition and submission format
- [x] Dataset file structure (studies/series/DICOM hierarchy, metadata
      fields) — schema confirmed; actual DICOM files not yet inspected
- [x] Label source and coverage — **confirmed with exact numbers**: 58/4,407
      (1.3%) directly labeled, 4,349 report-only, zero partial
- [x] Whether report text is available at inference — confirmed: no,
      training-time only
- [x] Final-submission deadline and rules (code-competition constraints,
      internet access, runtime limits, external pretrained model policy)
- [x] Efficiency-prize evaluation criteria
- [x] Data-security constraints on report text — identified (Section
      2.4.b), interpretation not host-confirmed, treating conservatively
- [x] Tabular ID/join integrity — confirmed clean: zero duplicates, zero
      orphans, zero train/test overlap
- [x] Whether every study has all 3 anatomical planes — confirmed: yes,
      all 4,407 studies
- [ ] Whether a study = one knee — weak circumstantial evidence from report
      text (single-knee language, rare "both knees" mentions), not
      confirmed; requires DICOM inspection
- [ ] Whether patient- or site-level leakage is possible — **confirmed
      that no grouping variable exists in the tabular data**; whether one
      exists in DICOM headers is unresolved and is now a blocking
      dependency for validation-strategy design

These should be answered directly from the official competition pages and
the actual downloaded data — not from secondary sources.