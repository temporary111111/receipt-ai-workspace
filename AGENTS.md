# Elective workspace handoff

## Scope

This workspace supports the Elective 3 machine-learning dataset preparation
for the thesis on AI-generated versus real text-containing receipt images.
It is an orchestration and documentation repository. Source code lives in the
independent repositories under `programs/`; generated data is kept outside
Git history.

## Labels

- `1_fake` = AI-generated, model label `1`
- `0_real` = non-AI, model label `0`

## Independent programs

- `programs/receipt_text_cropper`: projection-based cropper
- `programs/receipt_text_detector`: PaddleOCR detector and resumable crops
- `programs/receipt_elective_feature_extractor`: 14-feature Elective table
- `programs/receipt_feature_extractor`: thesis feature extractor; keep separate

## Current dataset plan

The working target is a curated balanced pool of 900 AI and 900 real receipts.
The detector can process all available source images first; quality filtering
comes before selecting the final 900+900 rows.

## Safety rules

- Do not commit raw images, generated crops, CSV outputs, Python environments,
  or model caches.
- Do not modify the thesis extractor when working on Elective features.
- Keep detector outputs resumable and preserve the visual audit artifacts.
- Read `STATUS.md` before continuing a long-running batch.
