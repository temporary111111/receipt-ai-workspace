# Current status

Updated: 2026-09-29

## Detector

- CPU pilot: 300 AI + 300 real completed and validated.
- A GPU run encountered a cuDNN mismatch after receipts 301–448; those 148
  fake output folders were removed for regeneration.
- The isolated GPU environment was corrected to cuDNN 9.9 and passed a 1-AI
  plus 1-real smoke test on `gpu:0`.
- Full detector run is resumable and currently being continued with GPU.

## Features

- 14-feature Elective extractor validated on 600 receipts and 33,019 regions.
- Full feature extraction completed: 2,081 receipt rows and 113,817 region
  rows; technical validation passed.
- A candidate selection was created using top mean detector confidence within
  each class. The selected 900+900 table has 98,313 matching audit rows and
  passed technical validation.

## Target

- Curated final tentative dataset: 900 AI + 900 real.
- Available source images: 1,095 AI and 986 real.
- Selection outputs are under `data/receipt_dataset/features/elective3_full_pool/selection_v1`.
