# Presentation Fixtures

Curated, version-controlled copies of generated development-run responses that
the thesis presentation layer and its figures depend on. These files are
evidence for the *presentation transformation* (a frontend/UX concern); they
are not part of the scientific model comparison and change nothing about the
frozen recognition configuration.

## dev_clean_01_basic_pitch_response.json

- Byte-identical copy of
  `evaluation_results/chords/20260806T054938Z_development/per_clip/dev_clean_01/basic_pitch_response.json`
  (the frozen-configuration development run; `evaluation_results/` itself is
  excluded from version control).
- SHA-256:
  `9057daca93ad5f1f8803f4afe468c24b78f4bbe2c8c72eb1030da72a4831be8d`
- Parity-verified on 2026-08-10: the live FastAPI endpoint
  (`/analyze-file?method=basic_pitch`, configuration
  `development_8clip_grid_20260806`, CoreML runtime) reproduced this response
  exactly — identical segments, progression, note-event count (43) and
  analysis provenance.

Used by:

- the thesis manuscript figure script
  (`manuscript/figures/make_figures.py` — raw prediction vs presentation
  summary timeline for `dev_clean_01`);
- the Flutter frontend presentation-filter tests
  (`test/fixtures/dev_clean_01_basic_pitch_response.json` in the frontend
  repository is a copy of this file).
