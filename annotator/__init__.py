"""Offline annotation of voice_dictation sessions for fine-tuning.

See FINETUNE_PLAN.md (Phase B) for the design. This package loads a
finalized ``sessions/<ts>/`` directory and exposes a per-chunk view of
segments — splitting segments that straddle a chunk boundary on word
midpoints — for an HTTP-served annotation UI.
"""
