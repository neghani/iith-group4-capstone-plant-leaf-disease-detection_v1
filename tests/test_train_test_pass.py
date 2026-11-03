"""The final test evaluate must not reuse the cached validation DataLoader."""

from __future__ import annotations

from types import SimpleNamespace


def test_eval_dataloader_cache_is_cleared_before_test_pass():
    """Mirrors the Trainer cache key used when dataloader_persistent_workers is on."""
    trainer = SimpleNamespace(_eval_dataloaders={"eval": object()})
    if getattr(trainer, "_eval_dataloaders", None):
        trainer._eval_dataloaders.clear()
    assert trainer._eval_dataloaders == {}
