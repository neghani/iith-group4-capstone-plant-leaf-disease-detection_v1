"""Compare the farmer's crop hint with what the classifier can actually recognise.

The classifier only knows the crops in `data/knowledge/diseases.json` (the 14 PlantVillage
crops). A hint outside that list (e.g. hibiscus) means any "disease" it returns is a guess
about a plant it has never seen, and a hint that differs from the predicted crop means one of
the two is wrong. Both must change how the result is presented.
"""

import re
from functools import lru_cache

from app.services.knowledge import list_disease_ids
from app.vision.predictor import crop_from_class

_ALIASES = {"maize": "corn", "capsicum": "pepper", "sweet": "pepper", "bell": "pepper"}
_NOISE = {"including", "sour", "the", "a", "leaf", "leaves", "plant"}


def _singular(word: str) -> str:
    if word.endswith("oes"):  # tomatoes, potatoes
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]  # apples, grapes
    return word


def _tokens(text: str) -> set[str]:
    words = (_singular(w) for w in re.findall(r"[a-z]+", text.lower()))
    return {_ALIASES.get(w, w) for w in words if w not in _NOISE}


@lru_cache
def supported_crops() -> tuple[str, ...]:
    return tuple(sorted({crop_from_class(d) for d in list_disease_ids()}))


def resolve_hint(hint: str) -> str | None:
    """Canonical supported crop for a free-text hint, or None if the model does not cover it."""
    wanted = _tokens(hint)
    if not wanted:
        return None
    for crop in supported_crops():
        if wanted & _tokens(crop):
            return crop
    return None


def same_crop(a: str, b: str) -> bool:
    return bool(_tokens(a) & _tokens(b))
