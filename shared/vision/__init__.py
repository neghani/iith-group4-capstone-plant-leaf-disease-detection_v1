"""Image preprocessing used by the live API and by every offline evaluation.

    leaf.py        vegetation mask, leaf regions, scale-normalised sharpness
    candidates.py  find_candidates(img): leaf candidates with leaf, symptom and sharpness scores
    quality.py     decode + the upload quality gate (the only copy of it)
    strategies.py  crop strategies A-E behind run_strategy(img, strategy_id, predictor)
    config.py      every threshold and HSV range
"""
