"""All preprocessing constants. PIL HSV uses 0..255 for every channel (hue 255 = 360 degrees)."""

# ---- Decode -------------------------------------------------------------------------------
MAX_PIXELS = 120_000_000  # decode guard (decompression bombs); ~12k x 10k
WORK_MAX_SIDE = 2048  # working copy long side; leaf search and crops use this copy

# ---- Classifier input ---------------------------------------------------------------------
INFERENCE_SIZE = 256  # = shortest_edge in the model's preprocessor_config.json
INFERENCE_JPEG_QUALITY = 95
PAD_COLOR = (128, 128, 128)
LEAF_FILL = 0.9  # leaf side / crop side, see leaf.square_window

# ---- Leaf finder (vegetation mask) --------------------------------------------------------
MASK_LONG_SIDE = 512  # leaf mask is computed on a copy this size
MIN_COMPONENT_FRAC = 0.015  # ignore green specks smaller than 1.5% of the frame
MAX_CANDIDATES = 6
SHARPNESS_NORM_SIDE = 256  # leaf crops are compared at this long side (downscale only)
REBLUR_SIGMA = 1.0  # extra Gaussian blur used by the re-blur sharpness ratio
FOCUS_WINDOW_FRAC = 0.03  # local sharpness is averaged over ~3% of the frame
FOCUS_KEEP_RATIO = 0.3  # keep vegetation at least 30% as sharp as the sharpest 10% of it
# Hue 25..120 is ~35..170 degrees: yellow-green to green-cyan. Brown/red soil, pots and skin
# fall outside; brown spots inside a leaf are recovered by closing + hole filling.
VEG_HUE_MIN, VEG_HUE_MAX = 25, 120
VEG_SAT_MIN, VEG_VAL_MIN = 50, 40

# ---- Symptom score (share of the leaf mask that is not healthy green) ---------------------
HEALTHY_HUE_MIN, HEALTHY_HUE_MAX = 38, 120  # ~54..170 degrees
HEALTHY_SAT_MIN, HEALTHY_VAL_MIN = 45, 50
WARM_HUE_WRAP = 230  # hue >= this (~325 degrees) is red/magenta, treated like warm hues
YELLOW_HUE_MIN = 25  # warm hue >= this and bright -> yellow (chlorosis)
YELLOW_SAT_MIN, YELLOW_VAL_MIN = 60, 110
BROWN_SAT_MIN = 40  # remaining warm, saturated pixels -> brown (necrosis, rust, blight)
DARK_VAL_MAX = 50  # value below this -> dark (black rot, sooty lesions)
PALE_SAT_MAX = 45  # low saturation, not dark -> pale or grey (mildew, mosaic bleaching)
# A symptom region must touch healthy green pixels of the same leaf and be at least this big
# (both at MASK_LONG_SIDE scale), so specks and enclosed background are not counted.
SYMPTOM_MIN_REGION_PX = 6
SYMPTOM_MIN_REGION_FRAC = 0.002  # of the leaf's pixels
SYMPTOM_RANK_FLOOR = 0.05  # strategy D ranks by (floor + symptom_score) * sharpness
SYMPTOM_PATCH_MIN_FRAC = 0.2  # strategy E: patch side >= this share of the leaf's long side
SYMPTOM_PATCH_FILL = 0.6  # strategy E: lesion side / patch side

# ---- Quality gate -------------------------------------------------------------------------
MIN_WIDTH = 128
MIN_HEIGHT = 128
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 230.0
# Leaf sharpness, both measured on the leaf at <=256px long side. Calibrated on a blur sweep of
# the reference photo at 1 MP and 12 MP and on 40 PlantVillage test images incl. smooth
# citrus/corn leaves; see docs/challenges-and-approach.md.
MIN_SHARPNESS_RATIO = 0.78  # re-blur ratio: moderate blur lands at 0.70-0.77, sharp at >=0.85
MIN_LAPLACIAN_VARIANCE = 25.0  # heavy blur (noise only) lands at 6-23, smooth sharp leaves >=37
MIN_LEAF_COVERAGE = 0.04  # leaf area / frame area below this is "too far away" ...
MIN_LEAF_NATIVE_PX = 128  # ... if the leaf is also this small in the original photo

# ---- Decision -----------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5
MIN_AGREEMENT = 2 / 3  # strategy D: share of crops whose top-1 equals the averaged top-1
