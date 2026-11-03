"""PlantDoc class name -> model class name.

``MAP`` is the PlantVillage label set (38 classes). ``overlap_map`` keeps only the
PlantDoc classes that also exist in a label list, so the MasterDataset can be scored
on the shared classes without training on PlantDoc.

Used by build_real_world_set.py, prepare_plantdoc.py, and plantdoc_ood.py.
27 mapped classes; "Tomato two spotted spider mites leaf" is excluded (2 images).
"""

import re

# Names as in the PlantVillage `color` folders, so they match the model's id2label exactly.
MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

SKIP_CLASS = "Tomato two spotted spider mites leaf"

_TOKEN_NOISE = {"leaf", "leaves", "including", "sour", "maize", "the", "a", "and"}

# PlantVillage id -> MasterDataset folder, used only when that folder is in the label list.
# The MasterDataset spells classes as crop_disease or healthy_crop, not Crop___Disease.
MASTER_ALIASES = {
    "Apple___Apple_scab": "apple_apple_scab",
    "Apple___healthy": "healthy_apple",
    "Apple___Cedar_apple_rust": "apple_cedar_apple_rust",
    "Pepper,_bell___Bacterial_spot": "bell_pepper_bacterial_spot",
    "Pepper,_bell___healthy": "healthy_bell_pepper",
    "Cherry_(including_sour)___healthy": "healthy_cherry",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn_gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight": "corn_northern_leaf_blight",
    "Corn_(maize)___Common_rust_": "corn_common_rust",
    "Peach___healthy": "healthy_peach",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Strawberry___healthy": "healthy_strawberry",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot",
    "Tomato___healthy": "healthy_tomato",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus",
    "Grape___healthy": "healthy_grape",
    "Grape___Black_rot": "grape_black_rot",
}


def _tokens(name: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", name.lower().replace("___", " "))
    return {w for w in words if w not in _TOKEN_NOISE}


def overlap_map(label_names: list[str]) -> dict[str, str]:
    """PlantDoc folder -> label, only where that disease exists in ``label_names``.

    Exact PlantVillage ids win. Otherwise a label whose tokens cover the PlantVillage
    id is used, and only when that match is unique.
    """
    have = set(label_names)
    out: dict[str, str] = {}
    for plantdoc_name, plantvillage_name in MAP.items():
        if plantvillage_name in have:
            out[plantdoc_name] = plantvillage_name
            continue
        alias = MASTER_ALIASES.get(plantvillage_name)
        if alias and alias in have:
            out[plantdoc_name] = alias
            continue
        want = _tokens(plantvillage_name)
        matches = [label for label in label_names if want and want <= _tokens(label)]
        if len(matches) == 1:
            out[plantdoc_name] = matches[0]
    return out
