/** Canonical crop names for the live PlantVillage pin (38 classes → 14 crops).
 *  Must token-match `crop_from_class` / `api/data/knowledge/diseases.plantvillage.json`
 *  via `api/app/services/crops.py` (`resolve_hint` / `same_crop`).
 */
export const MASTER_CROPS = [
  "Apple",
  "Blueberry",
  "Cherry",
  "Corn",
  "Grape",
  "Orange",
  "Peach",
  "Bell Pepper",
  "Potato",
  "Raspberry",
  "Soybean",
  "Squash",
  "Strawberry",
  "Tomato",
] as const;

export type MasterCrop = (typeof MASTER_CROPS)[number];

export const OTHER_CROP = "__other__";
