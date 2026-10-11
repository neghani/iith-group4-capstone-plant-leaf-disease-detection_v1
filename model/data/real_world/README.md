# Real-world test images

Images collected from the internet (as agreed with the coordinator) to check how the
PlantVillage-trained model performs on photos that do not come from the dataset.
Proposal target: **at least 50% of these images predicted correctly** in the demo.

## Layout

One folder per class, using the **exact PlantVillage class name** (the same names the
model emits and that appear as keys in `api/data/knowledge/diseases.json`):

```text
model/data/real_world/
├── Tomato___Early_blight/
│   ├── 01.jpg
│   └── 02.jpg
├── Tomato___Late_blight/
├── Tomato___healthy/
├── Potato___Early_blight/
├── Apple___Apple_scab/
├── Corn_(maize)___Common_rust_/
└── sources.csv
```

Run `python model/real_world_eval.py --list-classes` to print the class names the trained
model knows. Folders whose name is not one of those classes are skipped with a warning.

## Collection rules

- Cover the same crops and diseases the model predicts; start with the tomato classes.
- Aim for at least 5 images per class and at least 30 images overall.
- Prefer field photos taken with a phone: varied lighting, angles, backgrounds, partial
  leaves. The point is to be harder than PlantVillage, not to copy it.
- Do not reuse PlantVillage images or obvious crops of them.
- Record where each image came from in `sources.csv`:

```csv
file,class,source_url,notes
Tomato___Early_blight/01.jpg,Tomato___Early_blight,https://example.org/page,extension service photo
```

- Image files are **git-ignored** (`.gitignore`). Commit only this README and
  `sources.csv`. For Colab, zip the class folders once and put the zip at
  `MyDrive/plant-disease/real_world.zip`. `notebooks/colab_sync.py --real-world-from`
  always deletes existing class folders and copies from that zip (do not
  `files.upload()` each run).

## Evaluate

```bash
python model/real_world_eval.py --model neghani/plantvillage-mobilenetv2 --revision v1.0
```

Prints per-image top-3 predictions, top-1 and top-3 accuracy, pass/fail against the 50%
target, and appends a "Real-world results" section to the model card.
