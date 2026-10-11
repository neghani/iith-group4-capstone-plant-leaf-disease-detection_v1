### About this set

Sampled (seed 42) from Cropped-PlantDoc (Singh et al., 2020, CC BY 4.0). The model was not trained
on any PlantDoc image. Rebuild the images with `real_world_set/build_real_world_set.py`.

- **Small.** 25 images per split gives a confidence interval of roughly ±14 points.
- **Already cropped around leaves.** Cropped-PlantDoc tests the classifier on field-style leaves,
  not the full phone-photo path.
- **Label mapping.** A few PlantDoc classes were matched to the closest PlantVillage class, and
  some web labels may be noisy. 11 PlantVillage classes have no PlantDoc images and are not tested.

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., Batra, N. (2020). PlantDoc: A Dataset for
Visual Plant Disease Detection. CoDS-COMAD 2020. https://doi.org/10.1145/3371158.3371196
