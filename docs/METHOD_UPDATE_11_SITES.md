# Method update after adding 11 documented Maharashtra labyrinths

The project now supports two linked workflows.

## 1. Geometry-only Salem workflow

Use this before any classifier is trained.

```bash
python scripts/geometry/run_salem_geometry_only.py \
  --raster-path data/test/salem_cartosat3_pan.tif \
  --known-sites-csv data/sites/known_sites_salem.csv \
  --out-dir outputs/geometry_salem \
  --tile-sizes-m 16 32 64 \
  --overlap 0.5 \
  --top-k 200
```

## 2. Few-shot Maharashtra workflow

The 11 documented Maharashtra labyrinths are enough for a weak classifier, not a final detector.

Use group-based validation. Do not random split augmented chips.

Kalambi II, III, and IV must remain in the same validation group.

## Recommended claim

Use:

> A weakly supervised, few-shot, tile-based candidate discovery workflow for stone labyrinths in Indian archaeological landscapes.

Avoid:

> A fully supervised labyrinth detector.
