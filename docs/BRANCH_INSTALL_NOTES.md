# Branch install notes

## Create a new branch

```bash
git checkout -b geometry-fewshot-addon
```

## Unzip this package at the repository root

The package is organized to merge cleanly into your repo.

Expected result:

```text
Labyrinth_detector/
  scripts/
    geometry/
    fewshot/
    utils/
  data/
    sites/
  docs/
  requirements_addon.txt
```

## Add suggested gitignore entries

Open `.gitignore` and add the contents of `.gitignore_addon_suggested`.

Do not commit large rasters, outputs, or model checkpoints.

## Commit

```bash
git add scripts data/sites docs requirements_addon.txt .gitignore_addon_suggested
git commit -m "Add geometry-first and few-shot labyrinth workflows"
```
