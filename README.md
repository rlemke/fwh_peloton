# fwh_peloton

Split group cycling photos into **per-rider enhanced portraits**. A *peloton* is
the pack of riders — this finds each one and gives you a focused, sharpened
picture of them.

```
group photo → detect riders (YOLO person+bicycle) → crop each (person ∪ bike)
            → upscale (Real-ESRGAN) → face-restore (GFPGAN/CodeFormer)
            → out/<stem>_rider00.jpg, _rider01.jpg, …
```

A Facetwork domain package following the tools/handlers pattern. This first cut
ships the reusable **library + CLI tools** (`src/peloton/tools/`); the FFL
handlers/workflow are the next phase.

## Quick start

```bash
pip install -e .                       # core: Pillow/numpy (pipeline runs, degraded)
python src/peloton/tools/process_photo.py \
    --image group.jpg --out-dir out/ --use-mock     # offline smoke, no models

# real models:
pip install -e '.[detect,enhance]'
python src/peloton/tools/process_photo.py --image group.jpg --out-dir out/ --require-bike
```

- **`--use-mock`** runs the whole pipeline with a deterministic offline detector
  (no torch, no downloads) — used by the test suite.
- Without the `detect` extra, real detection raises a clear "install ultralytics"
  error. Without `enhance`, upscaling falls back to Lanczos and face-restore to
  passthrough (each run reports which backend ran).

## Layout

```
src/peloton/tools/
  {detect,crop}_riders.py  enhance_image.py  process_photo.py   + .sh wrappers
  _peloton_tools/          images crop detect enhance pipeline peloton_mocks
                           sidecar storage
tests/                     offline suite (13 tests, no network/models)
```

Tools reference: [`src/peloton/tools/README.md`](src/peloton/tools/README.md).

## Tests

```bash
pip install pytest pillow && pytest tests/ -q     # all offline (--use-mock path)
```
