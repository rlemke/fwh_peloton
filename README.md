# fwh_peloton

Split group cycling photos into **per-rider enhanced portraits**. A *peloton* is
the pack of riders — this finds each one and gives you a focused, sharpened
picture of them, then helps you size, organize, and cull the results.

```
group photo → detect riders (YOLO person+bicycle) → crop each (person ∪ bike)
            → [SAM cutout] → upscale (Real-ESRGAN) → face-restore (GFPGAN/CodeFormer)
            → [size to a standard 4×6/1:1/…] → out/<stem>_riderNN.jpg
```

A Facetwork domain package following the tools/handlers pattern. This ships the
reusable **library + CLI tools** (`src/peloton/tools/`); the FFL handlers/workflow
are the next phase.

**Input formats:** JPEG/PNG/WebP/TIFF/BMP, iPhone **HEIC/HEIF** (`pillow-heif`,
core), and camera **RAW** — `.nef` (Nikon), `.cr2/.cr3` (Canon), `.arw` (Sony),
`.dng`, `.raf`, `.orf`, `.rw2`, … (`rawpy`/LibRaw, `.[raw]` extra). A folder may
mix any of these — each file is decoded by its own format.

## Tools

| Tool | Does |
|------|------|
| `detect-riders`  | Detect each cyclist → JSON of person/bicycle boxes |
| `crop-riders`    | Detect + crop each rider (no enhance) |
| `enhance-image`  | Upscale + face-restore a single image |
| `process-photo`  | The pipeline: one photo → per-rider portraits (+ optional fixed size / SAM cutout) |
| `batch-photos`   | Run the pipeline over a whole directory (reuses models; running `manifest.json`) |
| `group-riders`   | Cluster the crops by face → one `rider_NNN/` folder per person, best-shot first |
| `cull-blurry`    | Score focus; move out-of-focus photos to `_unfocused/` |

Every tool: JSON on **stdout**, logs on **stderr**, `--use-mock` (offline, no
models), `--log-level`. Full reference + the graceful-degradation backends,
`--frame`/`--aspect`/`--size` framing, and `--segment` cutouts:
[`src/peloton/tools/README.md`](src/peloton/tools/README.md).

## Quick start

```bash
pip install -e '.[detect,enhance]'          # models; core alone runs degraded
# offline smoke (no models/network):
python src/peloton/tools/process_photo.py --image group.jpg --out-dir out/ --use-mock

# a folder → tight portrait + a print-ready 4×6 (1200×1800) per rider:
python src/peloton/tools/batch_photos.py --in-dir photos/ --out-dir out/ \
    --require-bike --frame both --size 1200x1800

# organize + cull:
python src/peloton/tools/group_riders.py --in-dir out/ --out-dir out/by_rider
python src/peloton/tools/cull_blurry.py  --in-dir photos/ --min-sharpness 900
```

## Extras (optional, lazy-imported — the pipeline degrades gracefully without them)

| Extra | Enables |
|-------|---------|
| `detect`    | YOLO rider detection + SAM cutouts (ultralytics/torch) |
| `enhance`   | Real-ESRGAN upscale + GFPGAN/CodeFormer face-restore (spandrel/gfpgan) |
| `recognize` | Same-rider grouping (insightface/onnxruntime) |
| `raw`       | Camera RAW decode (rawpy/LibRaw) |
| `s3`        | S3/MinIO storage (boto3) |

Model weights auto-cache to `~/.cache/peloton/weights`. Without a backend, upscale
falls back to Lanczos and face-restore to passthrough; each run records which
backend actually ran.

## Layout

```
src/peloton/tools/
  detect_riders  crop_riders  enhance_image  process_photo  batch_photos
  group_riders   cull_blurry                                  (+ .sh wrappers)
  _peloton_tools/  images crop detect segment enhance quality recognize
                   pipeline peloton_mocks sidecar storage
tests/             offline suite (29 tests, no network/models via --use-mock)
```

## Tests

```bash
pip install pytest pillow && pytest tests/ -q     # all offline
```

## Caveats

- **Face grouping** is imperfect on helmeted/sunglassed cyclists (weak embeddings);
  `--threshold` tunes precision/recall. Bib-number OCR would identify riders better.
- **NEF/RAW** decode path is wired + LibRaw-verified but not yet run on a real NEF.
