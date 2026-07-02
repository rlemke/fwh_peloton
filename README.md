# fwh_peloton

Split group cycling photos into **per-rider enhanced portraits**. A *peloton* is
the pack of riders — this finds each one and gives you a focused, sharpened
picture of them, then helps you size, organize, and cull the results.

```
group photo → detect riders (YOLO person+bicycle) → crop each (person ∪ bike)
            → dehaze + auto-brighten → keep NATIVE detail (upscale only when a crop
              is smaller than the target — a sharp high-res crop is left as-is;
              Real-ESRGAN would over-smooth it) → [SAM cutout] → [face-restore]
            → a tight _single (+ optional wider _context that also captures the
              surrounding riders) → out/<stem>_riderNN[.jpg | .tif]
```

Two enhancement fixes matter for high-resolution sources (e.g. 45 MP DSLR RAW):
- **Detail-preserving.** Each rider crop off a big frame is already high-res, so the
  Real-ESRGAN 4× upscaler (a *low-res* super-resolver that denoises) is **skipped**
  unless the crop is genuinely smaller than the target — running it then downscaling
  visibly softens a sharp crop. `--upscale-mode auto` (default) / `never` / `always`.
- **Dehaze.** These frames come off the camera flat (lifted blacks, muted colour); a
  per-image black/white-point stretch + mild contrast/colour removes the veil. On by
  default (`--no-dehaze` to disable).

**Output:** `--out-format jpg` (default, 8-bit) or **`tiff`** (lossless **16-bit** —
decodes RAW at 16-bit and does the tonal math in 16-bit so a heavy brighten/dehaze
stretch stays banding-free; the archival master, large files). Keep the TIFFs as
masters and derive shareable JPEGs on the side with **`tiffs-to-jpegs`** (a separate
`--in-dir`/`--out-dir` step, no re-processing).

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
| `tiffs-to-jpegs` | Derive shareable 8-bit JPEGs from a directory of 16-bit TIFF masters (separate step) |

Every tool: JSON on **stdout**, logs on **stderr**, `--use-mock` (offline, no
models), `--log-level`. Full reference + the graceful-degradation backends,
`--frame`/`--aspect`/`--size` framing, and `--segment` cutouts:
[`src/peloton/tools/README.md`](src/peloton/tools/README.md).

## Quick start

```bash
pip install -e '.[detect,enhance]'          # models; core alone runs degraded
# offline smoke (no models/network):
python src/peloton/tools/process_photo.py --image group.jpg --out-dir out/ --use-mock

# a folder → a tight _single + a wider _context per rider, cleaned up, native
# detail, lossless 16-bit TIFF (drop --out-format for 8-bit JPEG):
python src/peloton/tools/batch_photos.py --in-dir photos/ --out-dir out/ \
    --require-bike --context --auto-brighten --out-format tiff

# or a tight portrait + a print-ready 4×6 (1200×1800) per rider:
python src/peloton/tools/batch_photos.py --in-dir photos/ --out-dir out/ \
    --require-bike --frame both --size 1200x1800

# organize + cull:
python src/peloton/tools/group_riders.py --in-dir out/ --out-dir out/by_rider
python src/peloton/tools/cull_blurry.py  --in-dir photos/ --min-sharpness 900

# derive shareable JPEGs from the 16-bit TIFF masters (separate, idempotent step):
python src/peloton/tools/tiffs_to_jpegs.py --in-dir out/ --out-dir out_jpg/
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
  group_riders   cull_blurry  tiffs_to_jpegs                  (+ .sh wrappers)
  _peloton_tools/  images crop detect segment enhance quality recognize
                   pipeline peloton_mocks sidecar storage
tests/             offline suite (43 tests, no network/models via --use-mock)
```

## Tests

```bash
pip install pytest pillow && pytest tests/ -q     # all offline
```

## Caveats

- **Face grouping** is imperfect on helmeted/sunglassed cyclists (weak embeddings);
  `--threshold` tunes precision/recall. Bib-number OCR would identify riders better.
- **NEF/RAW** decode is validated end-to-end on real Nikon `.NEF` (8-bit and 16-bit),
  including a 1442-photo 45 MP run.
