# peloton tools — group cycling photo → per-rider enhanced portraits

Each stage is one `_peloton_tools` primitive + one CLI. Same code path from the
terminal and (later) from an FFL handler.

```
                 ┌─────────────┐   person + bicycle boxes
  group.jpg ───▶ │  detect     │ ─────────────────────────┐
                 │ (YOLO)      │   Rider(person, bike?)    │
                 └─────────────┘                           ▼
                                                    ┌─────────────┐
                                     per rider:     │  crop       │  person ∪ bike,
                                                    │ (pure geom) │  padded, clamped
                                                    └─────┬───────┘
                                                          ▼
                                                    ┌─────────────┐
                                                    │  upscale    │  Real-ESRGAN
                                                    │ (enhance)   │  → else Lanczos
                                                    └─────┬───────┘
                                                          ▼
                                                    ┌─────────────┐
                                                    │ face-restore│  GFPGAN/CodeFormer
                                                    │ (enhance)   │  → else passthrough
                                                    └─────┬───────┘
                                                          ▼
                                          out/<stem>_rider00.jpg, _rider01.jpg, …
```

## Tools

| Tool | Does |
|------|------|
| `detect-riders`  | Detect each cyclist → JSON of person/bicycle boxes |
| `crop-riders`    | Detect + crop each rider to its own image (no enhance) |
| `enhance-image`  | Upscale + face-restore a single image |
| `process-photo`  | The whole pipeline: photo → per-rider enhanced portraits |
| `batch-photos`   | Run the pipeline over a whole directory (model reuse) |
| `group-riders`   | Cluster the per-rider crops by face → one `rider_NNN/` folder per person, best-shot first |

### Organize & identify (`group-riders`)

Points at the crops from `batch-photos` and clusters them by face (InsightFace/
ArcFace embeddings + cosine clustering) into one folder per rider, ranked
best-shot first by a model-free quality score (`quality.py`: sharpness + exposure).
Crops with no detectable face → `_no_face/`.

> **Caveat — cyclists are hard for face recognition.** Helmets + sunglasses + distance
> give weak, noisy embeddings, so grouping is a useful-but-imperfect first pass: it
> reliably merges clear same-rider pairs but can over/under-merge ambiguous ones, and
> ~7% of crops have no detectable face at all. `--threshold` tunes precision/recall
> (default 0.5; 0.4 over-merges, 0.6 is stricter). **Bib-number OCR would identify
> riders far more reliably** — a natural future addition.

All tools: structured JSON on **stdout**, logs on **stderr**, `--use-mock` for an
offline deterministic run (no models), `--log-level`.

### Fixed output size (`--frame`)

By default each output is a tight crop in the rider's own proportions. To get a
**standard photo size**, `--aspect W:H` (e.g. `4:5`) or `--size WxH` (e.g.
`1080x1350`, exact pixels) expands the crop **outward** — real surrounding pixels
(possibly other riders), never distortion — to hit the target, padding
(`--pad-color name|#hex|blur`) only if the photo edge is reached. `--frame`
chooses `single` (tight), `framed` (fixed size), or `both` (writes `_single` +
`_framed`). `--aspect`/`--size` default `--frame` to `framed`.

```bash
process_photo.py --image g.jpg --out-dir out/ --frame both --size 1080x1350
```

```bash
# offline (no models) — verifies the whole path:
python process_photo.py --image group.jpg --out-dir out/ --use-mock

# real detection + enhancement (install the extras first):
pip install '.[detect,enhance]'
python process_photo.py --image group.jpg --out-dir out/ --require-bike --scale 4
```

## Backends (graceful degradation)

The pipeline **always** produces output; ML backends are best-effort, and each
run records which one actually ran (`upscale_backend`, `face_backend`):

- **detect** — Ultralytics YOLO (`.[detect]`), default `yolo11x.pt` (x-large). The
  nano model misses many bicycles → `--require-bike` drops real riders; on a
  23-photo sample yolo11x recovered 35 riders-with-bike that nano missed (0). Use
  `--model yolo11n.pt`/`yolov8n.pt` for a faster, lower-recall run. `--use-mock` =
  deterministic offline boxes.
- **segment** (`--segment`) — box-prompted **SAM** (`mobile_sam.pt`, via ultralytics)
  cuts each rider out of the background. `--cutout-bg white|black|blur|transparent`
  (transparent → PNG with alpha). Falls back to a filled box if SAM finds nothing.
- **upscale** — **Real-ESRGAN** via `spandrel` (plain torch, tiled, MPS/CUDA/CPU) →
  `realesrgan-ncnn-vulkan` binary → **Lanczos** (always).
- **face-restore** — `--face-backend gfpgan|codeformer` (both via facexlib align/paste;
  CodeFormer's `--fidelity` = its `weight`, higher = more faithful) → **passthrough**.

Weights auto-cache to `~/.cache/peloton/weights`. `pip install '.[detect,enhance]'`.

## Library (`_peloton_tools/`)

`images` (load/save) · `detect` (YOLO + `Rider`) · `crop` (pure box geometry) ·
`enhance` (upscale + face-restore) · `pipeline` (orchestration) ·
`peloton_mocks` (offline detector) · `sidecar`/`storage` (cache primitives).

Contract: [`agent-spec/tools-pattern.agent-spec.yaml`](../../../agent-spec/tools-pattern.agent-spec.yaml).

## Next phase (→ workflow)

`handlers/` + `ffl/` + the `facetwork.domains` entry point turn these primitives
into an FFL workflow — a `foreach` fan-out over riders (detect once, then
crop→upscale→restore each rider in parallel), outputs finalized to MinIO via
`storage`. Not wired yet; the tools above are the reusable substrate.
