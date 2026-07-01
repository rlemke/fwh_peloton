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

All tools: structured JSON on **stdout**, logs on **stderr**, `--use-mock` for an
offline deterministic run (no models), `--log-level`.

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

- **detect** — Ultralytics YOLO (`.[detect]`). `--use-mock` = deterministic offline boxes.
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
