# AGENTS.md — Image Utility Processing Pipeline

## Project Overview

**Image Utility** is a modular ecommerce product image processing pipeline for transforming manually captured photos into studio-like, ecommerce-ready assets. Designed for automotive/motorcycle/metallic products with target platforms including Wix storefronts.

**Philosophy**: Phase-based, composable, replaceable architecture with hard separation of concerns. Each phase has a single responsibility and can be tested/tuned independently.

---

## Architecture: Phase-Driven Pipeline (v2.0)

### Pre-Pipeline + Five Sequential Phases

| Phase | Responsibility | Input | Output | Key Rule |
|-------|------------------|-------|--------|----------|
| **classify** | Scene classification (white bg / hand / clean) | RGB image | SceneClassification | No image modification, < 100ms |
| **isolate** | Segment product from background (with pre-isolation inpainting) | RGB image | RGBA + alpha mask | No resize, no color alteration |
| **compose** | Place isolated product on white 2000×2000 canvas | RGBA | Composed RGB on white BG | No product appearance changes |
| **shadow** | Add subtle grounding shadow | Composed RGB | RGB with shadow | Subtle only, no dramatic shadows |
| **polish** | Selective contrast/sharpness enhancement (+ resize in enhance-only) | RGB | Enhanced RGB | Subtle enhancement, no HDR-like effects |
| **compress** | Export multi-format assets (WebP, JPEG, thumbnails) | Final RGB | WebP/JPEG files | Compression only here, no quality loss |

### Pipeline Paths (v2.0)

| Path | Triggered By | Phases |
|------|-------------|--------|
| **Full Pipeline** | `clean_product` or `hand_held` | classify → isolate → compose → shadow → polish → compress |
| **Enhance-Only** | `already_white` | classify → polish (with resize) → compress |

**Hard Rules**:
1. Phases MUST remain independent & reorderable
2. Compression happens ONLY in compress phase
3. Segmentation (rembg/SAM) is replaceable without affecting downstream phases
4. Context mutations propagate forward (immutable per-phase outputs NOT required)
5. Scene classification MUST execute before pipeline routing (v2.0)
6. Hand inpainting MUST happen BEFORE segmentation, never after (v2.0)
7. Only remove what is confidently hand/skin — default is KEEP (v2.0)

### Core Design Patterns

**Lazy Registration** (`pipeline/registry.py`):
- Phase registry initializes only on first use to avoid import cycles
- Phases self-register via `register_phase(PhaseInstance())`
- Lookup by phase name (lowercase, striped)

**Shared Mutable Context** (`pipeline/context.py`):
```python
@dataclass
class PipelineContext:
    input_path: Path               # source image
    output_path: Path              # output directory
    current_image: np.ndarray      # RGB for compose/polish/compress
    current_rgba: np.ndarray       # isolated pre-compose RGBA
    composed_rgba_canvas: ndarray  # full canvas RGBA for shadow refinement
    alpha_mask: np.ndarray         # binary mask from isolate
    metadata: dict                 # phase-to-phase state (flags, config)
    debug: dict                    # debug artifacts
```

**Phase Contract** (`pipeline/contracts.py`):
```python
class PipelinePhase(ABC):
    phase_name: str
    def process(self, context: PipelineContext) -> PipelineContext:
        """Mutate context in place and return it."""
```

### Execution Flow

```
Entry: python -m image_utility [job]
  ↓
dispatcher.run_job(job)  → registry lookup → job-specific runner
  ↓
pipeline.runner.run()
  ├─ load_image_utility_env()    (from image_utility/.env)
  ├─ init_job_logging("pipeline.log")
  ├─ resolve_pipeline_phases(steps from env)
  ├─ For each image in INPUT_DIR:
  │   ├─ Load as RGB
  │   ├─ Create PipelineContext
  │   ├─ For each phase in registry:
  │   │   ├─ phase.process(context)
  │   │   └─ Phase mutates context in place
  │   └─ _write_final_output(context)
  └─ Return summary (exit_code, processed, skipped)
```

---

## Job Types & Dispatcher

Three jobs routed via `dispatcher.py`:

| Job | Runner | Purpose | Env Steps |
|-----|--------|---------|-----------|
| `pipeline` | `pipeline.runner.run()` | Full pipeline (env-configured steps) | `IMAGE_UTIL_PIPELINE_STEPS` |
| `compress` | `compress.phase.CompressPhase.run()` | WebP/JPEG export + thumbnails | Compress only |
| `legacy_enhance` | `legacy_enhance_img.run()` | Deprecated single-pass enhancer | Legacy (avoid) |

**Default Job**: Set `IMAGE_UTIL_DEFAULT_JOB` in `.env` (default: "pipeline").

---

## Configuration: Environment-Driven Parameterization

### Global Config (`image_utility/config.py`)
```python
PACKAGE_DIR = Path(__file__).resolve().parent        # image_utility/
WORKSPACE_ROOT = PACKAGE_DIR.parent                  # python-utility/
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "log"
```

### Required Environment Variables

**Project-Level** (`.env` in `image_utility/`):
```dotenv
IMAGE_UTIL_DEFAULT_JOB=pipeline
IMAGE_UTIL_PIPELINE_STEPS=compress,isolate,compose,shadow,polish
IMAGE_UTIL_INPUT_DIR=/path/to/input/images
IMAGE_UTIL_OUTPUT_DIR=/path/to/output
IMAGE_UTIL_MAX_FILES=500              # Optional: limit batch size
IMAGE_UTIL_LOG_DIR=log                # Optional: relative or absolute
IMAGE_UTIL_THUMBNAIL=1                # Optional: thumbnail-only mode
```

**Phase-Specific Tuning** (examples):
```dotenv
# Isolate phase (~60+ vars for v2/v3 tuning)
IMAGE_UTIL_ISOLATE_DEBUG=true
IMAGE_UTIL_ISOLATE_SEMANTIC=true      # Enable SAM v3 refinement
IMAGE_UTIL_ISOLATE_SAM_CHECKPOINT=/path/to/mobile_sam.pt
IMAGE_UTIL_ISOLATE_MIN_COMPONENT_AREA=400
IMAGE_UTIL_ISOLATE_V2_MIN_CONF=0.06

# Compose phase
IMAGE_UTIL_COMPOSE_DEBUG=false
IMAGE_UTIL_COMPOSE_CANVAS_SIZE=2000

# Shadow/Polish/Compress have similar per-phase tunables
```

### Config Loading Pattern
```python
# Each phase defines load_<phase>_config() → dataclass with env helpers
from isolate.config import load_isolate_config, _int_env, _float_env, _bool_env
cfg = load_isolate_config()  # Reads IMAGE_UTIL_ISOLATE_* vars
```

**Never hardcode values** — all must be in env/config with defaults.

---

## Key Workflows

### Running the Full Pipeline
```bash
cd /Users/mohdwaseem/Desktop/my-workspace/python-utility
# Edit image_utility/.env: set INPUT_DIR, OUTPUT_DIR, PIPELINE_STEPS
python -m image_utility pipeline
# Logs to: log/pipeline.log + stdout
```

### Compress Only (WebP Export)
```bash
IMAGE_UTIL_PIPELINE_STEPS=compress python -m image_utility
```

### Debug Single Phase
```bash
# Enable debug output for isolate phase
export IMAGE_UTIL_ISOLATE_DEBUG=true
python -m image_utility pipeline
# Outputs to: debug/isolate/{alpha, components, selected, decomposition/, ...}
```

### Tuning & Iteration
1. Edit `image_utility/.env` with new parameter values (e.g., `IMAGE_UTIL_ISOLATE_V2_MIN_CONF=0.10`)
2. Run pipeline on sample images
3. Inspect `debug/<phase>/` artifacts
4. Repeat

---

## Critical Developer Patterns

### Adding a New Phase

1. **Create phase module** (`image_utility/<phasename>/`):
   ```python
   # phase.py
   from pipeline.contracts import PipelinePhase
   
   class MyPhase(PipelinePhase):
       phase_name = "myname"
       def process(self, context: PipelineContext) -> PipelineContext:
           return process_my_phase(context)
   
   # processor.py
   def process_my_phase(context: PipelineContext) -> PipelineContext:
       # Retrieve input → mutate context → return
       context.current_image = transform(context.current_image)
       return context
   
   # config.py (optional)
   @dataclass(frozen=True)
   class MyConfig:
       ...
   def load_my_config() -> MyConfig:
       return MyConfig(...)
   ```

2. **Register in `pipeline/registry.py`**:
   ```python
   from myname.phase import MyPhase
   register_phase(MyPhase())  # in _ensure_default_phases()
   ```

3. **Add to `.env`**:
   ```dotenv
   IMAGE_UTIL_PIPELINE_STEPS=isolate,compose,myname,shadow,polish,compress
   ```

### File I/O & Batch Processing

**Image Loading** (`pipeline/runner.py`):
```python
with Image.open(ctx.input_path) as img:
    ctx.current_image = np.array(img.convert("RGB"))
```

**Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.HEIC` (see `utils.IMAGE_EXTS`).

**Sorted File Discovery**:
```python
from utils import sorted_image_files
files = sorted_image_files(input_dir)  # Alphabetically sorted
```

**Filtering** (thumbnail mode):
```python
# Only process files with trailing _0 in stem (e.g., product_13_0.jpg)
files = [p for p in files if stem_trailing_index(p.stem) == 0]
```

### Error Handling & Resilience

**File-Level Skip** (safe to continue):
```python
try:
    phase.process(ctx)
except (OSError, Exception) as exc:
    logger.warning("Phase %s failed for %s: %s", phase.phase_name, ctx.input_path.name, exc)
    skipped_count += 1
    continue  # Skip this image, process next
```

**Pipeline-Level Skip**:
- Return `PipelineRunSummary(exit_code=1, ...)` if env validation fails
- Job exits, logs reason

### Logging Convention

**Initialization**:
```python
from utils import init_job_logging
init_job_logging("jobname.log")  # Logs to workspace/log/jobname.log + stdout
```

**Phase Logging** (per phase):
```python
logger = logging.getLogger(__name__)  # Auto-includes module name
logger.info("Phase: %s", phase.phase_name)
logger.info("Wrote %s", output_file.name)
logger.warning("Skip %s: reason", input_file.name)
```

### Debug Output Structure

**Enable per-phase**: Set `IMAGE_UTIL_<PHASE>_DEBUG=true`.

**Output locations**:
```
debug/
├── isolate/
│   ├── {stem}_alpha.png               # Alpha channel viz
│   ├── {stem}_components.png          # Labeled connected components
│   ├── {stem}_selected.png            # Selected mask with overlay
│   ├── {stem}_candidates.npy          # Numpy candidate data
│   ├── decomposition/                 # SAM decomposition debug
│   ├── filtering/                     # CC analysis debug
│   ├── ranking/                       # Confidence scoring debug
│   └── semantic/                      # SAM v3 refinement debug
├── compose/
│   ├── {stem}_scaled.png
│   └── {stem}_placed.png
└── shadow/, polish/
    └── (phase-specific artifacts)
```

---

## Isolate Phase: Advanced Tuning

The **isolate** phase is the most complex, with three versions:

| Version | Segmentation | Selection | Use |
|---------|--------------|-----------|-----|
| **v1** | rembg | Legacy heuristics | Deprecated |
| **v2** | rembg | Weighted logit → sigmoid confidence model | Default prod |
| **v3** | rembg → MobileSAM (optional) | v2 + semantic refinement on ambiguous cases | Advanced (optional) |

### v2 Tuning (Confidence Model)
```python
# Key weights in /config.py: v2_weight_relative_area, v2_weight_centrality, etc.
# Calibrated via: logit = w_area*ln(area) + w_center*centrality + ...
# Then: confidence = sigmoid(logit + bias) * scale

# To select smaller products: increase v2_weight_relative_area
# To penalize edge-contact more: increase v2_weight_border_contact
# Tweak: IMAGE_UTIL_ISOLATE_V2_MIN_CONF (rejection threshold, 0–1)
```

### v3 Tuning (SAM Semantic Refinement)
```dotenv
IMAGE_UTIL_ISOLATE_SEMANTIC=true
IMAGE_UTIL_ISOLATE_SAM_CHECKPOINT=/path/to/mobile_sam.pt
IMAGE_UTIL_ISOLATE_SAM_GPU=false              # CPU for stability
IMAGE_UTIL_ISOLATE_SAM_POINTS_PER_SIDE=24    # Grid density (higher = slower)
IMAGE_UTIL_ISOLATE_SEM_TRIG_V2_BELOW=0.92    # Activate if v2 conf < this
IMAGE_UTIL_ISOLATE_SEM_TRIG_AMBIG=0.72       # Activate if second/first ≥ this
IMAGE_UTIL_ISOLATE_SEM_MIN_CONF=0.22         # Final semantic confidence floor
```

---

## Common Issues & Debugging

### Pipeline Doesn't Start
- **Check**: `IMAGE_UTIL_INPUT_DIR` and `IMAGE_UTIL_OUTPUT_DIR` are valid directories
- **Check**: `.env` is in `image_utility/` and loaded via `load_image_utility_env()`
- **Check**: `IMAGE_UTIL_PIPELINE_STEPS` is non-empty CSV of valid phase names

### Many Images Skipped
- **Check**: Log file in `log/pipeline.log` for per-file errors
- **Check**: Image format (only `.jpg`, `.jpeg`, `.png`, `.HEIC` supported)
- **Check**: Enable debug mode (`IMAGE_UTIL_ISOLATE_DEBUG=true`) → inspect `debug/isolate/*`

### Segmentation Failures (isolate phase)
- **Issue**: All images rejected (alpha mask empty)
- **Debug**: Check `debug/isolate/{stem}_alpha.png` — is foreground detected?
- **Solution**: Lower `IMAGE_UTIL_ISOLATE_V2_MIN_CONF` or increase `IMAGE_UTIL_ISOLATE_CENTER_BIAS`

### SAM Crashes (v3 semantic)
- **Issue**: File not found or CUDA OOM
- **Verify**: `IMAGE_UTIL_ISOLATE_SAM_CHECKPOINT` points to valid `.pt` file
- **Try**: Set `IMAGE_UTIL_ISOLATE_SAM_GPU=false` (CPU inference is slower but stable)
- **Reduce**: `IMAGE_UTIL_ISOLATE_SAM_POINTS_PER_SIDE` from 24 to 12

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `image_utility/__main__.py` | CLI entry point, argparse setup |
| `image_utility/dispatcher.py` | Job registry & routing |
| `image_utility/utils.py` | Env loading, path resolution, image file discovery |
| `image_utility/pipeline/runner.py` | Main orchestration loop (load → phase-chain → write) |
| `image_utility/pipeline/contracts.py` | `PipelinePhase` abstract base |
| `image_utility/pipeline/context.py` | `PipelineContext` shared state |
| `image_utility/pipeline/registry.py` | Phase registry (lazy initialization) |
| `image_utility/isolate/phase.py`, `processor.py`, `config.py` | Isolate phase (most complex) |
| `image_utility/{compose,shadow,polish,compress}/phase.py` | Other phases (simpler pattern) |
| `app_logging/__init__.py` | File + stdout logging utility |

---

## External Dependencies & Replacements

**Current Stack** (requirements.txt):
- `Pillow>=10.0.0` — Image I/O, basic transforms
- `opencv-python-headless>=4.10.0.0` — Morphology, advanced CV
- `numpy>=1.26.0` — Array operations
- `rembg>=2.0.50` — Segmentation (default, replaceable)
- `onnxruntime>=1.18.0` — rembg inference backend

**Optional** (install for v3 semantic):
- `torch`, `torchvision` — PyTorch for SAM
- `timm` — Model zoo for SAM backbone
- `mobile-sam` — MobileSAM repository

**Future Replacements** (per authority):
- rembg → SAM2, GroundedSAM, YOLO segmentation, custom models
- Pillow/CV → Advanced relight models, reflection generation, category-specific composition

Architecture supports swapping segmentation without downstream impact.

---

## Testing & Local Development

**Run pipeline on sample**:
```bash
export IMAGE_UTIL_INPUT_DIR=./resource
export IMAGE_UTIL_OUTPUT_DIR=/tmp/out
python -m image_utility pipeline
```

**Inspect logs**:
```bash
tail -f log/pipeline.log
```

**Check debug artifacts**:
```bash
open debug/isolate/  # View alpha/component masks
```

---

## Summary for AI Agents

✓ **Phase-based architecture**: Each phase is independent, mutates shared context, easy to test.  
✓ **Registry pattern**: Phases lazy-load; no hard-coded dispatch chains.  
✓ **Environment-driven**: All config in `.env` with tunable defaults per phase.  
✓ **Batch resilient**: Skip bad images, continue processing.  
✓ **Extensible segmentation**: rembg replaceable with SAM/custom models.  
✓ **Debug-friendly**: Per-phase debug mode outputs artifacts for inspection.  
✓ **Isolate complexity**: v2/v3 confidence models with 60+ tuning parameters for product-specific refinement.  

When adding features or debugging: **Read the authority** (`image_utility/authority/IMAGE_PIPELINE_AUTHORITY.md`), **check `.env`**, **enable debug mode**, **inspect artifacts**.

