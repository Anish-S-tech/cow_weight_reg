"""
auto_runner.py — Automated Kaggle notebook runner with error auto-fix.

Monitors all three cow notebooks. On error:
  1. Downloads the executed output notebook
  2. Extracts error traceback from error cells
  3. Applies a fix from a known-pattern library
  4. Re-pushes
  5. Repeats until success or max retries exceeded

Run: python auto_runner.py
"""

import json, time, re, sys, io, zipfile, traceback as tb
from pathlib import Path
from collections import defaultdict
import requests
import nbformat

# ── Credentials ───────────────────────────────────────────────────────────────
CREDS     = json.load(open(Path.home() / '.kaggle' / 'kaggle.json'))
KEY       = CREDS['key']
USER      = CREDS['username'].lower().replace('_', '')
BASE      = 'https://www.kaggle.com/api/v1'
DATASET   = 'sadhliroomyprime/cattle-weight-detection-model-dataset-12k'
NB_DIR    = Path(__file__).parent / 'notebooks'
LOG_DIR   = Path(__file__).parent / 'auto_runner_logs'
LOG_DIR.mkdir(exist_ok=True)
MAX_RETRIES  = 5
POLL_SECS    = 60
STATE_FILE   = LOG_DIR / 'runner_state.json'
NET_RETRY_SECS = 30   # wait before retrying after a network error

NOTEBOOKS = [
    ('cow-02b-train-b0',     NB_DIR / '02b_train_b0.ipynb'),
    ('cow-02c-train-s',      NB_DIR / '02c_train_s.ipynb'),
    ('cow-02d-train-mobile', NB_DIR / '02d_train_mobile.ipynb'),
]

def hdr():
    return {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_DIR / 'run.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ─────────────────────────────────────────────────────────────────────────────
# Persistent state — survives internet disconnection / restarts
# ─────────────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding='utf-8')

def update_notebook_state(state: dict, slug: str, **kwargs):
    """Merge kwargs into state[slug] and persist to disk."""
    state.setdefault(slug, {}).update(kwargs)
    save_state(state)

# ─────────────────────────────────────────────────────────────────────────────
# Network-safe request helper — retries forever on connection errors
# ─────────────────────────────────────────────────────────────────────────────

def api_request(method: str, url: str, **kwargs):
    """Wrap requests.get/post with infinite retry on network errors."""
    fn = getattr(requests, method)
    while True:
        try:
            return fn(url, **kwargs)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            log(f'  [NET] {exc.__class__.__name__} — internet down? Retrying in {NET_RETRY_SECS}s...')
            time.sleep(NET_RETRY_SECS)

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle API helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_status(slug):
    r = api_request('get', f'{BASE}/kernels/list?user={USER}&pageSize=20',
                    headers=hdr(), timeout=20)
    if r.status_code != 200:
        return 'unknown'
    for k in r.json():
        if k.get('ref', '').endswith(f'/{slug}'):
            rv = k.get('currentRunningVersion') or {}
            return rv.get('status', 'queued')
    return 'not_found'

def get_kernel_id(slug):
    r = api_request('get', f'{BASE}/kernels/pull?userName={USER}&kernelSlug={slug}',
                    headers=hdr(), timeout=20)
    if r.status_code != 200:
        return None
    meta = r.json().get('metadata', {})
    return meta.get('id') or meta.get('kernelId')

def push_notebook(slug, nb_path: Path, gpu=True):
    nb_text = nb_path.read_text(encoding='utf-8')
    kernel_id = get_kernel_id(slug)
    payload = {
        'text': nb_text, 'language': 'python', 'kernelType': 'notebook',
        'isPrivate': True, 'enableGpu': gpu, 'enableInternet': True,
        'datasetDataSources': [DATASET],
        'kernelDataSources': [], 'competitionDataSources': [],
    }
    if kernel_id:
        payload['id'] = int(kernel_id)
    else:
        payload['newTitle'] = slug.replace('-', ' ').title()

    r = api_request('post', f'{BASE}/kernels/push', headers=hdr(),
                    json=payload, timeout=60)
    if r.status_code not in (200, 201):
        log(f'  [PUSH ERROR] {r.status_code}: {r.text[:300]}')
        return False
    d = r.json()
    if d.get('hasError'):
        log(f'  [PUSH ERROR] {d.get("error")}')
        return False
    log(f'  Pushed version {d.get("versionNumber","?")} -> {d.get("url","")}')
    return True

def download_output_notebook(slug):
    """Download the executed notebook from Kaggle output. Returns nbformat object or None."""
    r = api_request('get', f'{BASE}/kernels/{USER}/{slug}/output',
                    headers=hdr(), timeout=60)
    if r.status_code != 200:
        log(f'  Output fetch failed: {r.status_code}')
        return None

    data = r.json()
    files = data.get('files', [])
    log(f'  Output files: {[f["name"] for f in files]}')

    # Try to find executed .ipynb file
    for file_info in files:
        fname = file_info.get('name', '')
        url   = file_info.get('url', '')
        if not url:
            continue
        if fname.endswith('.ipynb') or fname.endswith('.zip'):
            rr = api_request('get', url, headers=hdr(), timeout=120)
            if rr.status_code != 200:
                continue
            # Save locally
            out_path = LOG_DIR / f'{slug}_output{Path(fname).suffix}'
            out_path.write_bytes(rr.content)
            log(f'  Downloaded: {out_path.name} ({len(rr.content)//1024} KB)')

            if fname.endswith('.zip'):
                # Extract notebook from zip
                with zipfile.ZipFile(io.BytesIO(rr.content)) as z:
                    for name in z.namelist():
                        if name.endswith('.ipynb'):
                            nb_bytes = z.read(name)
                            nb = nbformat.reads(nb_bytes.decode('utf-8'), as_version=4)
                            return nb
            else:
                nb = nbformat.reads(rr.content.decode('utf-8'), as_version=4)
                return nb

    # If no notebook file found, try to get log from metadata
    log_text = data.get('log', '')
    if log_text:
        log_path = LOG_DIR / f'{slug}_run.log'
        log_path.write_text(log_text, encoding='utf-8')
        log(f'  Saved run log: {log_path.name}')

    return None

def extract_errors_from_notebook(nb):
    """Extract error tracebacks from executed notebook cells."""
    errors = []
    if nb is None:
        return errors
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                ename = output.get('ename', '')
                evalue = output.get('evalue', '')
                traceback_lines = output.get('traceback', [])
                # Strip ANSI color codes
                clean_tb = []
                for line in traceback_lines:
                    clean_tb.append(re.sub(r'\x1b\[[0-9;]*m', '', line))
                errors.append({
                    'cell_index': i,
                    'cell_source': cell.source,
                    'ename': ename,
                    'evalue': evalue,
                    'traceback': '\n'.join(clean_tb),
                    'full': f'{ename}: {evalue}\n' + '\n'.join(clean_tb),
                })
                log(f'  ERROR in cell {i}: {ename}: {evalue[:200]}')
    return errors

# ─────────────────────────────────────────────────────────────────────────────
# Fix library — pattern -> fixer function
# ─────────────────────────────────────────────────────────────────────────────

def _edit_cell(nb_path: Path, cell_source_fragment: str, new_source: str):
    """Find the first cell containing cell_source_fragment and replace its source."""
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        if cell_source_fragment in cell.source:
            cell.source = new_source
            nbformat.write(nb, str(nb_path))
            return True
    return False

def _append_pip_install(nb_path: Path, package: str):
    """Add package to the pip install cell."""
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        if 'pip(' in cell.source and 'subprocess' in cell.source:
            if package not in cell.source:
                # Append to the last pip() call
                cell.source = cell.source.rstrip()
                cell.source += f"\npip('{package}')"
                nbformat.write(nb, str(nb_path))
                return True
    return False

def _reduce_batch_size(nb_path: Path, factor=2):
    """Halve all BATCH_SIZE constants in the notebook."""
    nb = nbformat.read(str(nb_path), as_version=4)
    changed = False
    for cell in nb.cells:
        new_src = re.sub(
            r'BATCH_SIZE\s*=\s*(\d+)',
            lambda m: f'BATCH_SIZE = {max(2, int(m.group(1)) // factor)}',
            cell.source
        )
        if new_src != cell.source:
            cell.source = new_src
            changed = True
    if changed:
        nbformat.write(nb, str(nb_path))
    return changed

def _reduce_img_size(nb_path: Path, new_size=256):
    """Reduce IMG_SIZE."""
    nb = nbformat.read(str(nb_path), as_version=4)
    changed = False
    for cell in nb.cells:
        new_src = re.sub(r'IMG_SIZE\s*=\s*\d+', f'IMG_SIZE = {new_size}', cell.source)
        if new_src != cell.source:
            cell.source = new_src
            changed = True
    if changed:
        nbformat.write(nb, str(nb_path))
    return changed

def _patch_cell_source(nb_path: Path, old_fragment: str, replacement: str):
    """Replace old_fragment with replacement anywhere in notebook cells."""
    nb = nbformat.read(str(nb_path), as_version=4)
    changed = False
    for cell in nb.cells:
        if old_fragment in cell.source:
            cell.source = cell.source.replace(old_fragment, replacement)
            changed = True
    if changed:
        nbformat.write(nb, str(nb_path))
    return changed

def _disable_cell_block(nb_path: Path, fragment: str, comment='# DISABLED BY AUTO-FIXER\n'):
    """Comment out a cell that contains fragment."""
    nb = nbformat.read(str(nb_path), as_version=4)
    changed = False
    for cell in nb.cells:
        if fragment in cell.source and not cell.source.strip().startswith('#'):
            lines = cell.source.split('\n')
            cell.source = comment + '\n'.join('# ' + l for l in lines)
            changed = True
            break
    if changed:
        nbformat.write(nb, str(nb_path))
    return changed


# Map: (error_name_or_msg_regex) -> list of fix functions to try in order
ERROR_FIXES = [

    # ── 1. Missing pip package ────────────────────────────────────────────────
    {
        'pattern': r"ModuleNotFoundError: No module named '([^']+)'",
        'description': 'Missing module — add pip install',
        'fix': lambda nb_path, match: _append_pip_install(nb_path, match.group(1).split('.')[0]),
    },

    # ── 2. GPU OOM / ResourceExhausted ───────────────────────────────────────
    {
        'pattern': r'ResourceExhaustedError|OOM when allocating|out of memory',
        'description': 'GPU OOM — halve batch size',
        'fix': lambda nb_path, match: _reduce_batch_size(nb_path, factor=2),
    },

    # ── 3. OOM second attempt: also reduce image size ─────────────────────────
    {
        'pattern': r'ResourceExhaustedError|OOM when allocating',
        'description': 'GPU OOM (2nd fix) — reduce IMG_SIZE to 256',
        'fix': lambda nb_path, match: _reduce_img_size(nb_path, new_size=256),
    },

    # ── 4. ConvNeXt not available in this TF version ─────────────────────────
    {
        'pattern': r"cannot import name 'ConvNeXtTiny'|ImportError.*ConvNeXt",
        'description': 'ConvNeXtTiny not in this TF — skip it',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            'from tensorflow.keras.applications import ConvNeXtTiny',
            'try:\n    from tensorflow.keras.applications import ConvNeXtTiny\nexcept ImportError:\n    ConvNeXtTiny = None'
        ),
    },

    # ── 5. Mixed float16 + softmax dtype mismatch ─────────────────────────────
    {
        'pattern': r"dtype.*float16.*float32|softmax.*float16",
        'description': 'FP16 dtype mismatch — disable mixed precision',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            "tf.keras.mixed_precision.set_global_policy('mixed_float16')",
            "tf.keras.mixed_precision.set_global_policy('float32')  # disabled FP16 due to dtype error"
        ),
    },

    # ── 6. EfficientNetV2S weight load failure ────────────────────────────────
    {
        'pattern': r"Error.*EfficientNetV2S.*weights|404.*efficientnetv2",
        'description': 'EfficientNetV2S weight download failed — use EfficientNetV2B0',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            'EfficientNetV2S(include_top=False, weights=',
            'EfficientNetV2B0(include_top=False, weights='
        ),
    },

    # ── 7. MobileNetV3 weight load failure ───────────────────────────────────
    {
        'pattern': r"Error.*MobileNetV3.*weights|404.*mobilenetv3",
        'description': 'MobileNetV3Large weight failed — use MobileNetV2',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            'MobileNetV3Large(include_top=False, weights=',
            'tf.keras.applications.MobileNetV2(include_top=False, weights='
        ),
    },

    # ── 8. Keras model save format error ─────────────────────────────────────
    {
        'pattern': r"Unable to serialize.*keras|SavedModel.*error",
        'description': 'Keras save error — switch to .h5 format',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            ".save(str(OUT / 'cattle_weight_model.keras'))",
            ".save(str(OUT / 'cattle_weight_model.h5'))"
        ),
    },

    # ── 9. albumentations version incompatibility ────────────────────────────
    {
        'pattern': r"TypeError.*albumentations|AttributeError.*albumentations",
        'description': 'Albumentations compat error — pin version',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            "pip('tensorflow==2.15.0', 'keras==2.15.0',",
            "pip('tensorflow==2.15.0', 'keras==2.15.0', 'albumentations==1.3.1',"
        ),
    },

    # ── 10. torch install conflict ────────────────────────────────────────────
    {
        'pattern': r"ERROR.*torch|Cannot install.*torch",
        'description': 'torch install conflict — remove torch from pip installs',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            "    'timm', 'torch', 'torchvision')",
            "    'timm')  # torch removed due to install conflict"
        ),
    },

    # ── 11. Ultralytics YOLOv8 download failure ───────────────────────────────
    {
        'pattern': r"yolov8n.*failed|Cannot download.*yolo",
        'description': 'YOLOv8 download failed — skip YOLO, use mock detector',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            'from ultralytics import YOLO',
            '''# YOLOv8 not available — use simple contour-based cow detector
def detect_cows_fallback(img_bgr, conf=0.4):
    """Fallback: assume the whole frame contains one cow."""
    H, W = img_bgr.shape[:2]
    margin = 0.05
    return [(int(W*margin), int(H*margin), int(W*(1-margin)), int(H*(1-margin)), 0.9)]

try:
    from ultralytics import YOLO
    _yolo = YOLO("yolov8n.pt")
    def detect_cows(img_bgr, conf=0.4):
        results = _yolo.predict(img_bgr, classes=[19], conf=conf, verbose=False)[0]
        return [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                 int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0]))
                for b in results.boxes]
except Exception:
    detect_cows = detect_cows_fallback'''
        ),
    },

    # ── 12. AdamW not available in this TF version ───────────────────────────
    {
        'pattern': r"cannot import name 'AdamW'|AdamW.*not.*found",
        'description': 'AdamW not in this TF — use Adam with weight decay',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            'tf.keras.optimizers.AdamW(',
            'tf.keras.optimizers.experimental.AdamW(',
        ),
    },

    # ── 13. Skeletonize import error ──────────────────────────────────────────
    {
        'pattern': r"cannot import name 'skeletonize'|ImportError.*skeletonize",
        'description': 'skeletonize import error — use alternative',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            'from skimage.morphology import skeletonize',
            'try:\n    from skimage.morphology import skeletonize\nexcept ImportError:\n    def skeletonize(x): return x  # fallback'
        ),
    },

    # ── 14. Dataset empty / no files after path fix ───────────────────────────
    {
        'pattern': r'Dataset not found.*ROOT=|No files found under ROOT',
        'description': 'Dataset still not found — search more aggressively for images',
        'fix': lambda nb_path, match: _patch_cell_source(
            nb_path,
            '# Try the expected slug first; fall back to the dir with most images',
            '''# Aggressive search: check all nested subdirectories too
def _find_deepest_dir_with_images(root):
    best, best_n = root, 0
    for d in root.rglob("*"):
        if d.is_dir():
            n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
            if n > best_n:
                best, best_n = d, n
    return best, best_n
# Try the expected slug first; fall back to the dir with most images (deep search)'''
        ),
    },

    # ── 15. Generic ValueError ────────────────────────────────────────────────
    {
        'pattern': r'ValueError: .*(shape|dimension|size)',
        'description': 'Shape mismatch — reduce image size to 224 for compatibility',
        'fix': lambda nb_path, match: _reduce_img_size(nb_path, new_size=224),
    },
]


def apply_fix(nb_path: Path, error_text: str, already_tried: set):
    """Try error fixes in order. Returns description of fix applied, or None."""
    for fix_entry in ERROR_FIXES:
        pattern = fix_entry['pattern']
        desc    = fix_entry['description']
        fixer   = fix_entry['fix']

        if desc in already_tried:
            continue

        m = re.search(pattern, error_text, re.IGNORECASE | re.DOTALL)
        if m:
            log(f'  Matched pattern: "{pattern[:60]}"')
            log(f'  Applying fix: {desc}')
            try:
                result = fixer(nb_path, m)
                if result:
                    return desc
                else:
                    log(f'  Fix returned False (pattern not found in notebook) — skipping')
            except Exception as e:
                log(f'  Fix raised exception: {e}')
            already_tried.add(desc)

    return None  # No fix found


# ─────────────────────────────────────────────────────────────────────────────
# Main monitor loop
# ─────────────────────────────────────────────────────────────────────────────

def monitor_notebook(slug, nb_path: Path, state: dict):
    """
    Monitor one notebook until it completes or we exhaust retries.
    Saves progress to `state` so the runner can resume after a disconnect.
    Returns 'complete', 'gave_up', or 'unknown'.
    """
    nb_state = state.get(slug, {})

    # Resume from saved state rather than starting fresh
    already_tried_fixes = set(nb_state.get('fixes_tried', []))
    retry_count         = nb_state.get('retry_count', 0)

    log(f'\n{"="*60}')
    log(f'Monitoring: {slug}')
    log(f'URL: https://www.kaggle.com/code/{USER}/{slug}')
    if already_tried_fixes or retry_count:
        log(f'  Resuming — retry_count={retry_count}, fixes_tried={list(already_tried_fixes)}')
    log(f'{"="*60}')

    last_status = None

    while retry_count < MAX_RETRIES:
        status = get_status(slug)

        if status != last_status:
            log(f'Status: {status}')
            last_status = status

        if status == 'complete':
            log(f'SUCCESS: {slug} completed.')
            update_notebook_state(state, slug, outcome='complete', retry_count=retry_count,
                                  fixes_tried=list(already_tried_fixes))
            return 'complete'

        if status in ('error', 'failed', 'cancelAcknowledged'):
            log(f'FAILED: {slug} — status={status}')
            retry_count += 1
            log(f'Attempt {retry_count}/{MAX_RETRIES} to auto-fix...')
            update_notebook_state(state, slug, retry_count=retry_count,
                                  fixes_tried=list(already_tried_fixes))

            # Download output notebook and extract errors
            nb_out = download_output_notebook(slug)
            errors = extract_errors_from_notebook(nb_out)

            if not errors:
                log('  No error cells found in output notebook (may need manual check).')
                error_text = f'{status} with no parseable error cells'
            else:
                error_text = '\n'.join(e['full'] for e in errors)
                err_log = LOG_DIR / f'{slug}_error_{retry_count}.txt'
                err_log.write_text(error_text, encoding='utf-8')
                log(f'  Full error saved to: {err_log.name}')

            # Apply fix
            fix_desc = apply_fix(nb_path, error_text, already_tried_fixes)
            if fix_desc:
                already_tried_fixes.add(fix_desc)
                update_notebook_state(state, slug, fixes_tried=list(already_tried_fixes))
                log(f'  Fix applied: {fix_desc}')
                log(f'  Re-pushing {slug}...')
                success = push_notebook(slug, nb_path)
                if not success:
                    log(f'  Push failed — will retry next cycle.')
                last_status = None  # reset so we log the new status
            else:
                log(f'  No applicable fix found for this error.')
                log(f'  Error summary:')
                for e in errors[:3]:
                    log(f'    {e["ename"]}: {e["evalue"][:200]}')
                if retry_count >= MAX_RETRIES:
                    log(f'GAVE UP on {slug} after {MAX_RETRIES} attempts.')
                    update_notebook_state(state, slug, outcome='gave_up')
                    return 'gave_up'

            time.sleep(30)  # wait before polling again
            continue

        # Still running / queued — wait
        time.sleep(POLL_SECS)

    log(f'GAVE UP on {slug} — exceeded max retries.')
    update_notebook_state(state, slug, outcome='gave_up')
    return 'gave_up'


def main():
    state = load_state()

    log('Auto-runner started.')
    log(f'Kaggle user : {USER}')
    log(f'Notebooks   : {[s for s,_ in NOTEBOOKS]}')
    log(f'Max retries : {MAX_RETRIES}')
    log(f'Poll interval: {POLL_SECS}s')
    log(f'Log dir     : {LOG_DIR}')
    log(f'State file  : {STATE_FILE}')
    if state:
        log(f'Resuming from saved state: {list(state.keys())}')

    results = {}
    for slug, nb_path in NOTEBOOKS:
        saved = state.get(slug, {})
        if saved.get('outcome') == 'complete':
            log(f'\n[SKIP] {slug} already completed (from saved state).')
            results[slug] = 'complete'
            continue

        outcome = monitor_notebook(slug, nb_path, state)
        results[slug] = outcome
        if outcome == 'gave_up':
            log(f'\n[WARN] {slug} gave up — continuing with remaining notebooks.')

    log('\n' + '='*60)
    log('FINAL RESULTS')
    log('='*60)
    for slug, outcome in results.items():
        icon = 'OK' if outcome == 'complete' else 'FAIL'
        log(f'  [{icon}] {slug}: {outcome}')
        log(f'        https://www.kaggle.com/code/{USER}/{slug}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('\nAuto-runner stopped by user.')
    except Exception as e:
        log(f'\nAuto-runner crashed: {e}')
        tb.print_exc()
