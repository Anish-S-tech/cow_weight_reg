"""
Kaggle Kernel Runner — Bearer-auth REST API (KGAT tokens).
Usage:
  python training/kaggle_runner.py --notebook training/notebooks/01_dataset_exploration.ipynb [--gpu] [--no-wait]
"""
import argparse, json, time, sys
from pathlib import Path
import requests

CREDS_PATH = Path.home() / '.kaggle' / 'kaggle.json'
BASE       = 'https://www.kaggle.com/api/v1'
DATASET    = 'sadhliroomyprime/cattle-weight-detection-model-dataset-12k'

def creds():
    with open(CREDS_PATH) as f:
        c = json.load(f)
    key = c['key']
    # Kaggle strips underscores from display username to form the URL slug
    # e.g. "Anish_1718" -> "anish1718"
    raw = c['username']
    slug_user = raw.lower().replace('_', '')
    return slug_user, key

def hdr(key):
    return {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

# ── Kernel push ───────────────────────────────────────────────────────────────
def get_kernel_id(username, key, slug):
    """Return numeric kernel ID via pull endpoint, or None if not found."""
    r = requests.get(
        f'{BASE}/kernels/pull?userName={username}&kernelSlug={slug}',
        headers=hdr(key), timeout=20)
    if r.status_code != 200:
        return None
    d = r.json()
    meta = d.get('metadata', d)
    return meta.get('id') or meta.get('kernelId')

def push(username, key, slug, nb_path: Path, gpu=True):
    nb_text = nb_path.read_text(encoding='utf-8')
    kernel_id = get_kernel_id(username, key, slug)

    payload = {
        'text':                   nb_text,
        'language':               'python',
        'kernelType':             'notebook',
        'isPrivate':              True,
        'enableGpu':              gpu,
        'enableInternet':         True,
        'datasetDataSources':     [DATASET],
        'kernelDataSources':      [],
        'competitionDataSources': [],
    }
    if kernel_id:
        # Update existing kernel — numeric id required
        payload['id'] = int(kernel_id)
        print(f'  Updating kernel id={kernel_id} ({username}/{slug}) ...')
    else:
        # New kernel — title becomes the slug
        payload['newTitle'] = slug.replace('-', ' ').title()
        print(f'  Creating new kernel {username}/{slug} ...')

    r = requests.post(f'{BASE}/kernels/push', headers=hdr(key),
                      json=payload, timeout=60)
    if r.status_code not in (200, 201):
        print(f'[ERROR] Push failed ({r.status_code}): {r.text[:600]}')
        sys.exit(1)
    d = r.json()
    if d.get('hasError'):
        print(f'[ERROR] {d.get("error")}')
        sys.exit(1)
    ver = d.get('versionNumber', '?')
    url = d.get('url', '')
    print(f'  Pushed version {ver} -> {url}')
    return d

# ── Status polling ────────────────────────────────────────────────────────────
def status(username, key, slug):
    r = requests.get(f'{BASE}/kernels/{username}/{slug}',
                     headers=hdr(key), timeout=30)
    if r.status_code == 404:
        return 'queued', {}
    if r.status_code != 200:
        return 'unknown', {}
    d = r.json()
    rv = d.get('currentRunningVersion') or {}
    s  = rv.get('status', d.get('status', 'unknown'))
    return s, d

def wait(username, key, slug, poll=45, timeout_min=180):
    print(f'\n  Polling {username}/{slug} every {poll}s  (timeout {timeout_min} min)')
    deadline = time.time() + timeout_min * 60
    last = None
    while time.time() < deadline:
        s, d = status(username, key, slug)
        if s != last:
            ts = time.strftime('%H:%M:%S')
            print(f'  [{ts}]  {s}')
            last = s
        if s in ('complete', 'error', 'failed', 'cancelAcknowledged'):
            return s, d
        time.sleep(poll)
    return 'timeout', {}

# ── Output download ───────────────────────────────────────────────────────────
def download(username, key, slug, out_dir: Path):
    r = requests.get(f'{BASE}/kernels/{username}/{slug}/output',
                     headers=hdr(key), timeout=60)
    if r.status_code != 200:
        print(f'  [WARN] Output fetch {r.status_code}')
        return
    files = r.json().get('files', [])
    print(f'  {len(files)} output file(s)')
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        url  = f.get('url', '')
        name = Path(f.get('name', 'out')).name
        if not url:
            continue
        rr = requests.get(url, headers=hdr(key), timeout=120)
        if rr.status_code == 200:
            dest = out_dir / name
            dest.write_bytes(rr.content)
            print(f'    OK {name}  ({len(rr.content)/1e6:.1f} MB)')
        else:
            print(f'    FAIL {name}  ({rr.status_code})')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', required=True)
    ap.add_argument('--slug',     default=None)
    ap.add_argument('--gpu',      action='store_true', default=True)
    ap.add_argument('--no-gpu',   dest='gpu', action='store_false')
    ap.add_argument('--no-wait',  action='store_true')
    args = ap.parse_args()

    username, key = creds()
    nb = Path(args.notebook)
    slug = args.slug or 'cow-' + nb.stem.replace('_', '-').replace(' ', '-')

    print(f'\n{"="*60}')
    print(f'Notebook : {nb}')
    print(f'Slug     : {slug}')
    print(f'GPU      : {args.gpu}')
    print(f'Dataset  : {DATASET}')
    print(f'{"="*60}')

    push(username, key, slug, nb, gpu=args.gpu)

    if args.no_wait:
        print('Pushed (not waiting). View at:')
        print(f'  https://www.kaggle.com/code/{username}/{slug}')
        return

    final, data = wait(username, key, slug)
    print(f'\nFinal: {final}')

    if final == 'complete':
        out = Path('kaggle_outputs') / slug
        print(f'Downloading → {out}/')
        download(username, key, slug, out)
        print('Done.\n')
    elif final in ('error', 'failed'):
        rv  = data.get('currentRunningVersion') or {}
        msg = rv.get('errorMessage', '') or data.get('errorMessage', '')
        if msg:
            print(f'Error: {msg[:1000]}')
        print(f'View logs: https://www.kaggle.com/code/{username}/{slug}')

if __name__ == '__main__':
    main()
