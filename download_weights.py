"""Download the pinned upstream checkpoint and reproduce the local decoder."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent

def main():
    provenance = json.loads((ROOT / 'reports/provenance.json').read_text())
    expected = json.loads((ROOT / 'reports/summary.json').read_text())['sha256']
    target = ROOT / 'weights/taew2_1.safetensors'
    target.parent.mkdir(exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        print('Downloading pinned TAEW2.1 checkpoint...', flush=True)
        with urllib.request.urlopen(provenance['weight_url'], timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Checkpoint SHA-256 mismatch; refusing to install')
        temporary = target.with_suffix('.download')
        temporary.write_bytes(data)
        temporary.replace(target)
    subprocess.run([sys.executable, str(ROOT / 'inspect_weights.py')], check=True)
    print('Verified weights ready.')

if __name__ == '__main__':
    main()
