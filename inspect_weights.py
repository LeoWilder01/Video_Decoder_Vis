"""Inspect and extract official F16 safetensors, using only Python's standard library."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent

def read_checkpoint(path):
    raw = Path(path).read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    return json.loads(raw[8:8+size]), raw[8+size:]

def values(info, payload):
    assert info['dtype'] == 'F16', info['dtype']
    lo, hi = info['data_offsets']
    return [v[0] for v in struct.iter_unpack('<e', payload[lo:hi])]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/taew2_1.safetensors')
    parser.add_argument('--tensor', help='Print every value of a named tensor as JSON')
    args = parser.parse_args()
    header, payload = read_checkpoint(args.weights)
    if args.tensor:
        info = header[args.tensor]
        print(json.dumps({'name': args.tensor, 'shape': info['shape'], 'values_flat_C_order': values(info, payload)}))
        return
    rows = []
    decoder_header, decoder_data = {}, bytearray()
    for name, info in header.items():
        if name == '__metadata__':
            continue
        v = values(info, payload)
        n = len(v)
        mean = math.fsum(v)/n
        rows.append(dict(name=name, shape=info['shape'], dtype=info['dtype'], parameters=n,
                         minimum=min(v), maximum=max(v), mean=mean,
                         std=(math.fsum((x-mean)**2 for x in v)/n)**0.5))
        if name.startswith('decoder.'):
            lo, hi = info['data_offsets']
            begin = len(decoder_data)
            decoder_data.extend(payload[lo:hi])
            decoder_header[name] = dict(info, data_offsets=[begin, len(decoder_data)])
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    (reports/'weights-inventory.json').write_text(json.dumps(rows, indent=2))
    encoded = json.dumps(decoder_header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    extracted = ROOT/'weights/taew2_1_decoder.safetensors'
    extracted.write_bytes(struct.pack('<Q', len(encoded))+encoded+decoder_data)
    summary = {'checkpoint_bytes': args.weights.stat().st_size,
               'sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
               'decoder_file_bytes': extracted.stat().st_size}
    for part in ('encoder', 'decoder'):
        subset = [r for r in rows if r['name'].startswith(part+'.')]
        summary[part] = {'tensors': len(subset), 'parameters': sum(r['parameters'] for r in subset)}
    (reports/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
