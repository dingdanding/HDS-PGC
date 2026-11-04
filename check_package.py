"""Syntax, file-integrity and accidental binary/private-path checks."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT/'docs/release_hashes.json').read_text(encoding='utf-8'))
    for rel, expected in manifest.items():
        p = ROOT/rel
        assert p.is_file(), f'Missing: {rel}'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, f'Changed: {rel}'
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        assert p.suffix.lower() not in {'.pth','.pt','.rar','.zip','.safetensors'}, p
        if p.suffix in {'.py','.md','.json','.txt','.yml'}:
            text = p.read_text(encoding='utf-8-sig')
            assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]', text), f'Absolute drive path: {p}'
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    print(f'PASS: {len(manifest)} file hashes; Python syntax; no weights/archives or absolute drive paths')

if __name__ == '__main__':
    main()
