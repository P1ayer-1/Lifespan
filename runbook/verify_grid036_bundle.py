"""Verify private grid036 bundle without displaying story contents."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
checks=json.loads((root/'checksums.json').read_text(encoding='utf-8'))
for name,expected in checks.items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):
        raise SystemExit('Unsafe checksum path')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Missing or changed bundle file: '+name)
print('Verified',len(checks),'bundle files')
