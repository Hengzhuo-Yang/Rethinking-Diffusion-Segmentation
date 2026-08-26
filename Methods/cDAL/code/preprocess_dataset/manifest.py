from pathlib import Path


def read_manifest(path):
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    values = []
    seen = set()
    for raw_line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        value = Path(value).stem
        if value in seen:
            raise ValueError(f"Duplicate manifest entry {value!r} in {manifest_path}")
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return values
