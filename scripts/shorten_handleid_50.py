from __future__ import annotations

import csv
import re
from pathlib import Path

MAX_LEN = 50

REPLACE = {
    "universal": "univ",
    "headlight": "headlt",
    "headlights": "headlts",
    "indicator": "ind",
    "indicators": "inds",
    "auxiliary": "aux",
    "accessories": "acc",
    "protection": "protect",
    "stainless": "ss",
    "yellow": "ylw",
    "exhaust": "exh",
    "charger": "chgr",
    "camera": "cam",
    "handlebar": "hbar",
    "motorcycle": "moto",
}

DROP_TOKENS = {
    "with",
    "and",
    "for",
    "the",
    "of",
    "all",
    "bike",
    "bikes",
    "scooty",
    "fitting",
    "available",
    "set",
    "single",
    "piece",
    "pc",
}


def compress_token(token: str) -> str:
    if len(token) <= 4:
        return token
    compact = token[0] + re.sub(r"[aeiou]", "", token[1:])
    return compact if len(compact) >= 3 else token


def shorten_slug(slug: str, max_len: int = MAX_LEN) -> str:
    slug = slug.strip().lower()
    if len(slug) <= max_len:
        return slug

    tokens = [REPLACE.get(t, t) for t in slug.split("-") if t]
    filtered = [t for t in tokens if t not in DROP_TOKENS]
    if filtered:
        tokens = filtered

    candidate = "-".join(tokens)
    if len(candidate) <= max_len:
        return candidate

    compressed = [t if re.search(r"\d", t) else compress_token(t) for t in tokens]
    candidate = "-".join(compressed)
    if len(candidate) <= max_len:
        return candidate

    compact = compressed[:4]
    for t in compressed[4:]:
        probe = "-".join(compact + [t])
        if len(probe) <= max_len:
            compact.append(t)
        else:
            break

    return ("-".join(compact) or slug)[:max_len].strip("-")


def main() -> None:
    csv_path = Path(
        "/Users/mohdwaseem/Documents/waseem-document/my-workspace/feature-info/media/new-product/new_product_detail.csv"
    )

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("rows=0 changed=0 max_len=0 over_limit=0")
        return

    fieldnames = list(rows[0].keys())

    changed = 0
    for row in rows:
        source = (row.get("handleId") or "").strip().lower()
        if not source:
            continue
        target = shorten_slug(source, MAX_LEN)
        if target != source:
            changed += 1
        row["handleId"] = target

    # Keep handle IDs unique after shortening.
    seen: dict[str, int] = {}
    for row in rows:
        base = row["handleId"]
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count > 1:
            suffix = f"-{count}"
            if len(base) + len(suffix) > MAX_LEN:
                base = base[: MAX_LEN - len(suffix)].rstrip("-")
            row["handleId"] = f"{base}{suffix}"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    max_len = max(len(row.get("handleId", "")) for row in rows)
    over_limit = sum(1 for row in rows if len(row.get("handleId", "")) > MAX_LEN)
    print(f"rows={len(rows)} changed={changed} max_len={max_len} over_limit={over_limit}")


if __name__ == "__main__":
    main()
