"""Update product detail CSV with handleId and collection columns."""

from __future__ import annotations

import csv
import re
from pathlib import Path

CSV_PATH = Path(
    "/Users/mohdwaseem/Documents/waseem-document/my-workspace/feature-info/media/new-product/new_product_detail.csv"
)


def normalize_name(name: str) -> str:
    text = re.sub(r"\s+", " ", (name or "").strip())
    if len(text) <= 80:
        return text

    # Remove low-value words first while preserving product intent.
    drop_words = {
        "for",
        "all",
        "and",
        "our",
        "bike",
        "scooty",
        "fitting",
        "universal",
        "available",
        "with",
    }
    words = [w for w in text.split(" ") if w.lower() not in drop_words]
    compact = " ".join(words)
    if len(compact) <= 80:
        return compact

    # If still long, keep the most meaningful leading segment at word boundary.
    truncated = compact[:80]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip(" -,:;")


def slugify(text: str) -> str:
    value = (text or "").lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[+/]", " ", value)
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"\s+", "-", value).strip("-")
    value = re.sub(r"-+", "-", value)
    return value or "product"


def infer_collection(name: str, description: str = "") -> str:
    text = f"{name} {description}".lower()

    if any(k in text for k in ["hand gurd", "hand guard", "hand gurad", "knuckle", "lever gurad", "lever gurd"]):
        return "Hand Guards"

    if any(k in text for k in ["bend pipe", "exhaust"]):
        if "stainless" in text:
            return "Stainless Bend Pipes"
        return "Bend Pipes"

    if any(k in text for k in ["headlight", "h4", "bulb"]):
        return "LED Headlights"

    if "projector" in text and "fog" in text:
        return "Projector Fog Lights"

    if any(k in text for k in ["indicator", "indigetr"]):
        if "sequential" in text:
            return "Sequential Indicators"
        return "LED Indicators"

    if any(k in text for k in ["fog light", "led lens", "biscuit fog", "4x4 light", "g-16", "g16", "gorrilla fog"]):
        if "yellow" in text:
            return "Yellow Fog Lights"
        if "mini" in text:
            return "Mini Fog Lights"
        return "LED Fog Lights"

    if any(k in text for k in ["mirror", "mirrer", "miirer", "morrer"]):
        return "Mirrors"

    if any(k in text for k in ["horn", "siren", "wintone", "mocc"]):
        if "dual" in text or "tone" in text:
            return "Dual Tone Horns"
        if "air" in text or "train" in text or "pipe horn" in text:
            return "Air Horns"
        return "Horns"

    if "switch" in text and not any(k in text for k in ["light", "fog", "headlight", "bulb"]):
        return "Switches"

    if any(k in text for k in ["charger", "camera stnd", "stand", "holder"]):
        return "Mobile Holders"

    if any(k in text for k in ["light", "led"]):
        return "Auxiliary Lights"

    return "Body Parts"


def main() -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        current_fields = reader.fieldnames or []

    seen_handles: dict[str, int] = {}
    for row in rows:
        name = normalize_name(row.get("name", ""))
        row["name"] = name

        base_handle = slugify(name)
        count = seen_handles.get(base_handle, 0) + 1
        seen_handles[base_handle] = count
        row["handleId"] = base_handle if count == 1 else f"{base_handle}-{count}"

        row["collection"] = infer_collection(name, row.get("description", ""))

    out_fields = [f for f in current_fields if f not in {"handleId", "collection"}]
    out_fields.extend(["handleId", "collection"])

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    max_name = max((len((r.get("name") or "")) for r in rows), default=0)
    print(f"Updated: {CSV_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Max name length: {max_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

