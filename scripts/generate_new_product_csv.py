"""Generate new_product.csv for StoreNova-1 products starting from ID 1001006."""

from __future__ import annotations

import csv
import re
from pathlib import Path

INPUT_DIR = Path(
    "/Users/mohdwaseem/Documents/waseem-document/my-workspace/feature-info/media/new-product/StoreNova-1"
)
SOURCE_CSV = INPUT_DIR.parent / "products.csv"
OUTPUT_CSV = INPUT_DIR / "new_product.csv"
MIN_ID = 1001006


def clean_title(raw: str, pid: str) -> str:
    text = raw or f"Product {pid}"
    text = re.sub(r"<This message was edited>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bproduct\s*(no\.?|number)?\s*\d*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -:")
    return text or f"Product {pid}"


def detect_brand(name: str) -> str:
    value = name.lower()
    if "yamaha" in value:
        return "Yamaha"
    if "honda" in value:
        return "Honda"
    if "bajaj" in value:
        return "Bajaj"
    if "tvs" in value:
        return "TVS"
    if "ktm" in value:
        return "KTM"
    if "royal enfield" in value:
        return "Royal Enfield"
    return "StoreNova"


def infer_pack_size(title: str) -> str:
    match = re.search(r"set\s*of\s*(\d+)", title, flags=re.IGNORECASE)
    if match:
        return f"Set of {match.group(1)} pieces"
    lowered = title.lower()
    if "single" in lowered or "1 pc" in lowered or "1pc" in lowered:
        return "Single piece"
    return "As per listing"


def infer_category(title: str) -> str:
    lowered = title.lower()
    if "fog" in lowered or "led" in lowered or "light" in lowered:
        return "Motorcycle lighting accessory"
    if "indicator" in lowered:
        return "Motorcycle indicator"
    if "mirrer" in lowered or "mirror" in lowered:
        return "Motorcycle mirror"
    if "stand" in lowered:
        return "Motorcycle stand"
    if "switch" in lowered:
        return "Motorcycle electrical accessory"
    return "Motorcycle accessory"


def to_spec_html(name: str, source_title: str) -> str:
    specs = [
        ("Model Name", name),
        ("Category", infer_category(source_title)),
        ("Pack Size", infer_pack_size(source_title)),
        (
            "Fitment",
            "Universal / model-specific as per listing; verify compatibility before purchase",
        ),
        ("Condition", "Aftermarket new"),
    ]
    rows = "".join(
        f"<tr><td><strong>{key} :</strong></td><td>{value}</td></tr>" for key, value in specs
    )
    return f'<div class="nei-table"><table><tbody>{rows}</tbody></table></div>'


def to_features_html(name: str, description: str) -> str:
    bullets = [
        "Designed for practical daily use and clean aftermarket appearance.",
        "Built for easy integration with common motorcycle setups.",
        "Please confirm fitment and electrical/mechanical compatibility before order.",
    ]
    bullet_rows = "".join(f"<p>- {item}</p>" for item in bullets)
    return (
        f'<p class="wmn-title-white">{name}</p>'
        f"<p>{description}</p>"
        "<p><strong>Key Features:</strong></p>"
        f"{bullet_rows}"
    )


def load_titles() -> dict[str, str]:
    titles: dict[str, str] = {}
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pid = (row.get("identifier") or "").strip()
            title = (row.get("title") or "").strip()
            if pid:
                titles[pid] = title
    return titles


def target_ids() -> list[str]:
    ids = sorted({p.stem.split("_")[0] for p in INPUT_DIR.glob("*.jpg") if "_" in p.stem})
    return [pid for pid in ids if pid.isdigit() and int(pid) >= MIN_ID]


def generate_rows(ids: list[str], titles: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pid in ids:
        source_title = titles.get(pid, f"Product {pid}")
        name = clean_title(source_title, pid)
        brand = detect_brand(name)
        description = (
            f"{name} from {brand} is an aftermarket motorcycle accessory designed for utility, style, "
            "and dependable daily use. Please verify fitment details with your vehicle model before purchase."
        )
        rows.append(
            {
                "identifier": pid,
                "brand": brand,
                "name": name,
                "description": description,
                "additionalInfoTitle1": "Product Specification",
                "additionalInfoDescription1": to_spec_html(name, source_title),
                "additionalInfoTitle2": "Features",
                "additionalInfoDescription2": to_features_html(name, description),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "identifier",
        "brand",
        "name",
        "description",
        "additionalInfoTitle1",
        "additionalInfoDescription1",
        "additionalInfoTitle2",
        "additionalInfoDescription2",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ids = target_ids()
    titles = load_titles()
    rows = generate_rows(ids, titles)
    write_csv(rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Rows: {len(rows)}")
    if rows:
        print(f"Range: {rows[0]['identifier']} -> {rows[-1]['identifier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

