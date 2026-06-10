#!/usr/bin/env python3
"""Generate product_detail.csv from product.csv/products.csv title data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_INPUT = Path("/Users/mohdwaseem/Desktop/my-workspace/python-utility/debug/products.csv")
DEFAULT_OUTPUT = Path("/Users/mohdwaseem/Desktop/my-workspace/python-utility/debug/product_detail.csv")
DEFAULT_IMAGE_DIR = Path("/Users/mohdwaseem/Desktop/my-workspace/python-utility/debug/whatsapp-chat")

# Curated metadata to keep naming clean and consistent for storefront use.
PRODUCT_OVERRIDES: Dict[str, Dict[str, str]] = {
    "1001072": {
        "brand": "Hero",
        "name": "Hero Xtreme 125R Luggage Carrier",
        "fitment": "Hero Xtreme 125R",
        "type": "luggage_carrier",
    },
    "1001073": {
        "brand": "Yamaha",
        "name": "Yamaha MT-15 Luggage Carrier",
        "fitment": "Yamaha MT-15",
        "type": "luggage_carrier",
    },
    "1001074": {
        "brand": "Bajaj",
        "name": "Bajaj Pulsar N160/N250/F250 Luggage Carrier",
        "fitment": "Bajaj Pulsar N160, N250, F250",
        "type": "luggage_carrier",
    },
    "1001075": {
        "brand": "Royal Enfield",
        "name": "Royal Enfield Hunter 350 Luggage Carrier",
        "fitment": "Royal Enfield Hunter 350",
        "type": "luggage_carrier",
    },
    "1001076": {
        "brand": "Royal Enfield",
        "name": "Royal Enfield Classic 350 Reborn Luggage Carrier",
        "fitment": "Royal Enfield Classic 350 Reborn",
        "type": "luggage_carrier",
    },
    "1001077": {
        "brand": "Yamaha",
        "name": "Yamaha FZ V3/V4 Luggage Carrier",
        "fitment": "Yamaha FZ V3 and V4",
        "type": "luggage_carrier",
    },
    "1001078": {
        "brand": "Yamaha",
        "name": "Yamaha MT Series Luggage Carrier",
        "fitment": "Yamaha MT series",
        "type": "luggage_carrier",
    },
    "1001079": {
        "brand": "TVS",
        "name": "TVS Apache RTR 160/180/200 Luggage Carrier",
        "fitment": "TVS Apache RTR 160, 180, 200",
        "type": "luggage_carrier",
    },
    "1001080": {
        "brand": "TVS",
        "name": "TVS Apache RTR 200 Luggage Carrier",
        "fitment": "TVS Apache RTR 200",
        "type": "luggage_carrier",
    },
    "1001081": {
        "brand": "Bajaj",
        "name": "Bajaj Pulsar 150/180/220 Luggage Carrier",
        "fitment": "Bajaj Pulsar 150, 180, 220",
        "type": "luggage_carrier",
    },
    "1001082": {
        "brand": "Bajaj",
        "name": "Bajaj Pulsar/Dominar Luggage Carrier",
        "fitment": "Bajaj Pulsar and Dominar",
        "type": "luggage_carrier",
    },
    "1001083": {
        "brand": "Royal Enfield",
        "name": "Royal Enfield Meteor 350 Backrest",
        "fitment": "Royal Enfield Meteor 350",
        "type": "backrest",
    },
    "1001084": {
        "brand": "Royal Enfield",
        "name": "Royal Enfield Classic 350 Reborn Backrest",
        "fitment": "Royal Enfield Classic 350 Reborn",
        "type": "backrest",
    },
    "1001085": {
        "brand": "Royal Enfield",
        "name": "Royal Enfield Classic 350 Reborn Metal Backrest",
        "fitment": "Royal Enfield Classic 350 Reborn",
        "type": "backrest",
    },
}


def _guess_brand(title: str) -> str:
    lowered = title.lower()
    if "yamaha" in lowered or "yahama" in lowered:
        return "Yamaha"
    if "hero" in lowered:
        return "Hero"
    if "bajaj" in lowered or "pulsar" in lowered or "dominar" in lowered:
        return "Bajaj"
    if "tvs" in lowered or "apache" in lowered or "rtr" in lowered:
        return "TVS"
    if any(token in lowered for token in ("hunter", "meteor", "reborn", "classic 350")):
        return "Royal Enfield"
    return "Generic"


def _guess_type(title: str) -> str:
    return "backrest" if "backrest" in title.lower() else "luggage_carrier"


def _clean_title(title: str) -> str:
    out = " ".join(title.strip().split())
    token_fixes = {
        "yahama": "Yamaha",
        "tvs": "TVS",
        "carrie": "carrier",
    }
    fixed_tokens = []
    for token in out.split(" "):
        key = token.lower()
        if key in token_fixes:
            fixed_tokens.append(token_fixes[key])
        else:
            fixed_tokens.append(token)
    return " ".join(fixed_tokens)


def _build_description(name: str, fitment: str, product_type: str) -> str:
    if product_type == "backrest":
        return (
            f"{name} is a durable motorcycle back support accessory designed for {fitment}. "
            "Built from strong metal with a clean finish, it improves pillion comfort on daily rides "
            "and long-distance touring while keeping a secure and stable fit on the bike."
        )
    return (
        f"{name} is a heavy-duty rear carrier designed for {fitment}. "
        "It is built to carry luggage safely on city commutes and touring rides, with a strong metal "
        "structure and stable mounting points for reliable day-to-day use."
    )


def _build_spec_html(name: str, fitment: str, product_type: str) -> str:
    usage = "Pillion support" if product_type == "backrest" else "Luggage support"
    return (
        "<div class=\"nei-table\"><table><tbody>"
        f"<tr><td><strong>Model Name :</strong></td><td>{name}</td></tr>"
        f"<tr><td><strong>Compatible Models :</strong></td><td>{fitment}</td></tr>"
        "<tr><td><strong>Material :</strong></td><td>Heavy-duty metal</td></tr>"
        "<tr><td><strong>Finish :</strong></td><td>Black powder-coated / painted</td></tr>"
        f"<tr><td><strong>Primary Use :</strong></td><td>{usage}</td></tr>"
        "<tr><td><strong>Installation :</strong></td><td>Bike-specific mounting</td></tr>"
        "<tr><td><strong>Pack Size :</strong></td><td>1 Piece</td></tr>"
        "</tbody></table></div>"
    )


def _build_feature_html(name: str, fitment: str, product_type: str) -> str:
    if product_type == "backrest":
        points = [
            "Strong metal construction for long service life.",
            "Improves pillion comfort during longer rides.",
            "Stable and secure mounting design.",
            "Designed to match bike styling with a clean finish.",
            "Suitable for regular commute and touring use.",
        ]
    else:
        points = [
            "Strong metal frame built for daily use.",
            "Helps carry bags and travel essentials safely.",
            "Stable mounting points reduce vibration during rides.",
            "Bike-specific fitment for better alignment.",
            "Durable surface finish for improved corrosion resistance.",
        ]
    points_html = "".join(f"<li>{point}</li>" for point in points)
    return (
        f"<p class=\"wmn-title-white\">{name}</p>"
        f"<p>{name} is designed for {fitment}, offering reliable performance and durable construction.</p>"
        "<p><strong>Key Features:</strong></p>"
        f"<ul>{points_html}</ul>"
    )


def build_output_rows(input_rows: List[Dict[str, str]], image_dir: Path) -> List[Dict[str, str]]:
    output_rows: List[Dict[str, str]] = []
    for row in input_rows:
        identifier = (row.get("identifier") or "").strip()
        title = (row.get("title") or "").strip()
        image_names = (row.get("image_names") or "").strip()
        if not identifier:
            continue

        override = PRODUCT_OVERRIDES.get(identifier)
        cleaned_title = _clean_title(title)

        brand = override["brand"] if override else _guess_brand(cleaned_title)
        name = override["name"] if override else cleaned_title.title()
        fitment = override["fitment"] if override else cleaned_title
        product_type = override["type"] if override else _guess_type(cleaned_title)

        # Keep generation robust even if image data has missing/incorrect names.
        for image_name in [item.strip() for item in image_names.split(",") if item.strip()]:
            if not (image_dir / image_name).exists():
                break

        description = _build_description(name, fitment, product_type)
        output_rows.append(
            {
                "identifier": identifier,
                "title": cleaned_title,
                "image_names": image_names,
                "brand": brand,
                "name": name,
                "description": description,
                "additionalInfoTitle1": "Product Specification",
                "additionalInfoDescription1": _build_spec_html(name, fitment, product_type),
                "additionalInfoTitle2": "Features",
                "additionalInfoDescription2": _build_feature_html(name, fitment, product_type),
            }
        )
    return output_rows


def read_input_rows(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_output_rows(output_csv: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "identifier",
        "title",
        "image_names",
        "brand",
        "name",
        "description",
        "additionalInfoTitle1",
        "additionalInfoDescription1",
        "additionalInfoTitle2",
        "additionalInfoDescription2",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate product_detail.csv from product title input.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Image directory path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input
    if not input_csv.exists():
        alt = input_csv.parent / "products.csv"
        if alt.exists():
            input_csv = alt
        else:
            raise FileNotFoundError(f"Input CSV not found: {args.input}")

    input_rows = read_input_rows(input_csv)
    output_rows = build_output_rows(input_rows, args.image_dir)
    write_output_rows(args.output, output_rows)
    print(f"Generated {len(output_rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


