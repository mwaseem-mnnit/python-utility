"""Generate feature-info CSV for the first 5 product IDs in a media directory."""

from __future__ import annotations

import csv
from pathlib import Path

INPUT_DIR = Path(
    "/Users/mohdwaseem/Documents/waseem-document/my-workspace/feature-info/media/new-product/StoreNova-1"
)
PRODUCTS_CSV = INPUT_DIR.parent / "products.csv"
OUTPUT_CSV = INPUT_DIR / "feature_info_first5.csv"


def _spec_html(spec_pairs: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<tr><td><strong>{k} :</strong></td><td>{v}</td></tr>" for k, v in spec_pairs
    )
    return f'<div class="nei-table"><table><tbody>{rows}</tbody></table></div>'


def _features_html(name: str, description: str, bullets: list[str]) -> str:
    parts = [
        f'<p class="wmn-title-white">{name}</p>',
        f"<p>{description}</p>",
        "<p><strong>Key Features:</strong></p>",
    ]
    parts.extend([f"<p>- {item}</p>" for item in bullets])
    return "".join(parts)


def main() -> int:
    ids = sorted({p.stem.split("_")[0] for p in INPUT_DIR.glob("*.jpg") if "_" in p.stem})[:5]

    # Keep copy conservative; values are derived from listing titles and intended for review.
    content = {
        "1001001": {
            "brand": "StoreNova",
            "name": "Owl Light Set (2 Pc) with On/Off Switch",
            "description": "Owl-style motorcycle auxiliary light set designed to improve front visibility and styling. Supplied as a pair with an on/off switch for quick control.",
            "spec": [
                ("Model Name", "Owl Light Set (as per listing title)"),
                ("Category", "Motorcycle auxiliary light"),
                ("Pack Size", "Set of 2 pieces"),
                ("Switch", "On/Off switch included"),
                (
                    "Vehicle Compatibility",
                    "Universal fit for most motorcycles (check bracket/space before purchase)",
                ),
            ],
            "features": [
                "Pair-pack format for balanced left-right mounting.",
                "On/off switch support for easy operation during rides.",
                "Good choice for custom front look and practical night use.",
            ],
        },
        "1001002": {
            "brand": "StoreNova",
            "name": "Jet Indicator Metal Set (4 Pc)",
            "description": "Metal-body jet style indicators for motorcycle turn signaling upgrades. This pack includes four pieces suitable for front and rear replacement.",
            "spec": [
                ("Model Name", "Jet Indicator Metal (as per listing title)"),
                ("Category", "Motorcycle indicator lights"),
                ("Body Material", "Metal"),
                ("Pack Size", "Set of 4 pieces"),
                ("Position", "Front and rear indicator replacement"),
            ],
            "features": [
                "Complete 4-piece set for both sides of the bike.",
                "Metal housing gives a premium aftermarket look.",
                "Suitable for replacing old or damaged indicators.",
            ],
        },
        "1001003": {
            "brand": "StoreNova",
            "name": "3 LED Shutter Light (Single Pc) White/Yellow with Flash",
            "description": "Compact 3-LED shutter style light in dual-color white/yellow output with flashing mode. Ideal for auxiliary visibility and custom lighting setups.",
            "spec": [
                ("Model Name", "3 LED Shutter Light"),
                ("Color Mode", "White and Yellow"),
                ("Lighting Effect", "Flash mode supported"),
                ("Pack Size", "Single piece"),
                ("Use Case", "Auxiliary lighting / visibility enhancement"),
            ],
            "features": [
                "Dual-color output helps adapt to different riding conditions.",
                "Flash mode improves noticeability in traffic.",
                "Single-piece format for focused custom installation.",
            ],
        },
        "1001004": {
            "brand": "StoreNova",
            "name": "8 LED Square Fog Light White/Yellow with Flash",
            "description": "Square-form fog light with 8 LEDs, dual white/yellow color mode, and flash function. Built for improved road illumination and weather-focused visibility.",
            "spec": [
                ("Model Name", "8 LED Square Fog Light"),
                ("LED Count", "8 LEDs"),
                ("Color Mode", "White and Yellow"),
                ("Lighting Effect", "Flash mode supported"),
                ("Application", "Motorcycle fog/auxiliary lighting"),
            ],
            "features": [
                "8-LED layout for strong and consistent light spread.",
                "Dual white/yellow output for day and low-visibility use.",
                "Square housing complements modern custom builds.",
            ],
        },
        "1001005": {
            "brand": "Yamaha",
            "name": "XSR Main Stand",
            "description": "Main stand for Yamaha XSR with stable center-lift support for parking, cleaning, chain maintenance, and daily convenience.",
            "spec": [
                ("Model Name", "XSR Main Stand"),
                ("Brand", "Yamaha (as per listing title)"),
                ("Category", "Motorcycle stand"),
                (
                    "Fitment",
                    "For Yamaha XSR platform (confirm exact variant before purchase)",
                ),
                ("Use", "Center support for parking and maintenance"),
            ],
            "features": [
                "Improves parking stability on flat surfaces.",
                "Helpful for chain lubrication and wheel-cleaning routines.",
                "Practical utility upgrade for regular riders.",
            ],
        },
    }

    rows: list[dict[str, str]] = []
    for pid in ids:
        entry = content[pid]
        rows.append(
            {
                "identifier": pid,
                "brand": entry["brand"],
                "name": entry["name"],
                "description": entry["description"],
                "additionalInfoTitle1": "Product Specification",
                "additionalInfoDescription1": _spec_html(entry["spec"]),
                "additionalInfoTitle2": "Features",
                "additionalInfoDescription2": _features_html(
                    entry["name"], entry["description"], entry["features"]
                ),
            }
        )

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

    print(f"Wrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

