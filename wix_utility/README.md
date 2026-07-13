# Wix Utility

Boilerplate for publishing prepared catalog data to Wix Stores.

## Setup

1. Copy `wix_utility/.env.example` to `wix_utility/.env`.
2. Fill in `WIX_API_KEY`, `WIX_SITE_ID`, and input paths.
3. Install dependencies:

```bash
python3 -m pip install -r wix_utility/requirements.txt
```

## Commands

```bash
python3 -m wix_utility healthcheck
python3 -m wix_utility dry-run
python3 -m wix_utility parse-csv
python3 -m wix_utility create-collections
python3 -m wix_utility assign-collection-to-product
python3 -m wix_utility collection-products-to-cms
python3 -m wix_utility run-flow
python3 -m wix_utility collection-sync
python3 -m wix_utility product-sync
python3 -m wix_utility media-upload
```

`WIX_DRY_RUN=true` is the default so early jobs log intended API calls without mutating Wix.
`run-flow` reads `WIX_FLOW`; valid values are `catalog-sync`, `collection-sync`,
`product-sync`, and `media-upload`.

For `create-collections`, set `WIX_DRY_RUN=false` so the job can read existing
Wix collections. Leave `WIX_COLLECTION_CREATE_ENABLED=false` to print a safe
create/skip plan without creating anything.

## Structure

- `core/config.py` loads `wix_utility/.env` and exposes `WixConfig`.
- `clients/wix_api.py` centralizes headers, retries, JSON decoding, and dry-run behavior.
- `io/csv_records.py` parses CSV rows into JSON-friendly dictionaries and can write JSON output.
- `flows/` contains class-based no-mutation flow runners for CSV, collection, product, media, and catalog sync jobs.
- `jobs/` contains class-based CLI jobs extending `WixJob`.
- `catalog/matching.py` compares product or collection-like objects to help avoid duplicates.
- `services/collections.py` contains collection/category payload mapping.
- `services/products.py` contains product payload mapping.
- `services/cms.py` contains Wix Data bulk save helpers.
- `services/media.py` contains image upload preparation placeholders.
- `catalog/models.py` contains local draft models that can be filled from CSV or WhatsApp parsing outputs.
