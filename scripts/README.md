# Adhoc Scripts

This folder contains one-off or operational scripts that do not belong in the core pipelines.

## WhatsApp Product Import Script

Script: `scripts/whatsapp_product_import.py`

### What it does

- Reads a WhatsApp export directory containing `_chat.txt` and image files.
- Groups messages into products using this pattern:
  - one or more `<attached: ...jpg>` messages
  - followed by one or more text messages used as product title
- Renames product images to `<identifier>_<index>.<ext>` where:
  - `identifier` starts from `WHATSAPP_PRODUCT_START_ID`
  - `index` starts from `1` for each product
- Creates a CSV with columns:
  - `identifier`
  - `title`
  - `image_names` (comma-separated renamed images)

### Configure

Copy `scripts/.env.example` to `scripts/.env` and update values.

### Run

```bash
python scripts/whatsapp_product_import.py
```

### Test

```bash
python -m unittest scripts/test_whatsapp_product_import.py -v
```

## Product Detail Generation Script

Script: `scripts/generate_product_detail_from_products.py`

### What it does

- Reads source rows from `product.csv`/`products.csv` with columns: `identifier,title,image_names`.
- Generates storefront content columns:
  - `brand`
  - `name`
  - `description`
  - `additionalInfoTitle1`
  - `additionalInfoDescription1`
  - `additionalInfoTitle2`
  - `additionalInfoDescription2`
- Writes `product_detail.csv` with source + generated content columns.

### Run

```bash
python scripts/generate_product_detail_from_products.py \
  --input /Users/mohdwaseem/Desktop/my-workspace/python-utility/debug/products.csv \
  --image-dir /Users/mohdwaseem/Desktop/my-workspace/python-utility/debug/whatsapp-chat \
  --output /Users/mohdwaseem/Desktop/my-workspace/python-utility/debug/product_detail.csv
```

### Test

```bash
python -m unittest scripts/test_generate_product_detail_from_products.py -v
```
