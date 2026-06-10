from __future__ import annotations

import csv
import logging
import tempfile
import unittest
from pathlib import Path

from scripts.whatsapp_product_import import process_whatsapp_export


class WhatsAppProductImportTests(unittest.TestCase):
    def test_renames_images_and_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "00000008-PHOTO-2026-05-31-11-45-23.jpg").write_bytes(b"a")
            (input_dir / "00000009-PHOTO-2026-05-31-11-45-23.jpg").write_bytes(b"b")
            (input_dir / "00000010-PHOTO-2026-05-31-11-45-50.jpg").write_bytes(b"c")

            chat = "\n".join(
                [
                    "[31/05/26, 11:45:23 AM] Mazhar Bhai: <attached: 00000008-PHOTO-2026-05-31-11-45-23.jpg>",
                    "[31/05/26, 11:45:23 AM] Mazhar Bhai: <attached: 00000009-PHOTO-2026-05-31-11-45-23.jpg>",
                    "[31/05/26, 11:45:40 AM] Mazhar Bhai: Alloy Wheel 17 inch",
                    "[31/05/26, 11:46:10 AM] Mazhar Bhai: <attached: 00000010-PHOTO-2026-05-31-11-45-50.jpg>",
                    "[31/05/26, 11:46:30 AM] Mazhar Bhai: Black crash guard",
                    "[31/05/26, 11:46:31 AM] Mazhar Bhai: Heavy duty",
                ]
            )
            (input_dir / "_chat.txt").write_text(chat, encoding="utf-8")

            logger = logging.getLogger("whatsapp-import-test")
            processed, rows = process_whatsapp_export(
                input_dir=input_dir,
                output_dir=input_dir,
                product_start_id=300,
                chat_filename="_chat.txt",
                output_csv_name="products.csv",
                logger=logger,
            )

            self.assertEqual(processed, 2)
            self.assertEqual(rows, 2)

            self.assertTrue((input_dir / "300_1.jpg").exists())
            self.assertTrue((input_dir / "300_2.jpg").exists())
            self.assertTrue((input_dir / "301_1.jpg").exists())

            with (input_dir / "products.csv").open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                out_rows = list(reader)

            self.assertEqual(out_rows[0]["identifier"], "300")
            self.assertEqual(out_rows[0]["title"], "Alloy Wheel 17 inch")
            self.assertEqual(out_rows[0]["image_names"], "300_1.jpg,300_2.jpg")
            self.assertEqual(out_rows[1]["identifier"], "301")
            self.assertEqual(out_rows[1]["title"], "Black crash guard Heavy duty")
            self.assertEqual(out_rows[1]["image_names"], "301_1.jpg")


if __name__ == "__main__":
    unittest.main()

