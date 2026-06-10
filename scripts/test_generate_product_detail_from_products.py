import tempfile
import unittest
from pathlib import Path

from scripts.generate_product_detail_from_products import (
    build_output_rows,
    read_input_rows,
    write_output_rows,
)


class GenerateProductDetailTests(unittest.TestCase):
    def test_build_output_rows_with_overrides(self) -> None:
        rows = [
            {
                "identifier": "1001073",
                "title": "Yahama Mt15 luggage carrier",
                "image_names": "1001073_1.jpg,1001073_2.jpg",
            }
        ]
        out = build_output_rows(rows, Path("/tmp/does-not-need-to-exist"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["brand"], "Yamaha")
        self.assertIn("MT-15", out[0]["name"])
        self.assertEqual(out[0]["additionalInfoTitle1"], "Product Specification")
        self.assertEqual(out[0]["additionalInfoTitle2"], "Features")

    def test_read_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_csv = tmp / "products.csv"
            out_csv = tmp / "product_detail.csv"
            in_csv.write_text(
                "identifier,title,image_names\n"
                "1001083,Meteor350 backrest,1001083_1.jpg\n",
                encoding="utf-8",
            )
            rows = read_input_rows(in_csv)
            out_rows = build_output_rows(rows, tmp)
            write_output_rows(out_csv, out_rows)
            content = out_csv.read_text(encoding="utf-8")
            self.assertIn("Royal Enfield Meteor 350 Backrest", content)
            self.assertIn("Product Specification", content)


if __name__ == "__main__":
    unittest.main()

