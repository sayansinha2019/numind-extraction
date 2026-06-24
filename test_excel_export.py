"""Tests for JSON-to-Excel conversion across document extraction scenarios."""

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from excel_export import (
    _flatten_fields,
    _format_cell_value,
    _is_table,
    _sanitize_sheet_name,
    write_excel_from_extraction,
)
from extract import deep_merge_data, merge_strings


def sheet_to_dict(ws) -> dict[str, str]:
    """Read a two-column Field/Value sheet into a dictionary."""
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    return {row[0]: row[1] for row in rows if row[0] is not None}


def sheet_table(ws) -> list[list[object]]:
    """Read a worksheet as a list of rows."""
    return [list(row) for row in ws.iter_rows(values_only=True)]


class TestExcelHelpers(unittest.TestCase):
    def test_format_cell_value_types(self) -> None:
        self.assertEqual(_format_cell_value(None), "")
        self.assertEqual(_format_cell_value(True), "true")
        self.assertEqual(_format_cell_value(False), "false")
        self.assertEqual(_format_cell_value(42), "42")
        self.assertEqual(_format_cell_value(3.5), "3.5")
        self.assertEqual(_format_cell_value("hello"), "hello")
        self.assertEqual(_format_cell_value(["a", "b"]), "a, b")

    def test_is_table(self) -> None:
        self.assertTrue(_is_table([{"a": 1}, {"a": 2}]))
        self.assertFalse(_is_table([]))
        self.assertFalse(_is_table(["a", "b"]))
        self.assertFalse(_is_table([{"a": 1}, "bad"]))

    def test_flatten_nested_fields(self) -> None:
        rows = _flatten_fields(
            {
                "name": "John",
                "address": {"city": "NYC", "zip": "10001"},
                "tags": ["vip", "new"],
            }
        )
        self.assertEqual(
            dict(rows),
            {
                "name": "John",
                "address.city": "NYC",
                "address.zip": "10001",
                "tags": "vip, new",
            },
        )

    def test_flatten_skips_table_arrays(self) -> None:
        rows = _flatten_fields(
            {
                "invoice_id": "INV-1",
                "line_items": [{"item": "pen", "qty": 2}],
            }
        )
        self.assertEqual(dict(rows), {"invoice_id": "INV-1"})
        self.assertNotIn("line_items", dict(rows))

    def test_sanitize_sheet_name(self) -> None:
        used: set[str] = set()
        self.assertEqual(_sanitize_sheet_name("line_items", used), "line_items")
        self.assertEqual(_sanitize_sheet_name("line/items", used), "line_items_1")
        self.assertEqual(_sanitize_sheet_name("line_items", used), "line_items_2")


class TestExcelExportScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _export(self, extracted_data: dict, name: str = "doc.pdf") -> Path:
        excel_path = self.output_dir / "output.xlsx"
        write_excel_from_extraction(extracted_data, excel_path, source_name=name)
        return excel_path

    def _workbook(self, excel_path: Path):
        return load_workbook(excel_path, read_only=True, data_only=True)

    def test_simple_flat_document(self) -> None:
        excel_path = self._export(
            {
                "page_count": 1,
                "data": {"invoice_number": "INV-100", "total": 250.5},
                "pages": [
                    {
                        "page": 1,
                        "status": "ok",
                        "data": {"invoice_number": "INV-100", "total": 250.5},
                    }
                ],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        summary = sheet_to_dict(wb["Summary"])
        self.assertEqual(summary["source_file"], "doc.pdf")
        self.assertEqual(summary["page_count"], "1")
        self.assertEqual(summary["invoice_number"], "INV-100")
        self.assertEqual(summary["total"], "250.5")
        wb.close()

    def test_nested_fields_and_cross_page_merge(self) -> None:
        excel_path = self._export(
            {
                "page_count": 2,
                "data": {
                    "patient_name": "John Doe",
                    "address": "123 Main Street, NYC",
                    "policy": {"provider": "Acme", "id": "P-55"},
                },
                "pages": [
                    {
                        "page": 1,
                        "status": "ok",
                        "data": {
                            "patient_name": "John Doe",
                            "address": "123 Main",
                            "policy": {"provider": "Acme"},
                        },
                    },
                    {
                        "page": 2,
                        "status": "ok",
                        "data": {
                            "address": "Street, NYC",
                            "policy": {"id": "P-55"},
                        },
                    },
                ],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        summary = sheet_to_dict(wb["Summary"])
        self.assertEqual(summary["patient_name"], "John Doe")
        self.assertEqual(summary["address"], "123 Main Street, NYC")
        self.assertEqual(summary["policy.provider"], "Acme")
        self.assertEqual(summary["policy.id"], "P-55")

        pages = sheet_table(wb["Pages"])
        self.assertEqual(pages[0], ["Page", "Field", "Value"])
        self.assertIn([1, "address", "123 Main"], pages)
        self.assertIn([2, "policy.id", "P-55"], pages)
        wb.close()

    def test_top_level_table_sheet(self) -> None:
        excel_path = self._export(
            {
                "page_count": 1,
                "data": {
                    "invoice_number": "INV-200",
                    "line_items": [
                        {"description": "Widget", "qty": 2, "price": 10},
                        {"description": "Gadget", "qty": 1, "price": 25},
                    ],
                },
                "pages": [{"page": 1, "status": "ok", "data": {}}],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        self.assertIn("line_items", wb.sheetnames)
        table = sheet_table(wb["line_items"])
        self.assertEqual(table[0], ["description", "qty", "price"])
        self.assertEqual(table[1], ["Widget", "2", "10"])
        self.assertEqual(table[2], ["Gadget", "1", "25"])
        wb.close()

    def test_nested_table_sheet(self) -> None:
        excel_path = self._export(
            {
                "page_count": 1,
                "data": {
                    "invoice": {
                        "number": "INV-300",
                        "line_items": [
                            {"sku": "A1", "amount": 100},
                            {"sku": "B2", "amount": 50},
                        ],
                    }
                },
                "pages": [{"page": 1, "status": "ok", "data": {}}],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        summary = sheet_to_dict(wb["Summary"])
        self.assertEqual(summary["invoice.number"], "INV-300")
        self.assertIn("invoice.line_items", wb.sheetnames)

        table = sheet_table(wb["invoice.line_items"])
        self.assertEqual(table[0], ["sku", "amount"])
        self.assertEqual(table[1], ["A1", "100"])
        wb.close()

    def test_multiple_table_sheets(self) -> None:
        excel_path = self._export(
            {
                "page_count": 1,
                "data": {
                    "items": [{"name": "A"}],
                    "payments": [{"method": "card", "amount": 10}],
                },
                "pages": [],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        self.assertIn("items", wb.sheetnames)
        self.assertIn("payments", wb.sheetnames)
        wb.close()

    def test_failed_page_and_errors_sheet(self) -> None:
        excel_path = self._export(
            {
                "page_count": 2,
                "data": {"field_a": "value"},
                "pages": [
                    {"page": 1, "status": "ok", "data": {"field_a": "value"}},
                    {
                        "page": 2,
                        "status": "error",
                        "error": "GPU timeout",
                        "data": {},
                    },
                ],
                "errors": [
                    {"stage": "extraction", "page": 2, "error": "GPU timeout"},
                    {"stage": "template", "pages": [1], "error": "partial template issue"},
                ],
            }
        )
        wb = self._workbook(excel_path)
        self.assertIn("Errors", wb.sheetnames)

        pages = sheet_table(wb["Pages"])
        self.assertIn([2, "status", "error"], pages)
        self.assertIn([2, "error", "GPU timeout"], pages)

        errors = sheet_table(wb["Errors"])
        self.assertEqual(errors[1], ["extraction", "2", "GPU timeout"])
        self.assertEqual(errors[2], ["template", "1", "partial template issue"])
        wb.close()

    def test_empty_document_data(self) -> None:
        excel_path = self._export(
            {
                "page_count": 0,
                "data": {},
                "pages": [],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        summary = sheet_to_dict(wb["Summary"])
        self.assertEqual(summary["source_file"], "doc.pdf")
        self.assertEqual(summary["page_count"], "0")
        self.assertNotIn("Errors", wb.sheetnames)
        wb.close()

    def test_boolean_and_null_fields(self) -> None:
        excel_path = self._export(
            {
                "page_count": 1,
                "data": {
                    "active": True,
                    "archived": False,
                    "notes": None,
                },
                "pages": [],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        summary = sheet_to_dict(wb["Summary"])
        self.assertEqual(summary["active"], "true")
        self.assertEqual(summary["archived"], "false")
        self.assertIn(summary.get("notes"), ("", None))
        wb.close()

    def test_merge_then_excel_end_to_end(self) -> None:
        """Simulate multi-page extraction merge and verify Excel output."""
        page_one = {
            "vendor": "Acme Corp",
            "address": "123 Main",
            "line_items": [{"item": "Paper", "qty": 5}],
        }
        page_two = {
            "address": "Street, Boston",
            "line_items": [{"item": "Ink", "qty": 2}],
            "total": 99.5,
        }

        merged = deep_merge_data(deep_merge_data({}, page_one), page_two)
        self.assertEqual(merged["address"], merge_strings("123 Main", "Street, Boston"))

        excel_path = self._export(
            {
                "page_count": 2,
                "data": merged,
                "pages": [
                    {"page": 1, "status": "ok", "data": page_one},
                    {"page": 2, "status": "ok", "data": page_two},
                ],
                "errors": [],
            },
            name="invoice.pdf",
        )

        wb = self._workbook(excel_path)
        summary = sheet_to_dict(wb["Summary"])
        self.assertEqual(summary["vendor"], "Acme Corp")
        self.assertEqual(summary["address"], "123 Main Street, Boston")
        self.assertEqual(summary["total"], "99.5")

        table = sheet_table(wb["line_items"])
        self.assertEqual(len(table), 3)  # header + 2 rows
        self.assertEqual(table[1][0], "Paper")
        self.assertEqual(table[2][0], "Ink")
        wb.close()

    def test_table_cells_with_nested_values(self) -> None:
        excel_path = self._export(
            {
                "page_count": 1,
                "data": {
                    "rows": [
                        {"name": "A", "meta": {"color": "red"}},
                        {"name": "B", "meta": {"color": "blue"}},
                    ]
                },
                "pages": [],
                "errors": [],
            }
        )
        wb = self._workbook(excel_path)
        table = sheet_table(wb["rows"])
        self.assertEqual(table[0], ["name", "meta"])
        self.assertEqual(table[1][1], "{'color': 'red'}")
        wb.close()


if __name__ == "__main__":
    unittest.main()
