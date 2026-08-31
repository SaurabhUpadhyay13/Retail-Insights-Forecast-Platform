from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from column_standardizer import COLUMN_ALIASES, _normalize, standardize_columns


class ColumnStandardizerTests(unittest.TestCase):
    def test_common_retail_fields_map_without_collision(self) -> None:
        source = pd.DataFrame(
            columns=["UnitPrice", "Selliing Price", "Brand Name", "Store", "Item"]
        )
        result = standardize_columns(source, verbose=False)
        self.assertEqual(
            list(result.columns),
            ["price_per_unit", "selling_price", "brand", "location", "item"],
        )

    def test_aliases_are_unique_across_canonical_fields(self) -> None:
        owners: dict[str, str] = {}
        collisions: list[str] = []
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                normalized = _normalize(alias)
                previous = owners.setdefault(normalized, canonical)
                if previous != canonical:
                    collisions.append(f"{normalized}: {previous}/{canonical}")
        self.assertEqual(collisions, [])


if __name__ == "__main__":
    unittest.main()
