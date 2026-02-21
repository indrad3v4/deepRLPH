import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_inspector import summarize_dataset


def test_parquet_summary(tmp_path: Path) -> None:
    # Create a tiny parquet dataset
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [0.1, 0.5, 0.9],
    })
    path = tmp_path / "train.parquet"

    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

    summary = summarize_dataset(path)

    assert summary["format"] == "parquet"
    assert summary["row_count"] == 3
    assert summary["column_count"] == 2
    assert set(summary["columns"]) == {"id", "value"}
    assert "numeric_stats" in summary
    assert summary["sample"] and len(summary["sample"]) == 3
