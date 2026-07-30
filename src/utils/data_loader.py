"""
Data Loader - Data Loading and Metadata Extraction Tool

Features:
    Read CSV files and extract DataFrame metadata
    Provide data analysis input for Profiler Node
"""
import pandas as pd
from typing import Dict, Any
from pathlib import Path


def load_and_profile_data(csv_path: str) -> Dict[str, Any]:
    """
    Load CSV file and extract metadata

    Args:
        csv_path: CSV file path

    Returns:
        Dictionary containing DataFrame metadata, including:
        - shape: (rows, cols)
        - columns: list of column names
        - dtypes: column type dictionary
        - null_counts: missing value count per column
        - total_nulls: total missing values
        - sample_data: first 10 rows data string
        - memory_usage: memory consumption
        - duplicate_count: duplicate row count

    Raises:
        FileNotFoundError: File does not exist
        ValueError: File format error
    """
    # 1. Check if file exists
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {csv_path}")

    # 2. Read CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"[OK] Successfully loaded: {csv_path}")
        print(f"   Data shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")

    # 3. Extract metadata
    original_df_info = extract_df_metadata(df)

    return original_df_info


def extract_df_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract DataFrame metadata

    Args:
        df: Pandas DataFrame

    Returns:
        Metadata dictionary
    """
    # Basic information
    shape = df.shape
    columns = df.columns.tolist()

    # Data types
    dtypes = df.dtypes.astype(str).to_dict()

    # Missing value statistics
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())

    # Sample data (first 10 rows)
    sample_data = df.head(10).to_string()

    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    memory_mb = memory_usage / (1024 * 1024)

    # Duplicate row statistics
    duplicate_count = df.duplicated().sum()

    # Build metadata dictionary
    metadata = {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
        "sample_data": sample_data,
        "memory_usage": f"{memory_mb:.2f} MB",
        "duplicate_count": duplicate_count,
    }

    return metadata


def print_data_summary(df_info: Dict[str, Any]) -> None:
    """
    Print data summary information

    Args:
        df_info: DataFrame metadata dictionary
    """
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Data dimensions: {df_info['shape']}")
    print(f"Total columns: {len(df_info['columns'])}")
    print(f"Total missing values: {df_info['total_nulls']}")
    print(f"Duplicate rows: {df_info['duplicate_count']}")
    print(f"Memory usage: {df_info['memory_usage']}")

    print(f"\nColumn information:")
    for col in df_info['columns']:
        dtype = df_info['dtypes'][col]
        null_count = df_info['null_counts'][col]
        print(f"  - {col}: {dtype} (missing values: {null_count})")

    print("="*60)
