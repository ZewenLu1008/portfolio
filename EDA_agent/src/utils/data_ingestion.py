"""
Data Ingestion Module - Multi-Source Heterogeneous Data Loader

Responsibilities:
    Load and merge data from multiple file formats (CSV, Excel, PDF)
    Handle schema misalignments and data extraction from diverse sources
    Support both native and scanned PDFs via OCR fallback
"""
import os
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import pdfplumber

# OCR dependencies (imported conditionally to allow graceful degradation)
try:
    from pdf2image import convert_from_path
    from img2table.document import Image
    from img2table.ocr import TesseractOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning(
        "OCR dependencies not available. Scanned PDF support disabled. "
        "Install with: pip install pdf2image img2table pytesseract"
    )


def load_and_merge_data(directory_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and merge data from all CSV, Excel, and PDF files in a directory

    Supports:
    - CSV files: loaded via pd.read_csv()
    - Excel files: all sheets loaded and concatenated
    - PDF files: tables extracted via pdfplumber

    Args:
        directory_path: Path to directory containing data files

    Returns:
        Tuple of (merged_dataframe, metadata_dict)
        metadata includes source file info and warnings

    Raises:
        ValueError: If directory is empty or no valid data found
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Storage for loaded dataframes
    dataframes = []
    metadata = {
        "source_files": [],
        "warnings": [],
        "file_counts": {"csv": 0, "excel": 0, "pdf": 0},
        "total_files_processed": 0
    }

    # Process all files in directory
    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file():
            continue

        file_ext = file_path.suffix.lower()
        file_name = file_path.name

        try:
            if file_ext == ".csv":
                df = load_csv(file_path)
                if df is not None:
                    dataframes.append(df)
                    metadata["source_files"].append({
                        "name": file_name,
                        "type": "CSV",
                        "shape": df.shape,
                        "columns": df.columns.tolist()
                    })
                    metadata["file_counts"]["csv"] += 1

            elif file_ext in [".xlsx", ".xls"]:
                dfs = load_excel(file_path)
                for sheet_name, df in dfs:
                    if df is not None:
                        dataframes.append(df)
                        metadata["source_files"].append({
                            "name": f"{file_name} (sheet: {sheet_name})",
                            "type": "Excel",
                            "shape": df.shape,
                            "columns": df.columns.tolist()
                        })
                metadata["file_counts"]["excel"] += 1

            elif file_ext == ".pdf":
                dfs = load_pdf(file_path)
                for page_num, df in dfs:
                    if df is not None:
                        dataframes.append(df)
                        metadata["source_files"].append({
                            "name": f"{file_name} (page {page_num})",
                            "type": "PDF",
                            "shape": df.shape,
                            "columns": df.columns.tolist()
                        })
                metadata["file_counts"]["pdf"] += 1

        except Exception as e:
            warning_msg = f"Failed to load {file_name}: {str(e)}"
            metadata["warnings"].append(warning_msg)
            warnings.warn(warning_msg)

    # Validate we have data
    if not dataframes:
        raise ValueError(f"No valid data files found in directory: {directory_path}")

    metadata["total_files_processed"] = sum(metadata["file_counts"].values())

    # Merge all dataframes
    print(f"\n[Data Ingestion] Found {len(dataframes)} data sources")
    print(f"  - CSV files: {metadata['file_counts']['csv']}")
    print(f"  - Excel files: {metadata['file_counts']['excel']}")
    print(f"  - PDF files: {metadata['file_counts']['pdf']}")

    merged_df = merge_dataframes(dataframes, metadata)

    print(f"[Data Ingestion] Merged shape: {merged_df.shape}")
    print(f"[Data Ingestion] Final columns: {merged_df.columns.tolist()}")

    return merged_df, metadata


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load CSV file with robust encoding detection

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame or None if failed
    """
    try:
        # Try UTF-8 with BOM first (common Windows export format)
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"  [CSV] Loaded {file_path.name}: {df.shape}")
        return df
    except UnicodeDecodeError:
        # Fallback to latin-1 for legacy files
        try:
            df = pd.read_csv(file_path, encoding="latin-1")
            print(f"  [CSV] Loaded {file_path.name} (latin-1 encoding): {df.shape}")
            return df
        except Exception as e:
            print(f"  [CSV] Failed to load {file_path.name}: {str(e)}")
            return None
    except Exception as e:
        print(f"  [CSV] Failed to load {file_path.name}: {str(e)}")
        return None


def load_excel(file_path: Path) -> List[Tuple[str, pd.DataFrame]]:
    """
    Load all sheets from Excel file

    Args:
        file_path: Path to Excel file

    Returns:
        List of (sheet_name, dataframe) tuples
    """
    results = []
    try:
        # Load all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        print(f"  [Excel] Loading {file_path.name} ({len(sheet_names)} sheets)")

        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    print(f"    - Sheet '{sheet_name}': {df.shape}")
                    results.append((sheet_name, df))
            except Exception as e:
                print(f"    - Failed to load sheet '{sheet_name}': {str(e)}")

    except Exception as e:
        print(f"  [Excel] Failed to load {file_path.name}: {str(e)}")

    return results


def extract_table_from_image(image_path: Path, page_num: int) -> Optional[pd.DataFrame]:
    """
    Extract table from scanned PDF page using OCR

    Uses img2table library which combines OCR with table structure detection.
    Supports both Tesseract and other OCR backends.

    Args:
        image_path: Path to the page image
        page_num: Page number (for logging)

    Returns:
        DataFrame if table extraction succeeds, None otherwise
    """
    if not OCR_AVAILABLE:
        logging.warning(f"    - Page {page_num}: OCR dependencies not available, skipping")
        return None

    try:
        # Initialize Tesseract OCR engine
        # img2table supports multiple OCR backends: Tesseract, EasyOCR, PaddleOCR
        ocr = TesseractOCR(n_threads=1, lang="eng")

        # Load image and extract tables
        img = Image(str(image_path), detect_rotation=False)
        extracted_tables = img.extract_tables(
            ocr=ocr,
            implicit_rows=True,  # Detect rows without explicit borders
            borderless_tables=True,  # Detect tables without borders
            min_confidence=50  # Minimum OCR confidence threshold (0-100)
        )

        if not extracted_tables:
            logging.info(f"    - Page {page_num}: No tables detected via OCR")
            return None

        # img2table returns a list of ExtractedTable objects
        # Each has a .df attribute containing the pandas DataFrame
        for table_idx, table_obj in enumerate(extracted_tables):
            df = table_obj.df

            if df is not None and not df.empty:
                # Clean up: remove completely empty rows/columns
                df = df.dropna(how='all', axis=0)
                df = df.dropna(how='all', axis=1)

                if not df.empty:
                    print(f"    - Page {page_num}, Table {table_idx + 1} (OCR): {df.shape}")
                    return df

        return None

    except Exception as e:
        logging.warning(f"    - Page {page_num}: OCR extraction failed: {str(e)}")
        return None


def load_pdf(file_path: Path) -> List[Tuple[int, pd.DataFrame]]:
    """
    Extract tables from PDF file with fallback strategy for scanned PDFs

    Strategy:
    1. Primary: Use pdfplumber to extract tables from native (machine-readable) PDFs
    2. Fallback: If no tables found, assume scanned PDF and use OCR (img2table + Tesseract)

    Args:
        file_path: Path to PDF file

    Returns:
        List of (page_number, dataframe) tuples
    """
    results = []

    try:
        with pdfplumber.open(file_path) as pdf:
            print(f"  [PDF] Processing {file_path.name} ({len(pdf.pages)} pages)")

            for page_num, page in enumerate(pdf.pages, start=1):
                page_has_data = False

                # PHASE 1: Try pdfplumber extraction (native PDF)
                tables = page.extract_tables()

                if tables:
                    for table_idx, table in enumerate(tables):
                        try:
                            # Convert table to DataFrame
                            # First row typically contains headers
                            if len(table) > 1:
                                df = pd.DataFrame(table[1:], columns=table[0])

                                # Clean up: remove empty rows/columns
                                df = df.dropna(how='all', axis=0)
                                df = df.dropna(how='all', axis=1)

                                if not df.empty:
                                    print(f"    - Page {page_num}, Table {table_idx + 1}: {df.shape}")
                                    results.append((page_num, df))
                                    page_has_data = True
                        except Exception as e:
                            logging.warning(f"    - Failed to parse table on page {page_num}: {str(e)}")

                # PHASE 2: OCR fallback for scanned pages (if pdfplumber found nothing)
                if not page_has_data and OCR_AVAILABLE:
                    try:
                        print(f"    - Page {page_num}: No native tables found, attempting OCR...")

                        # Convert PDF page to image using pdf2image
                        # first_page and last_page are 1-indexed
                        images = convert_from_path(
                            str(file_path),
                            first_page=page_num,
                            last_page=page_num,
                            dpi=300,  # Higher DPI = better OCR accuracy
                            fmt='png'
                        )

                        if images:
                            # Save temporary image for img2table processing
                            temp_image_path = Path(file_path.parent) / f"_temp_page_{page_num}.png"
                            images[0].save(temp_image_path, 'PNG')

                            try:
                                # Extract table from image
                                df = extract_table_from_image(temp_image_path, page_num)

                                if df is not None:
                                    results.append((page_num, df))
                                    page_has_data = True
                            finally:
                                # Clean up temporary image
                                if temp_image_path.exists():
                                    temp_image_path.unlink()

                    except Exception as e:
                        logging.warning(f"    - Page {page_num}: OCR fallback failed: {str(e)}")
                        # Don't crash the entire ingestion - just skip this page

                if not page_has_data:
                    logging.info(f"    - Page {page_num}: No tables extracted (tried both methods)")

    except Exception as e:
        logging.error(f"  [PDF] Failed to process {file_path.name}: {str(e)}")
        print(f"  [PDF] Failed to process {file_path.name}: {str(e)}")

    return results


def merge_dataframes(dataframes: List[pd.DataFrame], metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Merge multiple dataframes with graceful handling of schema differences

    Strategy:
    - If all dataframes have identical columns: simple concatenation
    - If columns differ: concatenate with NaN filling for missing columns

    Args:
        dataframes: List of DataFrames to merge
        metadata: Metadata dictionary to store warnings

    Returns:
        Merged DataFrame
    """
    if len(dataframes) == 1:
        return dataframes[0]

    # Check column alignment
    all_columns = [set(df.columns) for df in dataframes]
    common_columns = set.intersection(*all_columns)
    all_unique_columns = set.union(*all_columns)

    if len(common_columns) == len(all_unique_columns):
        # Perfect alignment - simple concatenation
        print("[Merge] All sources have identical schemas - simple concatenation")
        merged = pd.concat(dataframes, ignore_index=True)
    else:
        # Schema mismatch - concatenate with NaN filling
        print(f"[Merge] Schema mismatch detected:")
        print(f"  - Common columns: {len(common_columns)}")
        print(f"  - Total unique columns: {len(all_unique_columns)}")
        print(f"  - Missing columns will be filled with NaN")

        warning_msg = (
            f"Schema mismatch: {len(common_columns)} common columns out of "
            f"{len(all_unique_columns)} total unique columns"
        )
        metadata["warnings"].append(warning_msg)

        # Concatenate with outer join (fills missing columns with NaN)
        merged = pd.concat(dataframes, ignore_index=True, sort=False)

    return merged


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist

    Args:
        directory_path: Path to directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Directory ready: {directory_path}")
