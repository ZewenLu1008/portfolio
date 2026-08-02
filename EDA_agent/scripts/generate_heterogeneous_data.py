"""
Generate Heterogeneous E-commerce Test Data

Creates realistic CSV, Excel, and PDF files with overlapping schemas
for testing the multi-source data ingestion pipeline.

Domain: Global E-commerce Sales
Common Schema: order_id, order_date, customer_name, category, price
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors


def generate_csv_north_america(output_dir: Path) -> None:
    """
    Generate CSV file: North America Sales (~50 rows)

    Data quality issues:
    - A few duplicate order_ids
    - 2-3 missing values in price column
    """
    np.random.seed(42)

    # Generate 50 orders
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    names = ["John Smith", "Emily Johnson", "Michael Brown", "Sarah Davis", "David Wilson",
             "Jessica Garcia", "James Martinez", "Jennifer Rodriguez", "Robert Lee", "Linda Taylor"]

    data = {
        "order_id": [f"NA{1000 + i}" for i in range(50)],
        "order_date": [(datetime(2024, 1, 1) + timedelta(days=i*2)).strftime("%Y-%m-%d") for i in range(50)],
        "customer_name": np.random.choice(names, 50),
        "category": np.random.choice(categories, 50),
        "price": np.random.uniform(15.99, 599.99, 50).round(2)
    }

    df = pd.DataFrame(data)

    # Inject duplicate order_ids (3 duplicates)
    df.loc[5, "order_id"] = df.loc[3, "order_id"]
    df.loc[15, "order_id"] = df.loc[10, "order_id"]
    df.loc[25, "order_id"] = df.loc[20, "order_id"]

    # Inject missing prices (3 NaN values)
    df.loc[8, "price"] = np.nan
    df.loc[18, "price"] = np.nan
    df.loc[35, "price"] = np.nan

    output_path = output_dir / "sales_north_america.csv"
    df.to_csv(output_path, index=False)
    print(f"[Generated] CSV file: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Issues: 3 duplicate order_ids, 3 missing prices")


def generate_excel_europe(output_dir: Path) -> None:
    """
    Generate Excel file: Europe Sales (~50 rows)

    Data quality issues:
    - Different date format (DD/MM/YYYY instead of YYYY-MM-DD)
    - Price with "$" currency symbol prefix (strings instead of numbers)
    """
    np.random.seed(43)

    # Generate 50 orders
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    names = ["Oliver Smith", "Emma Johnson", "Liam Brown", "Sophia Davis", "Noah Wilson",
             "Ava Garcia", "Elijah Martinez", "Isabella Rodriguez", "William Lee", "Mia Taylor"]

    data = {
        "order_id": [f"EU{2000 + i}" for i in range(50)],
        "order_date": [(datetime(2024, 1, 1) + timedelta(days=i*2)).strftime("%d/%m/%Y") for i in range(50)],  # DD/MM/YYYY format
        "customer_name": np.random.choice(names, 50),
        "category": np.random.choice(categories, 50),
        "price": [f"${price:.2f}" for price in np.random.uniform(15.99, 599.99, 50)]  # String with $ prefix
    }

    df = pd.DataFrame(data)

    output_path = output_dir / "sales_europe.xlsx"
    df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"[Generated] Excel file: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Issues: DD/MM/YYYY date format, prices with '$' prefix")


def generate_pdf_asia(output_dir: Path) -> None:
    """
    Generate PDF file: Asia Sales (~30 rows)

    Data quality issues:
    - Irregular casing in category column
    - Trailing/leading spaces in category column
    """
    np.random.seed(44)

    # Generate 30 orders
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    names = ["Wei Chen", "Yuki Tanaka", "Raj Patel", "Min-jun Kim", "Priya Sharma",
             "Haruto Suzuki", "Aisha Khan", "Kenji Sato", "Anika Gupta", "Taro Yamamoto"]

    data = {
        "order_id": [f"AS{3000 + i}" for i in range(30)],
        "order_date": [(datetime(2024, 1, 1) + timedelta(days=i*3)).strftime("%Y-%m-%d") for i in range(30)],
        "customer_name": np.random.choice(names, 30),
        "category": np.random.choice(categories, 30),
        "price": np.random.uniform(15.99, 599.99, 30).round(2)
    }

    df = pd.DataFrame(data)

    # Inject irregular casing and spacing issues in category
    df.loc[2, "category"] = "  Electronics "  # Leading and trailing spaces
    df.loc[5, "category"] = "CLOTHING"  # All uppercase
    df.loc[8, "category"] = "home & garden"  # All lowercase
    df.loc[12, "category"] = "  SPORTS  "  # Uppercase with spaces
    df.loc[15, "category"] = "books "  # Lowercase with trailing space
    df.loc[18, "category"] = " Electronics"  # Leading space
    df.loc[22, "category"] = "BOOKS"  # Uppercase
    df.loc[25, "category"] = "clothing  "  # Lowercase with trailing spaces
    df.loc[28, "category"] = "  Home & Garden"  # Leading spaces with mixed case

    output_path = output_dir / "sales_asia.pdf"

    # Create PDF with table
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []

    # Convert DataFrame to table data (header + rows)
    table_data = [df.columns.tolist()] + df.values.tolist()

    # Create table with styling
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        # Body styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    story.append(table)
    doc.build(story)

    print(f"[Generated] PDF file: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Issues: Irregular casing and spacing in category column")


def main():
    """
    Generate all heterogeneous e-commerce test data files
    """
    print("\n" + "="*60)
    print("Generating Heterogeneous E-commerce Test Data")
    print("="*60 + "\n")
    print("Domain: Global E-commerce Sales")
    print("Schema: order_id, order_date, customer_name, category, price")
    print()

    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Output directory: {output_dir}\n")

    # Generate files
    try:
        generate_csv_north_america(output_dir)
        print()

        generate_excel_europe(output_dir)
        print()

        generate_pdf_asia(output_dir)
        print()

        print("="*60)
        print("[SUCCESS] All test files generated successfully!")
        print("="*60)
        print(f"\nGenerated files in {output_dir}:")
        print("  - sales_north_america.csv (50 rows)")
        print("  - sales_europe.xlsx (50 rows)")
        print("  - sales_asia.pdf (30 rows)")
        print(f"\nTotal expected rows after merge: 130 rows")
        print("\nData Quality Issues (All Recoverable):")
        print("  CSV: 3 duplicate order_ids, 3 missing prices")
        print("  Excel: DD/MM/YYYY date format, prices with '$' prefix")
        print("  PDF: Irregular casing and spacing in category")
        print("\nExpected retention rate: >90% after cleaning")

    except Exception as e:
        print(f"\n[ERROR] Failed to generate test data: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
