"""
Generate Heterogeneous Test Data

Creates sample CSV, Excel, and PDF files with tables for testing
the multi-source data ingestion pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def generate_csv_data(output_dir: Path) -> None:
    """
    Generate a sample CSV file with customer data
    """
    np.random.seed(42)

    data = {
        "customer_id": range(1, 51),
        "name": [f"Customer_{i}" for i in range(1, 51)],
        "age": np.random.randint(18, 70, 50),
        "purchase_amount": np.random.uniform(10, 500, 50).round(2),
        "region": np.random.choice(["North", "South", "East", "West"], 50)
    }

    df = pd.DataFrame(data)

    # Introduce some missing values
    df.loc[5:7, "age"] = np.nan
    df.loc[15, "purchase_amount"] = np.nan

    output_path = output_dir / "customers.csv"
    df.to_csv(output_path, index=False)
    print(f"[Generated] CSV file: {output_path}")
    print(f"  Shape: {df.shape}")


def generate_excel_data(output_dir: Path) -> None:
    """
    Generate a sample Excel file with multiple sheets (products and sales)
    """
    np.random.seed(43)

    # Sheet 1: Products
    products_data = {
        "product_id": range(101, 121),
        "product_name": [f"Product_{i}" for i in range(101, 121)],
        "category": np.random.choice(["Electronics", "Clothing", "Food", "Books"], 20),
        "price": np.random.uniform(5, 200, 20).round(2),
        "stock": np.random.randint(0, 100, 20)
    }
    products_df = pd.DataFrame(products_data)

    # Introduce some duplicates
    products_df = pd.concat([products_df, products_df.iloc[[0, 5, 10]]], ignore_index=True)

    # Sheet 2: Sales
    sales_data = {
        "sale_id": range(1, 31),
        "product_id": np.random.choice(range(101, 121), 30),
        "quantity": np.random.randint(1, 10, 30),
        "sale_date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "discount": np.random.uniform(0, 0.3, 30).round(2)
    }
    sales_df = pd.DataFrame(sales_data)

    # Introduce missing values
    sales_df.loc[10:12, "discount"] = np.nan

    output_path = output_dir / "products_sales.xlsx"
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        products_df.to_excel(writer, sheet_name='Products', index=False)
        sales_df.to_excel(writer, sheet_name='Sales', index=False)

    print(f"[Generated] Excel file: {output_path}")
    print(f"  - Sheet 'Products': {products_df.shape}")
    print(f"  - Sheet 'Sales': {sales_df.shape}")


def generate_pdf_with_table(output_dir: Path) -> None:
    """
    Generate a sample PDF file containing a table with employee data
    """
    np.random.seed(44)

    # Create employee data
    data = {
        "emp_id": range(1001, 1021),
        "employee_name": [f"Employee_{i}" for i in range(1001, 1021)],
        "department": np.random.choice(["HR", "IT", "Sales", "Finance"], 20),
        "salary": np.random.randint(30000, 100000, 20),
        "hire_year": np.random.randint(2015, 2024, 20)
    }

    df = pd.DataFrame(data)

    output_path = output_dir / "employees.pdf"

    # Create PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Add title
    title = Paragraph("<b>Employee Information Report</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Add description
    description = Paragraph(
        "This PDF contains a table with employee information including ID, name, department, salary, and hire year.",
        styles['Normal']
    )
    story.append(description)
    story.append(Spacer(1, 20))

    # Convert DataFrame to table data
    table_data = [df.columns.tolist()] + df.values.tolist()

    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(table)

    # Build PDF
    doc.build(story)

    print(f"[Generated] PDF file: {output_path}")
    print(f"  Table shape: {df.shape}")


def main():
    """
    Generate all heterogeneous test data files
    """
    print("\n" + "="*60)
    print("Generating Heterogeneous Test Data")
    print("="*60 + "\n")

    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Output directory: {output_dir}\n")

    # Generate files
    try:
        generate_csv_data(output_dir)
        print()

        generate_excel_data(output_dir)
        print()

        generate_pdf_with_table(output_dir)
        print()

        print("="*60)
        print("[SUCCESS] All test files generated successfully!")
        print("="*60)
        print(f"\nGenerated files in {output_dir}:")
        print("  - customers.csv (50 rows)")
        print("  - products_sales.xlsx (2 sheets)")
        print("  - employees.pdf (1 table with 20 rows)")
        print(f"\nTotal expected rows after merge: ~113 rows")

    except Exception as e:
        print(f"\n[ERROR] Failed to generate test data: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
