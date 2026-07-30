"""
Dirty Data Generation Script
Generate CSV file containing various data quality issues for testing data cleaning Agent
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


def generate_dirty_data(output_path: str = "data/dirty_data.csv", n_rows: int = 200) -> None:
    """
    Generate dirty data with various data quality issues

    Data quality issues include:
    1. Missing values (NaN, None, empty strings)
    2. Incorrect date formats
    3. Dirty strings (leading/trailing spaces, special characters, inconsistent case)
    4. Numeric outliers
    5. Duplicate records
    6. Inconsistent data types
    """
    np.random.seed(42)
    random.seed(42)

    # 1. ID column - contains duplicate values
    ids = list(range(1, n_rows + 1))
    ids.extend([10, 25, 50])  # Add 3 duplicate IDs
    random.shuffle(ids)
    ids = ids[:n_rows]

    # 2. Name column - contains spaces, None, special characters
    names = [
        "Zhang San", "  Li Si  ", "Wang Wu", None, "Zhao Liu\n", "   ",
        "Sun Qi", "Zhou Ba", "Wu Jiu", "Zheng Shi", "ALICE", "bob", "Charlie"
    ]
    name_col = [random.choice(names) for _ in range(n_rows)]

    # 3. Age column - contains outliers and missing values
    ages = []
    for _ in range(n_rows):
        if random.random() < 0.1:  # 10% missing
            ages.append(np.nan)
        elif random.random() < 0.05:  # 5% outliers
            ages.append(random.choice([-5, 0, 150, 999]))
        else:
            ages.append(random.randint(18, 65))

    # 4. Date column - contains various incorrect formats
    dates = []
    base_date = datetime(2020, 1, 1)
    date_formats = [
        lambda d: d.strftime("%Y-%m-%d"),           # Correct format
        lambda d: d.strftime("%Y/%m/%d"),           # Slash format
        lambda d: d.strftime("%d-%m-%Y"),           # Wrong format
        lambda d: d.strftime("%Y%m%d"),             # No separator
        lambda d: "2020-13-45",                     # Invalid date
        lambda d: "",                                # Empty string
        lambda d: None,                              # None
    ]

    for i in range(n_rows):
        if random.random() < 0.7:  # 70% valid dates
            date = base_date + timedelta(days=random.randint(0, 1000))
            dates.append(random.choice(date_formats[:4])(date))
        else:  # 30% problem dates
            dates.append(random.choice(date_formats[4:])(None))

    # 5. Salary column - contains string formats, missing values, negative values
    salaries = []
    for _ in range(n_rows):
        if random.random() < 0.1:  # 10% missing
            salaries.append(np.nan)
        elif random.random() < 0.15:  # 15% string formats
            salaries.append(random.choice(["5000 yuan", "$8000", "10k", "N/A"]))
        elif random.random() < 0.05:  # 5% negative values
            salaries.append(-1000)
        else:
            salaries.append(random.randint(3000, 50000))

    # 6. Department column - contains inconsistent case, spaces
    departments = ["Sales", "sales", "SALES", " Marketing ", "IT", "it", "HR", None, ""]
    dept_col = [random.choice(departments) for _ in range(n_rows)]

    # 7. Email column - contains format errors
    def generate_email():
        if random.random() < 0.15:  # 15% incorrect format
            return random.choice([
                "invalid.email",           # Missing @
                "test@",                   # Missing domain
                "@example.com",            # Missing username
                None,                      # Missing value
                ""                         # Empty string
            ])
        else:
            username = f"user{random.randint(1, 100)}"
            domain = random.choice(["gmail.com", "qq.com", "163.com"])
            return f"{username}@{domain}"

    emails = [generate_email() for _ in range(n_rows)]

    # 8. Score column - contains out-of-range values
    scores = []
    for _ in range(n_rows):
        if random.random() < 0.1:  # 10% missing
            scores.append(np.nan)
        elif random.random() < 0.08:  # 8% out of range
            scores.append(random.choice([-10, 150, 9999]))
        else:
            scores.append(round(random.uniform(0, 100), 2))

    # 9. Status column - contains inconsistent values
    statuses = ["Active", "active", "ACTIVE", "Inactive", "inactive", "Pending", None, "unknown"]
    status_col = [random.choice(statuses) for _ in range(n_rows)]

    # 10. Remarks column - contains special characters and overly long text
    def generate_remark():
        if random.random() < 0.2:
            return None
        elif random.random() < 0.1:
            return "Normal remark" * 100  # Overly long text
        else:
            return random.choice([
                "Normal remark",
                "Contains special chars!@#$%^&*()",
                "Contains newline\ncharacter",
                "Contains tab\tcharacter here",
                ""
            ])

    remarks = [generate_remark() for _ in range(n_rows)]

    # Create DataFrame
    df = pd.DataFrame({
        "ID": ids,
        "Name": name_col,
        "Age": ages,
        "Hire Date": dates,
        "Monthly Salary": salaries,
        "Department": dept_col,
        "Email": emails,
        "Performance Score": scores,
        "Status": status_col,
        "Remarks": remarks
    })

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[SUCCESS] Dirty data generated: {output_path}")
    print(f"[INFO] Data dimensions: {df.shape}")
    print(f"\nData quality issue statistics:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total missing values: {df.isnull().sum().sum()}")
    print(f"  - Duplicate IDs: {df['ID'].duplicated().sum()}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nFirst 5 rows preview:")
    print(df.head())


if __name__ == "__main__":
    generate_dirty_data()
