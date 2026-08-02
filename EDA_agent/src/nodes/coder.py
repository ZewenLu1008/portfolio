"""
Coder Node - Data Cleaning Code Generation Node

Responsibilities:
    Act as a Python data engineer, generate executable Pandas code based on cleaning strategy
    Support Self-Correction: regenerate code based on error information when execution fails
"""
import os
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..core.state import DataCleaningState


# ========== Prompt Template ==========
CODER_SYSTEM_PROMPT = """You are an experienced Python data engineer skilled in data cleaning using Pandas.

CRITICAL LANGUAGE REQUIREMENT: You MUST output your response ENTIRELY in English. Do NOT output any Chinese characters, phrases, greetings, or explanations. This is a strict requirement for the pipeline to function. Every comment, every variable name, every docstring must be in English only.

Your tasks are:
1. Write high-quality Pandas data cleaning code based on the cleaning strategy provided by the data analyst
2. The code must be robust, executable, and include necessary exception handling
3. The code style should be clear with detailed comments IN ENGLISH ONLY

**Extremely important output requirements:**
1. Must define a function named `clean_data(df)`
2. The function receives a Pandas DataFrame as parameter
3. The function must return the cleaned DataFrame
4. Do not include any file I/O operations (such as pd.read_csv, to_csv, etc.)
5. Do not include any print statements or visualization code
6. Only use Pandas and Python standard library, do not introduce other third-party libraries
7. ALL code comments MUST be in English (no Chinese comments allowed)

**Function signature example:**
```python
import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Data cleaning function

    Args:
        df: Original DataFrame

    Returns:
        Cleaned DataFrame
    \"\"\"
    # Create copy to avoid modifying original data
    df = df.copy()

    # Your cleaning logic...

    return df
```

**Code quality requirements:**
- Use df.copy() to avoid modifying original data
- Use try-except to handle possible exceptions
- For date parsing, use pd.to_datetime with errors='coerce' parameter
- For numeric conversion, use pd.to_numeric with errors='coerce' parameter
- Add clear comments explaining the purpose of each step (IN ENGLISH ONLY)
- **CRITICAL**: Do NOT use the deprecated 'infer_datetime_format' parameter with pd.to_datetime() - it is deprecated in newer Pandas versions
- When the cleaning strategy mentions translating Chinese column names, this MUST be the VERY FIRST operation in your code

MANDATORY DATA CLEANING RULES (CRITICAL):
1. **CRITICAL RULE FOR NUMERIC CLEANING (Mixed-Type Columns)**:
   - Columns merged from different files (CSV, Excel) often contain MIXED TYPES: some rows are floats, others are strings with '$', '€', commas, etc.
   - If you apply `.str.replace()` directly, it will FAIL on float rows and turn valid numbers into NaNs, causing data loss.
   - You MUST cast to string BEFORE regex replacement, then convert to numeric, then impute. Use exactly this pattern:

   ```python
   df['col'] = df['col'].astype(str).str.replace(r'[^\d.]', '', regex=True)
   df['col'] = pd.to_numeric(df['col'], errors='coerce')
   df['col'] = df['col'].fillna(df['col'].median())
   ```

   - This three-step pattern is MANDATORY for all numeric columns (price, quantity, revenue, etc.).
   - NEVER skip the `.astype(str)` step - it prevents the mixed-type failure mode.
   - NEVER apply pd.to_numeric() directly on columns that may contain currency symbols or formatting.

2. **CRITICAL RULE FOR DATE PARSING (Mixed-Format Datetime)**:
   - When converting string columns to datetime (like order_date, transaction_date), you MUST handle mixed formats from different data sources.
   - Different files may contain different formats: CSV uses YYYY-MM-DD, Excel uses DD/MM/YYYY, PDF may use MM-DD-YYYY.
   - You MUST use this exact syntax for robust mixed-format parsing:

   ```python
   df['col'] = pd.to_datetime(df['col'], format='mixed', dayfirst=True, errors='coerce')
   ```

   - The `format='mixed'` parameter tells pandas to infer each date individually rather than assuming one format.
   - The `dayfirst=True` parameter prioritizes DD/MM/YYYY interpretation when ambiguous (08/04/2024 → April 8th, not August 4th).
   - NEVER use a single fixed format string or dayfirst=False - this will fail on heterogeneous datasets.
   - After parsing, NaT values should be minimal (< 5% of rows). If you see many NaT values, the parsing strategy is wrong.

3. **Strict Row Retention Policy**:
   - DO NOT drop rows unless explicitly required by the cleaning strategy.
   - For missing numeric values (e.g., price, quantity): impute with median using `df['col'].fillna(df['col'].median())`
   - For missing categorical values: consider mode imputation or leave as-is if acceptable.
   - For missing dates: leave as NaT but keep the row.
   - For duplicates: ONLY use `drop_duplicates()` if the entire row is an exact match. Do NOT drop based on a single column unless explicitly instructed.
   - Target: Retain >90% of input rows after cleaning.

4. **Clean Schema Output**:
   - DO NOT add diagnostic columns like `_is_outlier`, `_invalid_date`, `_source_file`, or temporary flags.
   - If you create temporary columns for intermediate processing, DROP them before returning: `df = df.drop(columns=['_temp_col'])`
   - The returned DataFrame should contain ONLY the cleaned versions of the original columns (after translation/standardization).

5. **String Cleaning for Categorical Columns**:
   - Always strip leading/trailing whitespace: `df['category'] = df['category'].str.strip()`
   - Standardize casing if needed: `df['category'] = df['category'].str.title()` or `.str.lower()`
   - Remove extra internal spaces: `df['category'] = df['category'].str.replace(r'\s+', ' ', regex=True)`

REMINDER: Output MUST be 100% in English. No Chinese characters allowed in comments, strings, or any part of the code.
"""

CODER_HUMAN_TEMPLATE = """Please generate data cleaning code based on the following information:

CRITICAL LANGUAGE REQUIREMENT: Output MUST be 100% in English. No Chinese characters in comments or strings.

**Dataset Information:**
- Column names: {columns}
- Data types: {dtypes}

**Cleaning Strategy:**
{cleaning_plan}

Please generate complete `clean_data(df)` function code with all comments in English.

REMINDER: Do NOT use deprecated 'infer_datetime_format' parameter with pd.to_datetime().
"""

CODER_SELF_CORRECTION_TEMPLATE = """Please regenerate data cleaning code based on the following information:

CRITICAL LANGUAGE REQUIREMENT: You MUST output your response ENTIRELY in English. Do NOT output any Chinese characters, phrases, or comments. All code comments must be in English only.

**Dataset Information:**
- Column names: {columns}
- Data types: {dtypes}

**Cleaning Strategy:**
{cleaning_plan}

**WARNING: Previous code execution failed, error information as follows:**
```
{execution_error}
```

**Please analyze the error cause and fix the code. Common issues:**
1. Column name spelling error or does not exist (check if Chinese column names need translation first)
2. Data type conversion failure
3. Regular expression syntax error
4. Index out of bounds or improper empty data handling
5. Function call parameter error
6. Using deprecated Pandas parameters (e.g., 'infer_datetime_format' in pd.to_datetime)

**CRITICAL**: Do NOT use the deprecated 'infer_datetime_format' parameter with pd.to_datetime().

Please generate the corrected `clean_data(df)` function code with ALL comments in English.

REMINDER: Output MUST be 100% in English. No Chinese characters allowed.
"""


# ========== Node Function ==========
def coder_node(state: DataCleaningState) -> Dict[str, Any]:
    """
    Coder Node - Code Generation Node

    Functions:
        1. Read cleaning strategy and data information
        2. Check if it is a retry scenario (retry_count > 0)
        3. Call LLM to generate Pandas cleaning code
        4. Extract pure code (remove Markdown markers)
        5. Update state["generated_code"]

    Args:
        state: Global state object

    Returns:
        Dictionary containing updated fields:
        - generated_code: Generated Python code
    """
    print("\n" + "="*60)
    print("[Coder Node] Starting cleaning code generation...")
    print("="*60)

    # 1. Read input data
    cleaning_plan = state.get("cleaning_plan")
    original_df_info = state.get("original_df_info", {})
    retry_count = state.get("retry_count", 0)
    execution_error = state.get("execution_error")

    if not cleaning_plan:
        raise ValueError("Cleaning strategy is empty, cannot generate code")

    # 2. Determine if it is a retry scenario
    is_retry = retry_count > 0 and execution_error is not None

    if is_retry:
        print(f"[RETRY] Retry scenario detected (attempt {retry_count})")
        print(f"[INFO] Previous error: {execution_error[:100]}...")

    # 3. Format input information
    columns = ", ".join(original_df_info.get("columns", []))
    dtypes = "\n".join(
        [f"  - {col}: {dtype}" for col, dtype in original_df_info.get("dtypes", {}).items()]
    )

    # 4. Build Prompt (select different template based on retry)
    if is_retry:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CODER_SYSTEM_PROMPT),
            ("human", CODER_SELF_CORRECTION_TEMPLATE)
        ])
        input_vars = {
            "columns": columns,
            "dtypes": dtypes,
            "cleaning_plan": cleaning_plan,
            "execution_error": execution_error
        }
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CODER_SYSTEM_PROMPT),
            ("human", CODER_HUMAN_TEMPLATE)
        ])
        input_vars = {
            "columns": columns,
            "dtypes": dtypes,
            "cleaning_plan": cleaning_plan
        }

    # 5. Initialize LLM (DeepSeek Coder)
    llm = ChatOpenAI(
        model="deepseek-coder",  # Use code-specific model
        temperature=0.0,  # Zero temperature to ensure code generation stability
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
    )

    # 6. Build Chain and execute
    chain = prompt | llm

    print(f"[INFO] Calling LLM to generate code...")
    response = chain.invoke(input_vars)

    # 7. Extract pure code (remove Markdown code block markers)
    raw_output = response.content
    clean_code = extract_python_code(raw_output)

    if not clean_code:
        raise ValueError("Unable to extract valid Python code from LLM output")

    # 8. Verify if code contains clean_data function
    if "def clean_data(" not in clean_code:
        raise ValueError("Generated code is missing clean_data(df) function definition")

    print(f"\n[SUCCESS] Code generation complete ({len(clean_code)} characters)")
    print("\n" + "-"*60)
    print("Generated Code Preview (first 20 lines):")
    print("-"*60)
    code_lines = clean_code.split('\n')
    print('\n'.join(code_lines[:20]))
    if len(code_lines) > 20:
        print("...")
    print()

    # 9. Return updated fields
    return {
        "generated_code": clean_code
    }


# ========== Helper Functions ==========
def extract_python_code(text: str) -> str:
    """
    Extract pure Python code from LLM output

    Processing logic:
        1. First try to extract Markdown code block (```python ... ```)
        2. If no code block markers, return original text
        3. Remove leading and trailing whitespace

    Args:
        text: LLM's raw output text

    Returns:
        Extracted pure Python code

    Examples:
        >>> text = "Some explanation\\n```python\\nprint('hello')\\n```\\nMore explanation"
        >>> extract_python_code(text)
        "print('hello')"
    """
    # Regular expression to match ```python ... ``` or ``` ... ```
    # Use re.DOTALL to make . match newlines
    patterns = [
        r"```python\s*\n(.*?)\n```",  # Match ```python ... ```
        r"```\s*\n(.*?)\n```",        # Match ``` ... ```
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return code

    # If no code block markers found, return original text (remove leading/trailing whitespace)
    return text.strip()


def validate_generated_code(code: str) -> bool:
    """
    Validate if generated code meets specifications

    Validation items:
        1. Contains clean_data function definition
        2. Contains import pandas
        3. Does not contain file I/O operations
        4. Does not contain dangerous operations (eval, exec, etc.)

    Args:
        code: Generated code

    Returns:
        Whether validation passed
    """
    # Required content
    required_patterns = [
        r"def clean_data\(",           # Function definition
        r"import pandas",               # pandas import
        r"return\s+\w+",               # return statement
    ]

    # Forbidden dangerous operations
    forbidden_patterns = [
        r"pd\.read_csv",                # Prohibit reading files
        r"\.to_csv",                    # Prohibit writing files
        r"\beval\(",                    # Prohibit eval
        r"\bexec\(",                    # Prohibit exec
        r"__import__",                  # Prohibit dynamic import
        r"os\.system",                  # Prohibit system calls
    ]

    # Check required content
    for pattern in required_patterns:
        if not re.search(pattern, code):
            print(f"[WARNING] Validation failed: missing required pattern {pattern}")
            return False

    # Check forbidden operations
    for pattern in forbidden_patterns:
        if re.search(pattern, code):
            print(f"[WARNING] Validation failed: contains forbidden operation {pattern}")
            return False

    return True
