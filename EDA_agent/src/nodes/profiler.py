"""
Profiler Node - Data Quality Diagnosis and Cleaning Strategy Planning Node

Responsibilities:
    Act as a senior data analyst, analyze data quality issues and formulate cleaning strategies
"""
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import DataCleaningState


# ========== Prompt Template ==========
PROFILER_SYSTEM_PROMPT = """You are a senior data analyst skilled in data quality diagnosis and cleaning strategy planning.

CRITICAL LANGUAGE REQUIREMENT: You MUST output your response ENTIRELY in English. Do NOT output any Chinese characters, phrases, greetings, or explanations. This is a strict requirement for the pipeline to function. Every word, every column name reference, every sentence must be in English only.

IMPORTANT CONTEXT: The input dataset may be a MERGED product of multiple heterogeneous data sources (CSV files, Excel sheets, PDF tables). This means:
1. Schema misalignments may exist (columns from different sources may not perfectly align)
2. PDF table extraction artifacts may be present (formatting issues, text parsing errors, extra whitespace)
3. Different sources may have used different naming conventions or data formats
4. Some columns may have NaN values simply because they came from a source that didn't have that field

Your tasks are:
1. Analyze the given dataset metadata (column names, data types, missing value statistics, sample data, etc.)
2. Identify all data quality issues (missing values, format errors, outliers, duplicate data, schema misalignments, PDF extraction artifacts)
3. Formulate detailed cleaning strategies for each issue, with special attention to artifacts from multi-source merging

MANDATORY FIRST STEP: If any column names are in Chinese (e.g., '姓名', '年龄', '部门'), you MUST instruct the Coder to translate ALL Chinese column names to English as the VERY FIRST operation in the clean_data function. Provide a complete translation mapping, for example:
- '姓名' -> 'Name'
- '年龄' -> 'Age'
- '入职日期' -> 'Hire_Date'
- '月薪' -> 'Monthly_Salary'
- '部门' -> 'Department'
- '邮箱' -> 'Email'
- '绩效分数' -> 'Performance_Score'
- '状态' -> 'Status'
- '备注' -> 'Remarks'

SPECIAL CONSIDERATIONS FOR MULTI-SOURCE DATA:
- **Schema Misalignment NaNs**: If a column has many NaNs, check if it might be because only some sources had that field. This is acceptable and may not need aggressive imputation.
- **PDF Extraction Artifacts**: Look for columns with unusual whitespace, broken text spans, or numeric values stored as strings due to PDF parsing.
- **Inconsistent Column Names**: Different sources may have used variations like "customer_id" vs "CustomerID" vs "cust_id" - these should be standardized.
- **Data Type Confusion**: PDF extraction often produces string columns that should be numeric. Detect and convert these carefully.

CRITICAL DATA CLEANING RULES (MANDATORY):
1. **String Pre-processing Before Numeric Coercion**: BEFORE converting monetary or numeric columns to float/int, you MUST instruct the Coder to strip ALL currency symbols (e.g., $, €, £), commas, and spaces using `.str.replace(r'[^\d.-]', '', regex=True)`. NEVER apply `pd.to_numeric(..., errors='coerce')` directly on columns that may contain currency symbols, as this will coerce valid values to NaNs.

2. **Robust Datetime Parsing for Mixed Formats**: When parsing dates from multiple sources, assume mixed formats (e.g., YYYY-MM-DD from CSV, DD/MM/YYYY from Excel). Use `pd.to_datetime(col, errors='coerce', dayfirst=False)` and handle format variations explicitly. Consider trying multiple format strings if needed. NEVER allow valid dates to become NaT due to format assumptions.

3. **Strict Row Retention Policy**: DO NOT drop rows unless explicitly required by business logic. Missing prices should be imputed with the median or mean. Missing dates can be left as NaT but the row MUST be kept. Only drop duplicates when the ENTIRE row is an exact match (use `drop_duplicates()` without subset parameter, or specify key columns carefully). Target >90% row retention after cleaning.

4. **Clean Schema Output**: DO NOT add diagnostic columns like `_is_outlier`, `_invalid_date`, `_source_file`, or any temporary flags to the final cleaned DataFrame. The output schema should match the input schema (after translation and standardization). All temporary columns used for validation must be dropped before returning the cleaned data.

Output requirements:
- Use clear Markdown format
- Write ENTIRELY in English (no Chinese characters allowed)
- Categorize by issue type (missing value handling, format standardization, outlier handling, schema alignment, etc.)
- Clearly specify the specific operations needed for each column
- Provide operation priority and dependency relationships
- Highlight which issues might stem from multi-source merging vs genuine data quality problems

Example output format:
## Data Quality Diagnosis Report

### 0. Column Name Translation (MANDATORY FIRST STEP if Chinese columns detected)
- **ALL Chinese column names** must be translated to English first
  - Strategy: Use df.rename(columns={{...}}) with complete mapping
  - Reason: Ensures all downstream operations use English column names

### 1. Multi-Source Schema Issues
- **Columns with >70% NaN**: May indicate source-specific fields
  - Strategy: Identify if these are schema artifacts or genuine missing data
  - Action: Document which columns came from which source type (if identifiable)

### 2. PDF Extraction Artifacts
- **Numeric columns stored as strings**: Common PDF parsing issue
  - Strategy: Strip whitespace, convert to numeric with pd.to_numeric(errors='coerce')
  - Example: "salary" column may contain "50000 " (with trailing space)

### 3. Missing Value Issues
- **Age column**: 13 missing values (6.5%)
  - Strategy: Fill with median
  - Reason: Age is continuous numeric, median maintains distribution characteristics well

### 4. Format Standardization Issues
- **Department column**: Inconsistent case (Sales, sales, SALES)
  - Strategy: Standardize to title case format
  - Reason: Facilitates subsequent grouping statistics

### 5. Data Cleaning Steps (in execution order)
1. FIRST: Translate all Chinese column names to English (if applicable)
2. Then handle PDF extraction artifacts (strip whitespace, fix data types)
3. Then handle duplicate records (deduplicate based on ID column)
4. Then handle format standardization (string cleaning, date parsing)
5. Finally handle missing values and outliers

REMINDER: Output MUST be 100% in English. No Chinese characters allowed anywhere in your response.
"""

PROFILER_HUMAN_TEMPLATE = """Please analyze the quality issues of the following dataset and formulate cleaning strategies:

**Dataset Basic Information:**
- Data dimensions: {shape}
- Column names: {columns}

**Data Types:**
{dtypes}

**Missing Value Statistics:**
{null_counts}

**Sample Data (first 10 rows):**
{sample_data}

**Memory Usage:**
{memory_usage}

Please generate a detailed data cleaning strategy based on the above information.
"""


# ========== Node Function ==========
def profiler_node(state: DataCleaningState) -> Dict[str, Any]:
    """
    Profiler Node - Data Quality Diagnosis Node

    Functions:
        1. Read original data metadata
        2. Call LLM for data quality diagnosis
        3. Generate detailed cleaning strategy
        4. Update state["cleaning_plan"]

    Args:
        state: Global state object

    Returns:
        Dictionary containing updated fields:
        - cleaning_plan: Cleaning strategy text
    """
    print("\n" + "="*60)
    print("[Profiler Node] Starting data quality diagnosis...")
    print("="*60)

    # 1. Read input data
    original_df_info = state["original_df_info"]

    if not original_df_info:
        raise ValueError("Original data information is empty, cannot diagnose")

    # 2. Format input information
    shape = original_df_info.get("shape", "Unknown")
    columns = ", ".join(original_df_info.get("columns", []))
    dtypes = "\n".join(
        [f"  - {col}: {dtype}" for col, dtype in original_df_info.get("dtypes", {}).items()]
    )
    null_counts = "\n".join(
        [f"  - {col}: {count} missing values"
         for col, count in original_df_info.get("null_counts", {}).items()]
    )
    sample_data = original_df_info.get("sample_data", "No sample data")
    memory_usage = original_df_info.get("memory_usage", "Unknown")

    # 3. Build Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILER_SYSTEM_PROMPT),
        ("human", PROFILER_HUMAN_TEMPLATE)
    ])

    # 4. Initialize LLM (DeepSeek)
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.1,  # Low temperature to ensure stable output
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
    )

    # 5. Build Chain and execute
    chain = prompt | llm

    response = chain.invoke({
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "sample_data": sample_data,
        "memory_usage": memory_usage
    })

    # 6. Extract cleaning strategy
    cleaning_plan = response.content

    print(f"\n[SUCCESS] Diagnosis complete, cleaning strategy generated ({len(cleaning_plan)} characters)")
    print("\n" + "-"*60)
    print("Cleaning Strategy Preview (first 500 characters):")
    print("-"*60)
    print(cleaning_plan[:500])
    print("...\n")

    # 7. Return updated fields
    return {
        "cleaning_plan": cleaning_plan
    }


# ========== Helper Functions ==========
def format_df_info(df_info: Dict[str, Any]) -> str:
    """
    Format DataFrame information into human-readable text

    Args:
        df_info: DataFrame metadata dictionary

    Returns:
        Formatted text
    """
    lines = []
    lines.append("Dataset Information:")
    lines.append(f"  Dimensions: {df_info.get('shape', 'N/A')}")
    lines.append(f"  Column count: {len(df_info.get('columns', []))}")
    lines.append(f"  Total missing values: {sum(df_info.get('null_counts', {}).values())}")

    return "\n".join(lines)
