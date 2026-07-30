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

Your tasks are:
1. Analyze the given dataset metadata (column names, data types, missing value statistics, sample data, etc.)
2. Identify all data quality issues (missing values, format errors, outliers, duplicate data, etc.)
3. Formulate detailed cleaning strategies for each issue

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

Output requirements:
- Use clear Markdown format
- Write ENTIRELY in English (no Chinese characters allowed)
- Categorize by issue type (missing value handling, format standardization, outlier handling, etc.)
- Clearly specify the specific operations needed for each column
- Provide operation priority and dependency relationships

Example output format:
## Data Quality Diagnosis Report

### 0. Column Name Translation (MANDATORY FIRST STEP if Chinese columns detected)
- **ALL Chinese column names** must be translated to English first
  - Strategy: Use df.rename(columns={{...}}) with complete mapping
  - Reason: Ensures all downstream operations use English column names

### 1. Missing Value Issues
- **Age column**: 13 missing values (6.5%)
  - Strategy: Fill with median
  - Reason: Age is continuous numeric, median maintains distribution characteristics well

### 2. Format Standardization Issues
- **Department column**: Inconsistent case (Sales, sales, SALES)
  - Strategy: Standardize to title case format
  - Reason: Facilitates subsequent grouping statistics

### 3. Data Cleaning Steps (in execution order)
1. FIRST: Translate all Chinese column names to English (if applicable)
2. Then handle duplicate records (deduplicate based on ID column)
3. Then handle format standardization (string cleaning, date parsing)
4. Finally handle missing values and outliers

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
