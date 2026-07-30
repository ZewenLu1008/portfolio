"""
QA Node - Quality Assurance Node

Responsibilities:
    Act as a "strict data testing engineer" to assess the quality of cleaned data
    Ensure cleaning strategy is properly executed and data quality is improved
"""
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..core.state import DataCleaningState


# ========== Prompt Template ==========
QA_SYSTEM_PROMPT = """You are a strict data quality testing engineer responsible for evaluating data cleaning effectiveness.

CRITICAL LANGUAGE REQUIREMENT: You MUST output your response ENTIRELY in English. Do NOT output any Chinese characters, phrases, greetings, or explanations. This is a strict requirement for the pipeline to function. Every word, every assessment, every recommendation must be in English only.

Your tasks are:
1. Compare data metadata before and after cleaning
2. Evaluate whether cleaning strategy was properly executed
3. Identify potential data quality issues

**Evaluation Dimensions:**
1. **Data Completeness**: Whether too many rows were lost after cleaning (normally should retain >50%)
2. **Missing Value Handling**: Whether missing values were significantly reduced
3. **Duplicate Data Handling**: Whether duplicate rows were properly removed
4. **Data Types**: Whether key column data types are reasonable
5. **Strategy Execution**: Whether all operations mentioned in the cleaning strategy are reflected in results

**Output Requirements:**
- Provide clear "Pass" or "Fail" judgment
- List all discovered issues
- Provide improvement suggestions
- Write ENTIRELY in English (no Chinese characters allowed)

**Output Format:**
## QA Results

**Judgment**: [Pass/Fail]

**Score**: [0-100]

### Positive Indicators
- [List what was done well]

### Issue List
- [List discovered issues, fill "None" if none]

### Improvement Suggestions
- [Provide specific suggestions, fill "Data quality is good" if none]

REMINDER: Output MUST be 100% in English. No Chinese characters allowed anywhere in your response.
"""

QA_HUMAN_TEMPLATE = """Please evaluate the following data cleaning effectiveness:

CRITICAL: Your entire response must be in English only. No Chinese characters allowed.

**Original Cleaning Strategy:**
{cleaning_plan}

---

**Before Cleaning Data Metadata:**
- Data dimensions: {original_shape}
- Total missing values: {original_nulls}
- Duplicate rows: {original_duplicates}
- Column information: {original_columns}

---

**After Cleaning Data Metadata:**
- Data dimensions: {cleaned_shape}
- Total missing values: {cleaned_nulls}
- Duplicate rows: {cleaned_duplicates}
- Column information: {cleaned_columns}

---

Please provide a detailed QA report based on the above information in English.

REMINDER: Output must be 100% in English.
"""


# ========== Node Function ==========
def qa_node(state: DataCleaningState) -> Dict[str, Any]:
    """
    QA Node - Quality Assurance Node

    Functions:
        1. Check if code execution succeeded
        2. Execute deterministic rule checks
        3. Call LLM for intelligent assessment
        4. Generate QA report

    Args:
        state: Global state object

    Returns:
        Dictionary containing updated fields:
        - qa_result: QA results (contains passed, score, issues, suggestions)
    """
    print("\n" + "="*60)
    print("[QA Node] Starting data quality assurance...")
    print("="*60)

    # 1. Check execution status
    execution_success = state.get("execution_success", False)

    if not execution_success:
        print("[SKIP] Code execution failed, skipping QA")
        return {
            "qa_result": {
                "passed": False,
                "score": 0,
                "reason": "Code execution failed, cannot perform QA",
                "issues": ["Code did not execute successfully"],
                "suggestions": ["Please fix code errors and retry"]
            }
        }

    # 2. Read data information
    original_df_info = state.get("original_df_info", {})
    cleaned_df_info = state.get("cleaned_df_info", {})
    cleaning_plan = state.get("cleaning_plan", "")

    if not original_df_info or not cleaned_df_info:
        print("[SKIP] Missing data metadata, cannot perform QA")
        return {
            "qa_result": {
                "passed": False,
                "score": 0,
                "reason": "Missing data metadata",
                "issues": ["Original or cleaned data information missing"],
                "suggestions": ["Please ensure data loading and execution are normal"]
            }
        }

    # 3. Execute deterministic rule checks
    print("\n[INFO] Executing deterministic rule checks...")
    rule_results = run_deterministic_checks(original_df_info, cleaned_df_info)

    print(f"\nRule check results:")
    for rule_name, result in rule_results.items():
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"  {status} {rule_name}: {result['message']}")

    # Calculate rule pass rate
    total_rules = len(rule_results)
    passed_rules = sum(1 for r in rule_results.values() if r["passed"])
    rule_pass_rate = passed_rules / total_rules if total_rules > 0 else 0

    print(f"\nRule pass rate: {passed_rules}/{total_rules} ({rule_pass_rate*100:.1f}%)")

    # 4. Call LLM for intelligent assessment
    print("\n[INFO] Calling LLM for intelligent assessment...")
    llm_assessment = run_llm_assessment(
        cleaning_plan,
        original_df_info,
        cleaned_df_info
    )

    # 5. Comprehensive evaluation
    print("\n[INFO] Generating comprehensive QA report...")

    # Extract LLM assessment pass status
    llm_passed = "Pass" in llm_assessment or "pass" in llm_assessment.lower()

    # Comprehensive judgment: all rules passed AND LLM assessment passed
    overall_passed = rule_pass_rate == 1.0 and llm_passed

    # Calculate comprehensive score
    rule_score = rule_pass_rate * 60  # Rule checks account for 60 points
    llm_score = 40 if llm_passed else 20  # LLM assessment accounts for 40 points
    total_score = int(rule_score + llm_score)

    # Collect all issues
    issues = [r["message"] for r in rule_results.values() if not r["passed"]]
    if not llm_passed:
        issues.append("LLM assessment found quality issues")

    # Generate improvement suggestions
    suggestions = []
    if rule_pass_rate < 1.0:
        suggestions.append("Some deterministic rules failed, please check cleaning logic")
    if not llm_passed:
        suggestions.append("Please refer to specific suggestions in LLM assessment report")
    if not suggestions:
        suggestions.append("Data quality is good, no improvements needed")

    # 6. Construct QA results
    qa_result = {
        "passed": overall_passed,
        "score": total_score,
        "reason": "All checks passed" if overall_passed else "Quality issues exist",
        "rule_pass_rate": rule_pass_rate,
        "rule_results": rule_results,
        "llm_assessment": llm_assessment,
        "issues": issues if issues else ["None"],
        "suggestions": suggestions
    }

    # 7. Print QA report
    print("\n" + "="*60)
    print("QA Report")
    print("="*60)
    print(f"Overall Judgment: {'[PASS]' if overall_passed else '[FAIL]'}")
    print(f"Comprehensive Score: {total_score}/100")
    print(f"\nIssue List:")
    for issue in qa_result["issues"]:
        print(f"  - {issue}")
    print(f"\nImprovement Suggestions:")
    for suggestion in qa_result["suggestions"]:
        print(f"  - {suggestion}")
    print("="*60)

    return {
        "qa_result": qa_result
    }


# ========== Deterministic Rule Checks ==========
def run_deterministic_checks(
    original_info: Dict[str, Any],
    cleaned_info: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Execute deterministic rule checks

    Args:
        original_info: Original data metadata
        cleaned_info: Cleaned data metadata

    Returns:
        Rule check results dictionary
    """
    results = {}

    # Rule 1: Data retention rate check (should not be less than 50%)
    original_rows = original_info.get("shape", [0])[0]
    cleaned_rows = cleaned_info.get("shape", [0])[0]

    if original_rows > 0:
        retention_rate = cleaned_rows / original_rows
        results["Data retention rate"] = {
            "passed": retention_rate >= 0.5,
            "message": f"Retained {retention_rate*100:.1f}% of data ({cleaned_rows}/{original_rows} rows)",
            "value": retention_rate
        }
    else:
        results["Data retention rate"] = {
            "passed": False,
            "message": "Original data is empty",
            "value": 0
        }

    # Rule 2: Missing value improvement check
    original_nulls = original_info.get("total_nulls", 0)
    cleaned_nulls = cleaned_info.get("total_nulls", 0)

    null_reduction = original_nulls - cleaned_nulls
    null_reduction_rate = null_reduction / original_nulls if original_nulls > 0 else 1.0

    results["Missing value improvement"] = {
        "passed": cleaned_nulls <= original_nulls,
        "message": f"Missing values reduced from {original_nulls} to {cleaned_nulls} (reduced by {null_reduction_rate*100:.1f}%)",
        "value": null_reduction_rate
    }

    # Rule 3: Duplicate data improvement check
    original_dups = original_info.get("duplicate_count", 0)
    cleaned_dups = cleaned_info.get("duplicate_count", 0)

    dup_reduction = original_dups - cleaned_dups

    results["Duplicate data improvement"] = {
        "passed": cleaned_dups <= original_dups,
        "message": f"Duplicate rows reduced from {original_dups} to {cleaned_dups} (reduced by {dup_reduction} rows)",
        "value": dup_reduction
    }

    # Rule 4: Column count stability check (should not arbitrarily increase or decrease columns)
    original_cols = len(original_info.get("columns", []))
    cleaned_cols = len(cleaned_info.get("columns", []))

    col_change = abs(cleaned_cols - original_cols)

    results["Column count stability"] = {
        "passed": col_change <= 2,  # Allow at most 2 columns change
        "message": f"Column count changed from {original_cols} to {cleaned_cols} (changed by {col_change} columns)",
        "value": col_change
    }

    return results


# ========== LLM Intelligent Assessment ==========
def run_llm_assessment(
    cleaning_plan: str,
    original_info: Dict[str, Any],
    cleaned_info: Dict[str, Any]
) -> str:
    """
    Use LLM for intelligent quality assessment

    Args:
        cleaning_plan: Cleaning strategy
        original_info: Original data metadata
        cleaned_info: Cleaned data metadata

    Returns:
        LLM's assessment report text
    """
    # Format input information
    original_shape = original_info.get("shape", "Unknown")
    original_nulls = original_info.get("total_nulls", "Unknown")
    original_duplicates = original_info.get("duplicate_count", "Unknown")
    original_columns = ", ".join(original_info.get("columns", []))

    cleaned_shape = cleaned_info.get("shape", "Unknown")
    cleaned_nulls = cleaned_info.get("total_nulls", "Unknown")
    cleaned_duplicates = cleaned_info.get("duplicate_count", "Unknown")
    cleaned_columns = ", ".join(cleaned_info.get("columns", []))

    # Build Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", QA_HUMAN_TEMPLATE)
    ])

    # Initialize LLM (DeepSeek)
    llm = ChatOpenAI(
        model="deepseek-chat",  # Use general dialogue model
        temperature=0.1,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
    )

    # Execute assessment
    chain = prompt | llm
    response = chain.invoke({
        "cleaning_plan": cleaning_plan,
        "original_shape": original_shape,
        "original_nulls": original_nulls,
        "original_duplicates": original_duplicates,
        "original_columns": original_columns,
        "cleaned_shape": cleaned_shape,
        "cleaned_nulls": cleaned_nulls,
        "cleaned_duplicates": cleaned_duplicates,
        "cleaned_columns": cleaned_columns,
    })

    return response.content


# ========== Helper Functions ==========
def calculate_quality_score(
    rule_results: Dict[str, Dict[str, Any]],
    llm_passed: bool
) -> int:
    """
    Calculate comprehensive quality score

    Args:
        rule_results: Rule check results
        llm_passed: Whether LLM assessment passed

    Returns:
        Quality score from 0-100
    """
    # Rule score (60 points)
    total_rules = len(rule_results)
    passed_rules = sum(1 for r in rule_results.values() if r["passed"])
    rule_score = (passed_rules / total_rules * 60) if total_rules > 0 else 0

    # LLM score (40 points)
    llm_score = 40 if llm_passed else 20

    return int(rule_score + llm_score)
