"""
EDA Node - Exploratory Data Analysis Node

Features:
    Perform visual analysis on cleaned data
    Generate insight reports
    Use LLM to automatically generate plotting code
"""
import os
import re
import traceback
from typing import Dict, Any
import pandas as pd
from langchain_openai import ChatOpenAI

from ..core.state import DataCleaningState


def eda_node(state: DataCleaningState) -> Dict[str, Any]:
    """
    EDA Node - Data Insight Expert

    Core Logic:
    1. Check if QA passed, skip EDA if not
    2. Call LLM to generate visualization code and insights
    3. Execute code to generate charts
    4. Return updated state

    Args:
        state: Global state

    Returns:
        Updated state dictionary
    """
    print("\n" + "="*60)
    print("[EDA Node] Data Insight Expert")
    print("="*60)

    # 1. Check if QA passed
    qa_result = state.get("qa_result", {})
    if not qa_result.get("passed", False):
        print("[SKIP] QA did not pass, skipping EDA analysis")
        return {
            "eda_plan": "Skipped due to QA failure",
            "eda_code": None,
            "eda_error": "QA validation failed"
        }

    # 2. Prepare data information
    cleaned_df_info = state.get("cleaned_df_info", {})
    if not cleaned_df_info:
        print("[ERROR] No cleaned data info available")
        return {
            "eda_plan": None,
            "eda_code": None,
            "eda_error": "No cleaned data info available"
        }

    print(f"[INFO] Generating EDA plan for cleaned data...")
    print(f"  - Shape: {cleaned_df_info.get('shape', 'N/A')}")
    print(f"  - Columns: {len(cleaned_df_info.get('columns', []))}")

    # 3. Build Prompt
    prompt = _build_eda_prompt(cleaned_df_info)

    # 4. Call LLM
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.2,  # Slightly higher temperature for more creative visualizations
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com",
        )

        response = llm.invoke(prompt)
        llm_output = response.content

        print(f"[OK] LLM response received ({len(llm_output)} chars)")

    except Exception as e:
        error_msg = f"LLM invocation failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            "eda_plan": None,
            "eda_code": None,
            "eda_error": error_msg
        }

    # 5. Parse code and insights
    try:
        code, insights = _parse_llm_output(llm_output)
        print(f"[OK] Parsed code ({len(code)} chars) and insights")

    except Exception as e:
        error_msg = f"Failed to parse LLM output: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            "eda_plan": None,
            "eda_code": None,
            "eda_error": error_msg
        }

    # 6. Execute code to generate charts
    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)

    execution_error = _execute_plot_code(
        code=code,
        input_file=state.get("output_file_path", "outputs/cleaned_data.csv"),
        output_dir=output_dir
    )

    if execution_error:
        print(f"[ERROR] Plot generation failed")
        print(f"  Error: {execution_error[:200]}...")
        return {
            "eda_plan": insights,
            "eda_code": code,
            "eda_error": execution_error
        }

    print(f"[SUCCESS] Plots generated successfully")
    print(f"  Output directory: {output_dir}")

    return {
        "eda_plan": insights,
        "eda_code": code,
        "eda_error": None
    }


def _build_eda_prompt(cleaned_df_info: Dict[str, Any]) -> str:
    """
    Build EDA Prompt

    Args:
        cleaned_df_info: Cleaned data information

    Returns:
        Prompt string
    """
    shape = cleaned_df_info.get('shape', 'Unknown')
    columns = cleaned_df_info.get('columns', [])
    dtypes = cleaned_df_info.get('dtypes', {})
    sample_data = cleaned_df_info.get('sample_data', 'N/A')

    # Format column information
    columns_info = "\n".join([
        f"  - {col}: {dtypes.get(col, 'unknown')}"
        for col in columns
    ])

    prompt = f"""You are a Data Insight Expert. Your task is to create a comprehensive Exploratory Data Analysis (EDA) for the cleaned dataset.

CRITICAL LANGUAGE REQUIREMENT: You MUST output your response ENTIRELY in English. Do NOT output any Chinese characters, phrases, greetings, or explanations anywhere in your response. This is a strict requirement for the pipeline to function. Every comment, insight, label, and text must be in English only.

**Dataset Information:**
- Shape: {shape}
- Columns:
{columns_info}

**Sample Data (first 10 rows):**
```
{sample_data}
```

**Requirements:**
1. Generate Python code that creates exactly 3 informative plots:
   - Plot 1: Distribution plot for numeric columns (histogram/density)
   - Plot 2: Bar chart for categorical columns (top categories)
   - Plot 3: Correlation heatmap for numeric columns

2. CRITICAL CODE CONSTRAINTS - STRICTLY ENFORCE:
   - The code MUST be 100% in English (no Chinese characters, no Emoji symbols)
   - All function names, variable names, comments, plot titles, axis labels, and legends MUST be in English only
   - Use only ASCII characters to avoid matplotlib font rendering issues
   - The code must define a function: `generate_plots(df, output_dir)`
   - Save all plots as PNG files in the output_dir
   - Use seaborn and matplotlib for visualization
   - Handle edge cases (e.g., no numeric/categorical columns)

3. **MANDATORY DATA PREPROCESSING FOR CHINESE CHARACTERS (EXTREMELY CRITICAL):**
   - At the VERY BEGINNING of the `generate_plots` function, BEFORE any plotting operations, you MUST:
     a) Rename ALL Chinese column names to English equivalents using `.rename(columns={{...}})`
        Example: {{'Name_CN': 'Name', 'Age_CN': 'Age', 'Dept_CN': 'Department', 'Salary_CN': 'Salary'}}
     b) For categorical columns with Chinese values (like department names), use `.map()` or `.replace()` to translate them to English
        Example: df['Department'] = df['Department'].map({{'Sales_CN': 'Sales', 'Tech_CN': 'Tech', 'HR_CN': 'HR'}})
   - This ensures that NO Chinese characters will appear in matplotlib/seaborn plots, preventing font rendering errors
   - The translated column names and values should be meaningful English words that reflect the data semantics

4. **MANDATORY INSIGHTS LANGUAGE REQUIREMENT (100% ENGLISH ONLY):**
   - The business insights section MUST be written in 100% English
   - NO Chinese characters are allowed in the <INSIGHTS> section
   - Write in clear, professional English suitable for business reports
   - Provide business insights in Markdown format explaining:
     * What patterns each plot reveals
     * Key business insights from the data
     * Recommendations based on the analysis

**Output Format:**
Your response MUST be structured as follows:

<CODE>
[Put the complete Python code here - 100% English only, all comments in English]
</CODE>

<INSIGHTS>
[Put the Markdown insights here - MUST be 100% in English, NO Chinese characters allowed]
</INSIGHTS>

FINAL REMINDER: Your ENTIRE response (code comments, insights, everything) MUST be in English. No Chinese characters anywhere.

**Example Code Structure:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plots(df, output_dir):
    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Numeric distribution
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 4))
        # ... plot code ...
        plt.savefig(f"{{output_dir}}/plot_1_numeric_distribution.png")
        plt.close()

    # Plot 2: Categorical bar chart
    # ... similar structure ...

    # Plot 3: Correlation heatmap
    # ... similar structure ...
```

Now generate the EDA code and insights for the given dataset.
"""

    return prompt


def _parse_llm_output(llm_output: str) -> tuple[str, str]:
    """
    Parse LLM output, extract code and insights

    Args:
        llm_output: LLM's raw output

    Returns:
        (code, insights) tuple

    Raises:
        ValueError: If parsing fails
    """
    # Extract <CODE> tag content
    code_match = re.search(r'<CODE>\s*(.*?)\s*</CODE>', llm_output, re.DOTALL)
    if not code_match:
        raise ValueError("Cannot find <CODE> tag in LLM output")

    code = code_match.group(1).strip()

    # Remove code block markers (if present)
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'```\s*$', '', code)

    # Extract <INSIGHTS> tag content
    insights_match = re.search(r'<INSIGHTS>\s*(.*?)\s*</INSIGHTS>', llm_output, re.DOTALL)
    if not insights_match:
        raise ValueError("Cannot find <INSIGHTS> tag in LLM output")

    insights = insights_match.group(1).strip()

    return code, insights


def _execute_plot_code(code: str, input_file: str, output_dir: str) -> str | None:
    """
    Execute plotting code

    Args:
        code: Python code string
        input_file: Input cleaned data file path
        output_dir: Output chart directory

    Returns:
        Error message (if execution failed), otherwise None
    """
    print(f"[INFO] Executing plot generation code...")
    print(f"  - Input file: {input_file}")
    print(f"  - Output dir: {output_dir}")

    try:
        # Read cleaned data
        df = pd.read_csv(input_file)
        print(f"  - Loaded data: {df.shape}")

        # Create sandbox environment
        sandbox = {
            'pd': pd,
            'df': df,
            'output_dir': output_dir,
            '__builtins__': __builtins__,
        }

        # Dynamically import required libraries
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            sandbox['matplotlib'] = matplotlib
            sandbox['plt'] = plt
            sandbox['sns'] = sns
            sandbox['np'] = np
        except ImportError as e:
            return f"Missing required library: {str(e)}"

        # Execute code
        exec(code, sandbox)

        # Call generate_plots function
        if 'generate_plots' not in sandbox:
            return "Code does not define 'generate_plots' function"

        generate_plots_func = sandbox['generate_plots']
        generate_plots_func(df, output_dir)

        print(f"[OK] Plots generated successfully")
        return None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return error_msg
