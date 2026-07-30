"""
Run Agent - Data Cleaning Agent Entry Point

Features:
    Orchestrate complete data cleaning workflow
    Display Agent execution results
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Environment variables loaded")
except ImportError:
    print("[WARNING] python-dotenv not installed, skipping .env file loading")
    print("   To use .env file, run: pip install python-dotenv")

# Check API Key
if not os.getenv("OPENAI_API_KEY"):
    print("\n" + "="*60)
    print("[WARNING] OPENAI_API_KEY environment variable not detected")
    print("="*60)
    print("Please ensure one of the following environment variables is set:")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("  - DEEPSEEK_API_KEY")
    print("\nMethod 1: Create .env file")
    print("  Create .env file in project root with content:")
    print("  OPENAI_API_KEY=sk-your-key-here")
    print("\nMethod 2: Set system environment variable")
    print("  export OPENAI_API_KEY=sk-your-key-here")
    print("="*60 + "\n")

from src.utils.data_loader import load_and_profile_data, print_data_summary
from src.core.state import StateFactory
from src.core.graph import app, visualize_graph


def print_banner():
    """Print startup banner"""
    print("\n" + "="*60)
    print("Adaptive Data Cleaning & QA Agent")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


def print_final_report(final_state: dict):
    """
    Print final execution report and save to file

    Args:
        final_state: Final state
    """
    # Use list to collect report content for both printing and saving
    report_lines = []

    def add_line(line=""):
        """Add a line to report"""
        report_lines.append(line)
        print(line)

    add_line("\n" + "="*60)
    add_line("Final Report")
    add_line("="*60)

    # 1. Execution status
    retry_count = final_state.get("retry_count", 0)
    execution_success = final_state.get("execution_success", False)

    add_line(f"\nExecution Status:")
    add_line(f"  - Retry Count: {retry_count}")
    add_line(f"  - Result: {'[SUCCESS]' if execution_success else '[FAILED]'}")

    if not execution_success:
        error = final_state.get("execution_error", "Unknown error")
        add_line(f"\nError Message:")
        add_line("-"*60)
        add_line(error[:500])  # Only print first 500 characters
        if len(error) > 500:
            add_line("...")
        add_line("-"*60)

    # 2. Data cleaning results
    if execution_success:
        original_info = final_state.get("original_df_info", {})
        cleaned_info = final_state.get("cleaned_df_info", {})

        add_line(f"\nData Cleaning Results:")
        add_line(f"  - Original: {original_info.get('shape', 'N/A')}")
        add_line(f"  - Cleaned: {cleaned_info.get('shape', 'N/A')}")
        add_line(f"  - Null values: {original_info.get('total_nulls', 0)} -> {cleaned_info.get('total_nulls', 0)}")
        add_line(f"  - Duplicates: {original_info.get('duplicate_count', 0)} -> {cleaned_info.get('duplicate_count', 0)}")

    # 3. QA results
    qa_result = final_state.get("qa_result")
    if qa_result:
        add_line(f"\nQA Results:")
        add_line(f"  - Status: {'[PASSED]' if qa_result.get('passed') else '[FAILED]'}")
        add_line(f"  - Score: {qa_result.get('score', 0)}/100")
        add_line(f"  - Reason: {qa_result.get('reason', 'N/A')}")

        issues = qa_result.get("issues", [])
        add_line(f"\n  Issues:")
        for issue in issues:
            add_line(f"    - {issue}")

        suggestions = qa_result.get("suggestions", [])
        add_line(f"\n  Suggestions:")
        for suggestion in suggestions:
            add_line(f"    - {suggestion}")

        # Display LLM Assessment Report
        llm_assessment = qa_result.get("llm_assessment")
        if llm_assessment:
            add_line(f"\n  LLM Assessment Report:")
            add_line("-"*60)
            add_line(llm_assessment)
            add_line("-"*60)

    # 4. EDA results - Complete display of Insights (no truncation)
    eda_error = final_state.get("eda_error")
    eda_plan = final_state.get("eda_plan")

    if eda_plan or eda_error:
        add_line(f"\nEDA Results:")

        if eda_error:
            add_line(f"  - Status: [FAILED]")
            add_line(f"  - Error: {eda_error[:200]}...")
        else:
            add_line(f"  - Status: [SUCCESS]")

        # Complete display of insights (no truncation)
        if eda_plan and eda_plan != "Skipped due to QA failure":
            add_line(f"\n  Business Insights:")
            add_line("-"*60)
            add_line(eda_plan)
            add_line("-"*60)

    add_line("\n" + "="*60)

    # 7. Save report to file
    try:
        report_content = "\n".join(report_lines)
        report_path = "outputs/final_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n[SUCCESS] Complete execution report saved to: {report_path}")
    except Exception as e:
        print(f"\n[WARNING] Report save failed: {str(e)}")


def main():
    """Main function"""
    # 1. Print startup banner
    print_banner()

    # 2. Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    print("[OK] Output directories created")

    # 3. Load data
    input_file = "data/dirty_data.csv"
    output_file = "outputs/cleaned_data.csv"

    print(f"[INFO] Loading data: {input_file}")

    try:
        original_df_info = load_and_profile_data(input_file)
    except Exception as e:
        print(f"\n[ERROR] Data loading failed: {str(e)}")
        print("\nPlease generate test data first:")
        print("  python scripts/generate_dirty_data.py")
        return

    # Print data summary
    print_data_summary(original_df_info)

    # 3. Initialize state
    print("\n[INFO] Initializing Agent state...")
    initial_state = StateFactory.create_initial_state(
        input_file_path=input_file,
        output_file_path=output_file
    )
    initial_state["original_df_info"] = original_df_info

    # 4. Optional: Visualize workflow graph
    try:
        visualize_graph("docs/graph.mmd")
    except Exception as e:
        print(f"[WARNING] Visualization skipped: {str(e)}")

    # 5. Run Agent
    print("\n" + "="*60)
    print("[INFO] Starting Agent workflow...")
    print("="*60)

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        print(f"\n[ERROR] Agent execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 6. Print final report
    print_final_report(final_state)

    # 7. End
    print("\n[SUCCESS] Agent execution completed!")


if __name__ == "__main__":
    main()
