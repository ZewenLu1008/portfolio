"""
Unit Tests for Executor Node Sandbox Execution

This module tests the executor node's ability to safely execute generated code
in an isolated sandbox environment without making LLM calls.
"""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.nodes.executor import executor_node, extract_df_info, validate_execution_safety
from src.core.state import DataCleaningState


class TestExecutorNode:
    """Test suite for executor_node sandbox execution"""

    def test_executor_executes_simple_cleaning_code_successfully(self, tmp_path):
        """
        Test that executor can execute simple Pandas cleaning code without LLM calls
        """
        # Arrange - Create temporary CSV file
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        # Create test data with missing values
        test_df = pd.DataFrame({
            "name": ["Alice", "Bob", None, "David"],
            "age": [25, None, 35, 40],
            "score": [85.5, 90.0, None, 88.0]
        })
        test_df.to_csv(input_file, index=False)

        # Mock state with safe cleaning code
        safe_cleaning_code = """
def clean_data(df):
    # Fill missing values
    df['name'] = df['name'].fillna('Unknown')
    df['age'] = df['age'].fillna(df['age'].median())
    df['score'] = df['score'].fillna(df['score'].mean())
    return df
"""

        state: DataCleaningState = {
            "generated_code": safe_cleaning_code,
            "input_file_path": str(input_file),
            "output_file_path": str(output_file),
            "retry_count": 0,
            "original_df_info": {},
            "cleaning_plan": None,
            "execution_error": None,
            "execution_success": False,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = executor_node(state)

        # Assert
        assert result["execution_success"] is True
        assert result["execution_error"] is None
        assert result["cleaned_df_info"] is not None
        assert output_file.exists()

        # Verify cleaned data
        cleaned_df = pd.read_csv(output_file)
        assert cleaned_df.shape[0] == 4
        assert cleaned_df["name"].isna().sum() == 0
        assert cleaned_df["age"].isna().sum() == 0
        assert cleaned_df["score"].isna().sum() == 0

    def test_executor_handles_code_with_syntax_error(self, tmp_path):
        """
        Test that executor captures syntax errors in generated code
        """
        # Arrange
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        test_df = pd.DataFrame({"col1": [1, 2, 3]})
        test_df.to_csv(input_file, index=False)

        # Code with actual syntax error (missing closing parenthesis)
        bad_code = """
def clean_data(df):
    df['new_col'] = df['col1'] * 2
    return df.fillna(0
"""

        state: DataCleaningState = {
            "generated_code": bad_code,
            "input_file_path": str(input_file),
            "output_file_path": str(output_file),
            "retry_count": 0,
            "original_df_info": {},
            "cleaning_plan": None,
            "execution_error": None,
            "execution_success": False,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = executor_node(state)

        # Assert - Should capture the error
        assert result["execution_success"] is False
        assert result["execution_error"] is not None
        assert "execution_history" in result
        assert result["execution_history"][0]["stage"] == "code_execution"

    def test_executor_handles_missing_clean_data_function(self, tmp_path):
        """
        Test that executor detects when clean_data function is missing
        """
        # Arrange
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        test_df = pd.DataFrame({"col1": [1, 2, 3]})
        test_df.to_csv(input_file, index=False)

        # Code without clean_data function
        code_without_function = """
# Just some random code
x = 10
y = 20
result = x + y
"""

        state: DataCleaningState = {
            "generated_code": code_without_function,
            "input_file_path": str(input_file),
            "output_file_path": str(output_file),
            "retry_count": 0,
            "original_df_info": {},
            "cleaning_plan": None,
            "execution_error": None,
            "execution_success": False,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = executor_node(state)

        # Assert
        assert result["execution_success"] is False
        assert "clean_data function not found" in result["execution_error"]

    def test_executor_handles_wrong_return_type(self, tmp_path):
        """
        Test that executor validates clean_data returns a DataFrame
        """
        # Arrange
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        test_df = pd.DataFrame({"col1": [1, 2, 3]})
        test_df.to_csv(input_file, index=False)

        # Code that returns wrong type
        code_with_wrong_return = """
def clean_data(df):
    return "This is not a DataFrame"
"""

        state: DataCleaningState = {
            "generated_code": code_with_wrong_return,
            "input_file_path": str(input_file),
            "output_file_path": str(output_file),
            "retry_count": 0,
            "original_df_info": {},
            "cleaning_plan": None,
            "execution_error": None,
            "execution_success": False,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = executor_node(state)

        # Assert
        assert result["execution_success"] is False
        assert "returned wrong type" in result["execution_error"]

    def test_executor_tracks_retry_count(self, tmp_path):
        """
        Test that executor correctly tracks retry attempts in execution history
        """
        # Arrange
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        test_df = pd.DataFrame({"col1": [1, 2, 3]})
        test_df.to_csv(input_file, index=False)

        safe_code = """
def clean_data(df):
    return df
"""

        state: DataCleaningState = {
            "generated_code": safe_code,
            "input_file_path": str(input_file),
            "output_file_path": str(output_file),
            "retry_count": 2,  # Third attempt
            "original_df_info": {},
            "cleaning_plan": None,
            "execution_error": None,
            "execution_success": False,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = executor_node(state)

        # Assert
        assert result["execution_history"][0]["attempt"] == 3


class TestExtractDfInfo:
    """Test suite for extract_df_info helper function"""

    def test_extract_df_info_captures_basic_metadata(self):
        """
        Test that extract_df_info captures shape, columns, and dtypes
        """
        # Arrange
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "score": [85.5, 90.0]
        })

        # Act
        info = extract_df_info(df)

        # Assert
        assert info["shape"] == (2, 3)
        assert info["columns"] == ["name", "age", "score"]
        assert "name" in info["dtypes"]
        assert "age" in info["dtypes"]
        assert "score" in info["dtypes"]

    def test_extract_df_info_captures_null_counts(self):
        """
        Test that extract_df_info correctly counts missing values
        """
        # Arrange
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": [None, None, 6],
            "col3": [7, 8, 9]
        })

        # Act
        info = extract_df_info(df)

        # Assert
        assert info["null_counts"]["col1"] == 1
        assert info["null_counts"]["col2"] == 2
        assert info["null_counts"]["col3"] == 0
        assert info["total_nulls"] == 3

    def test_extract_df_info_captures_duplicate_count(self):
        """
        Test that extract_df_info correctly counts duplicate rows
        """
        # Arrange
        df = pd.DataFrame({
            "col1": [1, 2, 1, 3],
            "col2": [10, 20, 10, 30]
        })

        # Act
        info = extract_df_info(df)

        # Assert
        assert info["duplicate_count"] == 1  # One duplicate row


class TestValidateExecutionSafety:
    """Test suite for validate_execution_safety function"""

    def test_validate_safe_pandas_code_passes(self):
        """
        Test that safe Pandas code passes validation
        """
        # Arrange
        safe_code = """
def clean_data(df):
    df['new_col'] = df['old_col'].fillna(0)
    df = df.drop_duplicates()
    return df
"""

        # Act
        is_safe = validate_execution_safety(safe_code)

        # Assert
        assert is_safe is True

    def test_validate_detects_eval_call(self):
        """
        Test that validation detects dangerous eval() call
        """
        # Arrange
        dangerous_code = """
def clean_data(df):
    eval('import os; os.system("rm -rf /")')
    return df
"""

        # Act
        is_safe = validate_execution_safety(dangerous_code)

        # Assert
        assert is_safe is False

    def test_validate_detects_system_call(self):
        """
        Test that validation detects os.system calls
        """
        # Arrange
        dangerous_code = """
import os
def clean_data(df):
    os.system('echo "dangerous"')
    return df
"""

        # Act
        is_safe = validate_execution_safety(dangerous_code)

        # Assert
        assert is_safe is False
