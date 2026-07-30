"""
Unit Tests for Graph Routing Logic

This module tests the conditional routing functions in the LangGraph workflow
to ensure proper path selection based on state conditions.
"""
import pytest
from src.core.graph import route_after_executor, route_after_qa
from src.core.state import DataCleaningState


class TestRouteAfterExecutor:
    """Test suite for route_after_executor conditional routing"""

    def test_route_to_qa_when_execution_succeeds(self):
        """
        Test that successful execution routes to QA node
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "execution_error": None,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_executor(state)

        # Assert
        assert next_node == "qa"

    def test_route_to_coder_when_execution_fails_and_retries_available(self):
        """
        Test that failed execution with retry_count < 3 routes to coder for self-correction
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": False,
            "retry_count": 1,
            "execution_error": "SyntaxError: invalid syntax",
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_executor(state)

        # Assert
        assert next_node == "coder"

    def test_route_to_end_when_retry_limit_exceeded(self):
        """
        Test that execution failure with retry_count >= 3 routes to END
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": False,
            "retry_count": 3,
            "execution_error": "PersistentError: cannot fix",
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_executor(state)

        # Assert
        assert next_node == "end"

    def test_route_to_coder_on_first_failure(self):
        """
        Test that first execution failure (retry_count=0) routes to coder
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": False,
            "retry_count": 0,
            "execution_error": "ValueError: invalid operation",
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_executor(state)

        # Assert
        assert next_node == "coder"

    def test_route_to_coder_on_second_failure(self):
        """
        Test that second execution failure (retry_count=2) still routes to coder
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": False,
            "retry_count": 2,
            "execution_error": "TypeError: unsupported operand",
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "cleaned_df_info": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_executor(state)

        # Assert
        assert next_node == "coder"


class TestRouteAfterQA:
    """Test suite for route_after_qa conditional routing"""

    def test_route_to_eda_when_qa_passed(self):
        """
        Test that QA pass routes to EDA node
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "qa_result": {
                "passed": True,
                "score": 85,
                "reason": "Data cleaning successful",
                "issues": [],
                "suggestions": [],
            },
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "execution_error": None,
            "cleaned_df_info": {},
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_qa(state)

        # Assert
        assert next_node == "eda"

    def test_route_to_end_when_qa_failed(self):
        """
        Test that QA failure routes to END
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "qa_result": {
                "passed": False,
                "score": 45,
                "reason": "Excessive data loss detected",
                "issues": ["Data retention below threshold"],
                "suggestions": ["Review cleaning logic"],
            },
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "execution_error": None,
            "cleaned_df_info": {},
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_qa(state)

        # Assert
        assert next_node == "end"

    def test_route_to_end_when_qa_result_missing(self):
        """
        Test that missing qa_result defaults to failed QA and routes to END
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "qa_result": None,
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "execution_error": None,
            "cleaned_df_info": {},
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_qa(state)

        # Assert
        assert next_node == "end"

    def test_route_to_end_when_passed_field_missing(self):
        """
        Test that missing 'passed' field in qa_result defaults to False
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "qa_result": {
                "score": 50,
                "reason": "Incomplete QA result",
            },
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "original_df_info": {},
            "cleaning_plan": None,
            "generated_code": None,
            "execution_error": None,
            "cleaned_df_info": {},
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        next_node = route_after_qa(state)

        # Assert
        assert next_node == "end"
