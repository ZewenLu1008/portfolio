"""
Unit Tests for AgentState Initialization

This module tests the state factory and AgentState structure to ensure
that initial states are created correctly with expected default values.
"""
import pytest
from src.core.state import DataCleaningState, StateFactory


class TestStateInitialization:
    """Test suite for AgentState initialization"""

    def test_state_factory_creates_valid_initial_state(self):
        """
        Test that StateFactory creates a valid initial state with all required fields
        """
        # Arrange
        input_path = "data/test_input.csv"
        output_path = "outputs/test_output.csv"

        # Act
        state = StateFactory.create_initial_state(
            input_file_path=input_path,
            output_file_path=output_path
        )

        # Assert - Check all required fields exist
        assert "input_file_path" in state
        assert "output_file_path" in state
        assert "original_df_info" in state
        assert "cleaning_plan" in state
        assert "generated_code" in state
        assert "execution_error" in state
        assert "retry_count" in state
        assert "execution_success" in state
        assert "cleaned_df_info" in state
        assert "qa_result" in state
        assert "eda_plan" in state
        assert "eda_code" in state
        assert "eda_error" in state

    def test_initial_state_has_correct_default_values(self):
        """
        Test that initial state has correct default values for optional fields
        """
        # Arrange
        input_path = "data/test.csv"
        output_path = "outputs/test.csv"

        # Act
        state = StateFactory.create_initial_state(
            input_file_path=input_path,
            output_file_path=output_path
        )

        # Assert - Check default values
        assert state["input_file_path"] == input_path
        assert state["output_file_path"] == output_path
        assert state["cleaning_plan"] is None
        assert state["generated_code"] is None
        assert state["execution_error"] is None
        assert state["retry_count"] == 0
        assert state["execution_success"] is False
        assert state["cleaned_df_info"] is None
        assert state["qa_result"] is None
        assert state["eda_plan"] is None
        assert state["eda_code"] is None
        assert state["eda_error"] is None

    def test_state_retry_count_is_integer(self):
        """
        Test that retry_count is initialized as an integer
        """
        # Arrange & Act
        state = StateFactory.create_initial_state(
            input_file_path="data/test.csv",
            output_file_path="outputs/test.csv"
        )

        # Assert
        assert isinstance(state["retry_count"], int)
        assert state["retry_count"] >= 0

    def test_state_execution_success_is_boolean(self):
        """
        Test that execution_success is initialized as a boolean
        """
        # Arrange & Act
        state = StateFactory.create_initial_state(
            input_file_path="data/test.csv",
            output_file_path="outputs/test.csv"
        )

        # Assert
        assert isinstance(state["execution_success"], bool)
        assert state["execution_success"] is False

    def test_state_paths_are_preserved(self):
        """
        Test that input and output paths are preserved correctly
        """
        # Arrange
        input_path = "custom/path/input.csv"
        output_path = "custom/path/output.csv"

        # Act
        state = StateFactory.create_initial_state(
            input_file_path=input_path,
            output_file_path=output_path
        )

        # Assert
        assert state["input_file_path"] == input_path
        assert state["output_file_path"] == output_path
