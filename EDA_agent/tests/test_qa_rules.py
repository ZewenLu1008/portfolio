"""
Unit Tests for QA Node Deterministic Rule Checks

This module tests the deterministic quality assurance rules in the QA node
to ensure that hardcoded thresholds trigger pass/fail correctly.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.nodes.qa import qa_node, run_deterministic_checks
from src.core.state import DataCleaningState


class TestDeterministicRuleChecks:
    """Test suite for deterministic QA rule checks"""

    def test_successful_cleaning_passes_all_rules(self):
        """
        Test that successful cleaning with good metrics passes all deterministic rules
        """
        # Arrange - Simulate successful cleaning
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 50,
            "duplicate_count": 10
        }

        cleaned_info = {
            "shape": (95, 5),  # 95% retention (>50% threshold)
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 5,  # Reduced missing values
            "duplicate_count": 0  # Removed duplicates
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert - All rules should pass
        assert rule_results["Data retention rate"]["passed"] is True
        assert rule_results["Missing value improvement"]["passed"] is True
        assert rule_results["Duplicate data improvement"]["passed"] is True
        assert rule_results["Column count stability"]["passed"] is True

    def test_excessive_data_loss_fails_retention_rule(self):
        """
        Test that excessive data loss (>50%) triggers retention rate failure
        """
        # Arrange - Simulate massive data loss
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 20,
            "duplicate_count": 5
        }

        cleaned_info = {
            "shape": (40, 5),  # Only 40% retained (<50% threshold)
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 0,
            "duplicate_count": 0
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Data retention rate"]["passed"] is False
        assert rule_results["Data retention rate"]["value"] == 0.4
        assert "40.0%" in rule_results["Data retention rate"]["message"]

    def test_increased_missing_values_fails_null_improvement_rule(self):
        """
        Test that increased missing values fail the null improvement check
        """
        # Arrange - Missing values increased after cleaning (bad!)
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 10,
            "duplicate_count": 5
        }

        cleaned_info = {
            "shape": (95, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 20,  # Missing values increased!
            "duplicate_count": 0
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Missing value improvement"]["passed"] is False
        assert cleaned_info["total_nulls"] > original_info["total_nulls"]

    def test_increased_duplicates_fails_duplicate_improvement_rule(self):
        """
        Test that increased duplicate rows fail the duplicate improvement check
        """
        # Arrange - Duplicates increased (should never happen)
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 10,
            "duplicate_count": 5
        }

        cleaned_info = {
            "shape": (95, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 5,
            "duplicate_count": 8  # Duplicates increased!
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Duplicate data improvement"]["passed"] is False
        assert cleaned_info["duplicate_count"] > original_info["duplicate_count"]

    def test_excessive_column_changes_fails_stability_rule(self):
        """
        Test that adding/removing >2 columns fails the column stability check
        """
        # Arrange - Too many columns added
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 10,
            "duplicate_count": 5
        }

        cleaned_info = {
            "shape": (95, 9),  # Added 4 columns (>2 threshold)
            "columns": ["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9"],
            "total_nulls": 5,
            "duplicate_count": 0
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Column count stability"]["passed"] is False
        assert rule_results["Column count stability"]["value"] == 4

    def test_minimal_column_changes_passes_stability_rule(self):
        """
        Test that adding/removing <=2 columns passes the stability check
        """
        # Arrange - Add exactly 2 columns (at threshold)
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 10,
            "duplicate_count": 5
        }

        cleaned_info = {
            "shape": (95, 7),  # Added 2 columns (exactly at threshold)
            "columns": ["col1", "col2", "col3", "col4", "col5", "col6", "col7"],
            "total_nulls": 5,
            "duplicate_count": 0
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Column count stability"]["passed"] is True
        assert rule_results["Column count stability"]["value"] == 2

    def test_retention_rate_exactly_at_threshold_passes(self):
        """
        Test that exactly 50% retention rate passes the check (boundary test)
        """
        # Arrange - Exactly 50% retention
        original_info = {
            "shape": (100, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 10,
            "duplicate_count": 5
        }

        cleaned_info = {
            "shape": (50, 5),  # Exactly 50% retained
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 5,
            "duplicate_count": 0
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Data retention rate"]["passed"] is True
        assert rule_results["Data retention rate"]["value"] == 0.5

    def test_empty_original_data_fails_retention_rule(self):
        """
        Test that empty original data triggers retention rate failure
        """
        # Arrange - Original data is empty
        original_info = {
            "shape": (0, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 0,
            "duplicate_count": 0
        }

        cleaned_info = {
            "shape": (0, 5),
            "columns": ["col1", "col2", "col3", "col4", "col5"],
            "total_nulls": 0,
            "duplicate_count": 0
        }

        # Act
        rule_results = run_deterministic_checks(original_info, cleaned_info)

        # Assert
        assert rule_results["Data retention rate"]["passed"] is False
        assert "empty" in rule_results["Data retention rate"]["message"].lower()


class TestQANodeIntegration:
    """Test suite for QA node integration with mocked LLM calls"""

    @patch('src.nodes.qa.run_llm_assessment')
    def test_qa_node_passes_with_good_cleaning_and_mocked_llm(self, mock_llm):
        """
        Test that QA node passes when all rules pass and LLM is mocked to pass
        """
        # Arrange - Mock LLM to return positive assessment
        mock_llm.return_value = """## QA Results

**Judgment**: Pass

**Score**: 95

### Positive Indicators
- All data quality metrics improved significantly
- Missing values reduced from 50 to 5
- Duplicate rows completely removed

### Issue List
- None

### Improvement Suggestions
- Data quality is good
"""

        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "original_df_info": {
                "shape": (100, 5),
                "columns": ["col1", "col2", "col3", "col4", "col5"],
                "total_nulls": 50,
                "duplicate_count": 10
            },
            "cleaned_df_info": {
                "shape": (95, 5),
                "columns": ["col1", "col2", "col3", "col4", "col5"],
                "total_nulls": 5,
                "duplicate_count": 0
            },
            "cleaning_plan": "Remove duplicates and fill missing values",
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "generated_code": None,
            "execution_error": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = qa_node(state)

        # Assert
        assert result["qa_result"]["passed"] is True
        assert result["qa_result"]["score"] >= 60
        assert mock_llm.called

    @patch('src.nodes.qa.run_llm_assessment')
    def test_qa_node_fails_with_poor_cleaning_and_mocked_llm(self, mock_llm):
        """
        Test that QA node fails when rules fail even if LLM passes
        """
        # Arrange - Mock LLM to return positive assessment (but rules will fail)
        mock_llm.return_value = """## QA Results

**Judgment**: Pass

**Score**: 80
"""

        state: DataCleaningState = {
            "execution_success": True,
            "retry_count": 0,
            "original_df_info": {
                "shape": (100, 5),
                "columns": ["col1", "col2", "col3", "col4", "col5"],
                "total_nulls": 10,
                "duplicate_count": 5
            },
            "cleaned_df_info": {
                "shape": (30, 5),  # Only 30% retained - FAIL
                "columns": ["col1", "col2", "col3", "col4", "col5"],
                "total_nulls": 20,  # Nulls increased - FAIL
                "duplicate_count": 8  # Duplicates increased - FAIL
            },
            "cleaning_plan": "Clean the data",
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "generated_code": None,
            "execution_error": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = qa_node(state)

        # Assert - Should fail because rule_pass_rate < 1.0
        assert result["qa_result"]["passed"] is False
        assert result["qa_result"]["rule_pass_rate"] < 1.0
        assert len(result["qa_result"]["issues"]) > 0

    def test_qa_node_skips_when_execution_failed(self):
        """
        Test that QA node skips assessment when code execution failed
        """
        # Arrange
        state: DataCleaningState = {
            "execution_success": False,  # Execution failed
            "retry_count": 2,
            "execution_error": "SyntaxError in generated code",
            "original_df_info": {},
            "cleaned_df_info": None,
            "cleaning_plan": None,
            "input_file_path": "data/test.csv",
            "output_file_path": "outputs/test.csv",
            "generated_code": None,
            "qa_result": None,
            "eda_plan": None,
            "eda_code": None,
            "eda_error": None,
        }

        # Act
        result = qa_node(state)

        # Assert
        assert result["qa_result"]["passed"] is False
        assert result["qa_result"]["score"] == 0
        assert "execution failed" in result["qa_result"]["reason"].lower()
