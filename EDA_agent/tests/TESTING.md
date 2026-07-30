# Testing Guide for EDA Agent

This document explains how to run the comprehensive unit test suite for the EDA Agent project.

## Test Suite Overview

The test suite includes 36 unit tests covering:

1. **State Initialization** (`test_state.py`) - 5 tests
   - Validates AgentState creation with correct default structures
   - Verifies data types and field initialization

2. **Graph Routing Logic** (`test_graph.py`) - 9 tests
   - Tests conditional routing after executor (success, retry, abort)
   - Tests conditional routing after QA (pass to EDA, fail to END)
   - Validates edge cases like missing qa_result

3. **Executor Sandbox** (`test_executor.py`) - 11 tests
   - Tests safe code execution in isolated environment
   - Validates error handling (syntax errors, missing functions, wrong types)
   - Tests DataFrame metadata extraction
   - Tests execution safety validation

4. **QA Deterministic Rules** (`test_qa_rules.py`) - 11 tests
   - Tests data retention rate threshold (≥50%)
   - Tests missing value improvement validation
   - Tests duplicate removal validation
   - Tests column count stability (≤2 column changes)
   - Integration tests with mocked LLM calls

## Prerequisites

Install test dependencies:

```bash
# Using uv (recommended)
uv pip install pytest pytest-mock pytest-cov

# Or using pip
pip install pytest pytest-mock pytest-cov
```

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run with verbose output
```bash
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_state.py -v
python -m pytest tests/test_graph.py -v
python -m pytest tests/test_executor.py -v
python -m pytest tests/test_qa_rules.py -v
```

### Run specific test class
```bash
python -m pytest tests/test_graph.py::TestRouteAfterExecutor -v
python -m pytest tests/test_qa_rules.py::TestDeterministicRuleChecks -v
```

### Run specific test function
```bash
python -m pytest tests/test_executor.py::TestExecutorNode::test_executor_executes_simple_cleaning_code_successfully -v
```

### Run with coverage report
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

## Test Design Principles

### 1. English-Only Policy
All test code, docstrings, comments, and variable names are 100% in English, maintaining strict consistency with the project's language policy.

### 2. No LLM API Calls
Tests use `unittest.mock` to patch LangChain LLM calls (`ChatOpenAI.invoke`, etc.), ensuring:
- Tests run instantly without network delays
- No API credits consumed during testing
- Deterministic test results

### 3. Isolated Test Environment
- Each test creates temporary files using `pytest`'s `tmp_path` fixture
- No shared state between tests
- Tests clean up after themselves automatically

### 4. Comprehensive Coverage
Tests cover:
- ✓ Happy path scenarios (successful execution)
- ✓ Error handling (syntax errors, missing functions, type errors)
- ✓ Edge cases (empty data, None values, boundary thresholds)
- ✓ Retry mechanisms and self-correction loops

## Test Results

All 36 tests pass successfully:

```
============================= 36 passed in 1.08s ==============================
```

### Test Breakdown by Category
- **State Tests**: 5/5 passed ✓
- **Graph Routing Tests**: 9/9 passed ✓
- **Executor Tests**: 11/11 passed ✓
- **QA Rules Tests**: 11/11 passed ✓

## Continuous Integration

To integrate tests into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python -m pytest tests/ -v --cov=src
```

## Bug Fixes Found During Testing

The test suite discovered and helped fix a bug in `src/core/graph.py`:

**Issue**: The `route_after_qa` function did not handle `qa_result=None` correctly.

**Fix**: Changed `state.get("qa_result", {})` to `state.get("qa_result") or {}` to properly handle None values.

## Contributing

When adding new features:
1. Write tests first (Test-Driven Development)
2. Ensure all tests pass before committing
3. Maintain 100% English-only code and comments
4. Mock all external API calls
5. Cover both success and failure scenarios
