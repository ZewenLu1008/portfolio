import os
import shutil
import tempfile
import streamlit as st
import pandas as pd

# Import project components
from src.utils.data_ingestion import load_and_merge_data
from scripts.run_agent import main as run_agent_pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Multi-Source EDA Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Multi-Source Automated Data Cleaning & EDA Agent")
st.markdown(
    "Upload raw datasets (CSV, Excel, or PDF) to let the Multi-Agent system clean, validate, and generate EDA insights.")

# 2. Session State Initialization
# Streamlit re-runs the whole script on every interaction.
# We store execution results in st.session_state to avoid redundant LLM calls.
if "execution_result" not in st.session_state:
    st.session_state["execution_result"] = None
if "merged_df" not in st.session_state:
    st.session_state["merged_df"] = None
if "eda_page" not in st.session_state:
    st.session_state["eda_page"] = 0

# 3. Sidebar - Input & Configuration
with st.sidebar:
    st.header("1. Input Data")
    uploaded_files = st.sidebar.file_uploader(
        "Choose raw data files",
        type=["csv", "xlsx", "pdf"],
        accept_multiple_files=True
    )

    st.header("2. Execution")
    start_button = st.button("Run EDA Agent", type="primary", disabled=not uploaded_files)

# 4. Handle File Processing and Agent Execution
if start_button and uploaded_files:
    # Reset session state
    st.session_state["execution_result"] = None
    st.session_state["merged_df"] = None
    st.session_state["eda_page"] = 0  # Reset EDA page counter

    # Create a temporary directory for raw file processing
    temp_dir = tempfile.mkdtemp()

    try:
        # Save uploaded files into temp directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Step A: Data Ingestion
        with st.status("Ingesting and Merging Data Sources...", expanded=True) as status:
            st.write("Extracting tables from uploaded files (CSV/Excel/PDF)...")
            merged_df, ingestion_metadata = load_and_merge_data(temp_dir)
            st.session_state["merged_df"] = merged_df

            # Display ingestion metadata
            st.write(f"Processed {ingestion_metadata['total_files_processed']} file(s)")
            st.write(f"CSV: {ingestion_metadata['file_counts']['csv']}, Excel: {ingestion_metadata['file_counts']['excel']}, PDF: {ingestion_metadata['file_counts']['pdf']}")
            if ingestion_metadata['warnings']:
                st.warning(f"Warnings: {len(ingestion_metadata['warnings'])} issue(s) detected")

            status.update(label="Data Ingestion Complete!", state="complete", expanded=False)

        # Step B: Agent Execution Pipeline
        with st.status("Agent Workflow Execution in Progress...", expanded=True) as status:
            st.write("🤖 Profiler: Diagnosing raw data issues...")
            st.write("💻 Coder: Generating python cleaning script...")
            st.write("🛡️ Executor: Running code in sandbox...")
            st.write("⚖️ QA Node: Validating retention rate and data rules...")

            # Execute the graph
            # Note: Ensure temp_dir is passed to the ingestion step inside agent execution if needed
            result = run_agent_pipeline(raw_data_dir=temp_dir)
            st.session_state["execution_result"] = result

            status.update(label="Agent Workflow Finished Successfully!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
    finally:
        # Cleanup temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

# 5. Display Results
if st.session_state["merged_df"] is not None:
    st.subheader("Raw Data Preview")
    st.dataframe(st.session_state["merged_df"].head(10), width="stretch")

if st.session_state["execution_result"] is not None:
    res = st.session_state["execution_result"]

    st.divider()
    st.subheader("Cleaning Metrics & QA Summary")

    # Metric Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Rows", res.get("orig_rows", "N/A"))
    col2.metric("Cleaned Rows", res.get("cleaned_rows", "N/A"))
    col3.metric("Null Values Reduced", f"{res.get('orig_nulls', 0)} -> {res.get('cleaned_nulls', 0)}")
    col4.metric("QA Status", res.get("qa_status", "UNKNOWN"))

    # Detailed Reports
    tab1, tab2, tab3 = st.tabs(["LLM Assessment Report", "EDA Analysis", "Cleaned Data View"])

    with tab1:
        st.markdown(res.get("qa_report", "No report available."))

    with tab2:
        # Create a stable wrapper container to prevent tab jumping on layout changes
        eda_wrapper = st.container()

        with eda_wrapper:
            eda_report = res.get("eda_report")

            # Check if eda_report is a dictionary (structured JSON format)
            if isinstance(eda_report, dict) and "summary" in eda_report and "plots" in eda_report:
                # Filter plots to only include those that actually exist on disk
                all_plots = eda_report["plots"]
                valid_plots = []
                for plot in all_plots:
                    filename = plot.get("filename", "")
                    img_path = os.path.join("outputs/charts", filename)
                    if os.path.exists(img_path):
                        valid_plots.append(plot)
                    else:
                        print(f"[WARNING] Skipping missing plot: {filename}")

                # Structured JSON format - implement pagination
                total_pages = 1 + len(valid_plots)  # Page 0 = summary, rest = individual plots
                current_page = st.session_state.get("eda_page", 0)

                # Render current page content
                if current_page == 0:
                    # Page 0: Summary page
                    st.subheader("EDA Summary")
                    st.markdown(eda_report["summary"])
                    st.info(f"Total visualizations: {len(valid_plots)}")
                else:
                    # Page 1+: Individual plot pages
                    plot_index = current_page - 1
                    if plot_index < len(valid_plots):
                        plot_data = valid_plots[plot_index]
                        filename = plot_data.get("filename", "")
                        interpretation = plot_data.get("interpretation", "No interpretation available.")

                        st.subheader(f"Visualization {current_page} of {len(valid_plots)}")

                        # Side-by-side layout: image on left, interpretation on right
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            img_path = os.path.join("outputs/charts", filename)
                            st.image(img_path, use_container_width=True)

                        with col2:
                            st.markdown("### Analysis")
                            st.markdown(interpretation)

                # Pagination UI at bottom
                st.divider()
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    if st.button("Previous", disabled=(current_page == 0), use_container_width=True):
                        st.session_state["eda_page"] = max(0, current_page - 1)
                        st.rerun()

                with col2:
                    st.markdown(f"<div style='text-align: center; padding-top: 8px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)

                with col3:
                    if st.button("Next", disabled=(current_page >= total_pages - 1), use_container_width=True):
                        st.session_state["eda_page"] = min(total_pages - 1, current_page + 1)
                        st.rerun()

            else:
                # Fallback: display as plain text (for backward compatibility)
                st.markdown(res.get("eda_report", "No EDA report available."))

                # Render charts if saved in output directory (old format)
                output_img_dir = "outputs/charts"
                if os.path.exists(output_img_dir):
                    chart_files = [os.path.join(output_img_dir, f) for f in os.listdir(output_img_dir) if
                                   f.endswith(('.png', '.jpg'))]
                    if chart_files:
                        st.subheader("Generated EDA Visualizations")
                        for img_path in chart_files:
                            st.image(img_path, use_container_width=True)

    with tab3:
        cleaned_df = res.get("cleaned_df")
        if cleaned_df is not None:
            st.info(f"Dataset Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")

            # Show data preview (20 rows for UI performance)
            st.dataframe(cleaned_df.head(20), width="stretch")

            # Export CSV - ALWAYS use full dataset, never .head()
            # This download provides the COMPLETE cleaned dataset even if QA failed
            csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

            # Show per-column missing value breakdown for debugging
            missing_vals = cleaned_df.isna().sum()
            if missing_vals.sum() > 0:
                st.warning(f"⚠️ Total Missing Values: {missing_vals.sum()}")
                with st.expander("Show Missing Values by Column"):
                    for col, count in missing_vals.items():
                        if count > 0:
                            st.write(f"- **{col}**: {count} missing")
        else:
            st.warning("No cleaned data available. The pipeline may have failed before completion.")