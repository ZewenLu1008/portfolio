# Zewen Lu's Portfolio

## About Me

I am a data scientist and AI engineer with a strong passion for building practical, scalable machine learning solutions. My expertise spans the full development lifecycle—from research and experimentation to deployment and production engineering. I don't just understand theoretical ML; I implement, deploy, and engineer AI systems that solve real-world problems.

My core interests lie in Data Science, Artificial Intelligence, and Large Language Model (LLM) Agents. I am driven by the challenge of translating complex algorithms into robust, maintainable code and identifying opportunities where automation can eliminate tedious, repetitive workflows.

---

## Projects

### Self-Driven Innovation

#### **Dataman: Enterprise-Grade Multi-Agent Data Cleaning & EDA Pipeline**
**Repository**: [`EDA_agent/`](./EDA_agent)

An autonomous, production-ready data preparation system built with LangGraph that transforms raw, heterogeneous data sources into analysis-ready datasets. Unlike simple LLM wrappers, Dataman implements a sophisticated multi-agent architecture with self-healing capabilities, deterministic quality gates, and defensive execution strategies designed for enterprise reliability.

**Architecture**:
```
Raw Data → Profiler → Coder ⇄ Executor → QA → EDA → Clean Data + Insights
                        ↑_________|
                     (Self-Correction Loop)
```

**Key Highlights**:
- **Multi-Source Data Fusion**: Ingests CSV, Excel, and PDFs with intelligent fallback—`pdfplumber` for native tables, automatic OCR degradation (`img2table` + Tesseract) for scanned documents. Handles schema misalignments via outer-join concatenation with defensive NaN filling.
- **Multi-agent Autonomous Workflow**: Profiler compresses raw data to metadata (avoids token limits), Coder generates defensive Pandas code (zero temperature), Executor runs in restricted namespace, QA implements hybrid validation (60% deterministic rules + 40% LLM-as-Judge), EDA enforces server-safe plotting.
- **Self-Healing Execution Loop**: Failed code execution automatically routes back to Coder with exact tracebacks. Implements retry logic with exponential backoff (max 3 attempts) to regenerate corrected code.
- **Production-Ready Engineering**: Custom state serialization for HITL readiness (NumPy/Pandas → native Python for `msgpack`), prompt injection defense via System/Human prompt separation, quality circuit breakers preventing >50% data loss.
- **Comprehensive Testing**: 36 unit tests (100% passing) with mocked LLM calls, Streamlit web UI with real-time progress tracking, tested at scale (100K+ rows).

**Tech Stack**: Python, LangGraph, LangChain, OpenAI/Anthropic/DeepSeek APIs, Pandas, pdfplumber, pdf2image, img2table, Tesseract OCR, Matplotlib, Seaborn, pytest

**Impact**: Reduces 4+ hours of manual data wrangling to <5 minutes of autonomous processing. Demonstrates enterprise-grade agentic AI workflows with safeguards against silent data corruption—bridging research prototypes and production systems.

---

#### **Nutrition Prediction**
**Repository**: [`nutrition_prediction/`](./nutrition_prediction)

A computer vision project focused on estimating caloric content from RGB and depth images. The project emphasizes architectural reasoning and feature fusion rather than transfer learning, with all networks initialized from scratch.

**Key Highlights**:
- **Dual-Branch Architecture**: Separates geometric (depth) and appearance (RGB) information through specialized convolutional branches, then fuses features via an adaptive weighting mechanism.
- **From-Scratch Design**: All networks randomly initialized to demonstrate deep understanding of CNN architecture and feature learning. No transfer learning is involved in this project.
- **Comprehensive Evaluation**: Includes training curves, validation scatter plots, error analysis, and feature importance visualization.
- **End-to-End Implementation**: Complete training pipeline with preprocessing, augmentation, model training, and Kaggle submission generation.

**Tech Stack**: Python, PyTorch, Torchvision, Pandas, NumPy, Matplotlib, Scikit-learn, Pillow

**Outcome**: Successfully predicted caloric content ($$R^2 \approx 0.81$$) from visual inputs, with detailed analysis of model behavior and error patterns documented in a comprehensive research report.

---

#### **Domain-Aware Text Classifier**
**Repository**: [`domain_aware_text_classifier/`](./domain_aware_text_classifier)

An academic project addressing the challenge of distinguishing machine-generated text from human writing across different topical domains. Built for a Kaggle competition, the classifier achieved 96% accuracy through domain-adaptive feature extraction.

**Key Highlights**:
- **Domain Adaptation Strategy**: Uses a two-stage pipeline where a logistic regression model first predicts the domain, then routes samples to domain-specific classifiers.
- **Sparse Text Classification**: Leverages TF-IDF feature extraction to handle high-dimensional text data efficiently.
- **Class Imbalance Handling**: Addresses underrepresented classes through domain-aware routing and targeted classification.

**Tech Stack**: Python, Pandas, Scikit-learn, SciPy, TF-IDF, Logistic Regression

**Outcome**: Demonstrated the effectiveness of domain adaptation in text classification, with practical application to AI-generated text detection.

---

#### **Road Accident Analysis and Injury Prediction**
**Repository**: [`road_accident_analysis/`](./road_accident_analysis)

A typical data science project analyzing real-world Victorian official records to investigate and demonstrate how seatbelt usage, seating position, and vehicle type influence injury severity. The project combines statistical analysis with supervised machine learning to inform road safety strategies.

**Key Highlights**:
- **Structured Data Pipeline**: End-to-end preprocessing including filtering, encoding, aggregation, and feature engineering.
- **Correlation Analysis**: Uses Normalized Mutual Information (NMI) and Pearson correlation to evaluate feature importance.
- **Multi-Model Comparison**: Implements and compares Logistic Regression, K-Nearest Neighbor, and Decision Tree classifiers with hyperparameter tuning.
- **Feature Impact Analysis**: Investigates how additional variables (e.g., hospital admission) improve prediction accuracy.

**Tech Stack**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy, Jupyter Notebook

**Outcome**: Comprehensive analysis with visualizations, confusion matrices, and classification reports that provide actionable insights for road safety policy.

---

## Technical Skills

**Programming Languages**: Python, SQL

**Machine Learning & AI**: PyTorch, Scikit-learn, TensorFlow/Keras, LangChain, LangGraph, Pandas, NumPy

**Computer Vision**: Convolutional Neural Networks, Image Processing, Multi-Modal Fusion

**Natural Language Processing**: TF-IDF, Text Classification, LLM Integration, Prompt Engineering

**LLM & Agent Frameworks**: OpenAI/Anthropic/DeepSeek API, Multi-Agent Systems, Agentic Workflows

**Data Science Tools**: Pandas, Matplotlib, Seaborn, Jupyter Notebook, Data Cleaning, Exploratory Data Analysis

**Deployment & Engineering**: End-to-End Pipeline Design, Code Generation, Automated Testing, Error Handling

---

## Contact

**LinkedIn**: [(19) Sebastian Lu | LinkedIn](https://www.linkedin.com/in/zewenlu-datascience/)

Feel free to explore my projects and reach out for collaboration opportunities in Data Science, AI Engineering, or LLM Agent development.
