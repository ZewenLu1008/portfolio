# Zewen Lu's Portfolio

## About Me

I am a data scientist and AI engineer with a strong passion for building practical, scalable machine learning solutions. My expertise spans the full development lifecycle—from research and experimentation to deployment and production engineering. I don't just understand theoretical ML; I implement, deploy, and engineer AI systems that solve real-world problems.

My core interests lie in Data Science, Artificial Intelligence, and Large Language Model (LLM) Agents. I am driven by the challenge of translating complex algorithms into robust, maintainable code and identifying opportunities where automation can eliminate tedious, repetitive workflows.

---

## Projects

### Self-Driven Innovation

#### **Adaptive Data Cleaning & QA Agent**
**Repository**: [`EDA_agent/`](./EDA_agent)

A self-initiated project born from firsthand experience with the tedious and error-prone nature of exploratory data analysis and data cleaning. Leveraging my domain knowledge in data science and software engineering, I designed and built a multi-agent system using LangGraph that fully automates the data cleaning pipeline.

**Key Highlights**:
- **Autonomous Multi-Agent Architecture**: Implements a state-driven workflow with specialized nodes (Profiler, Coder, Executor, QA, EDA) that communicate and coordinate through LangGraph.
- **Self-Correction Mechanism**: Automatically retries failed data cleaning operations up to 3 times, using execution errors as feedback to regenerate improved Pandas code.
- **Intelligent Code Generation**: Uses LLMs (DeepSeek, OpenAI GPT-4, Anthropic Claude) to generate data cleaning code dynamically based on profiling insights.
- **Quality Assurance**: Implements deterministic rule checks combined with LLM-based assessment to validate data cleaning effectiveness.
- **End-to-End Automation**: From raw CSV input to cleaned data, visualizations, and comprehensive markdown reports—fully automated.

**Tech Stack**: Python, LangGraph, LangChain, Pandas, Matplotlib, Seaborn, OpenAI API, Anthropic API, DeepSeek API

**Impact**: Transforms hours of manual data wrangling into a single automated pipeline, demonstrating practical application of agentic AI workflows.

---

### Academic Rigor & Competitive Data Science

#### **Nutrition Prediction using Convolutional Networks**
**Repository**: [`nutrition_prediction/`](./nutrition_prediction)

A computer vision project developed as part of rigorous academic coursework, focused on estimating caloric content from RGB and depth images. The project emphasizes architectural reasoning and feature fusion rather than transfer learning, with all networks initialized from scratch.

**Key Highlights**:
- **Dual-Branch Architecture**: Separates geometric (depth) and appearance (RGB) information through specialized convolutional branches, then fuses features via an adaptive weighting mechanism.
- **From-Scratch Design**: All networks randomly initialized to demonstrate deep understanding of CNN architecture and feature learning.
- **Comprehensive Evaluation**: Includes training curves, validation scatter plots, error analysis, and feature importance visualization.
- **End-to-End Implementation**: Complete training pipeline with preprocessing, augmentation, model training, and Kaggle submission generation.

**Tech Stack**: Python, PyTorch, Torchvision, Pandas, NumPy, Matplotlib, Scikit-learn, Pillow

**Outcome**: Successfully predicted caloric content from visual inputs, with detailed analysis of model behavior and error patterns documented in a comprehensive research report.

---

#### **Domain-Aware Text Classifier**
**Repository**: [`domain_aware_text_classifier/`](./domain_aware_text_classifier)

An academic project addressing the challenge of distinguishing machine-generated text from human writing across different topical domains. Built for a Kaggle competition, the classifier achieved 96% accuracy through domain-adaptive feature extraction.

**Key Highlights**:
- **Domain Adaptation Strategy**: Uses a two-stage pipeline where a logistic regression model first predicts the domain, then routes samples to domain-specific classifiers.
- **Sparse Text Classification**: Leverages TF-IDF feature extraction to handle high-dimensional text data efficiently.
- **Class Imbalance Handling**: Addresses underrepresented classes through domain-aware routing and targeted classification.
- **Competition Success**: Achieved 96% accuracy on Kaggle competition test set.

**Tech Stack**: Python, Pandas, Scikit-learn, SciPy, TF-IDF, Logistic Regression

**Outcome**: Demonstrated the effectiveness of domain adaptation in text classification, with practical application to AI-generated text detection.

---

#### **Road Accident Analysis and Injury Prediction**
**Repository**: [`road_accident_analysis/`](./road_accident_analysis)

An academic data science project analyzing real-world Victorian crash records to understand how seatbelt usage, seating position, and vehicle type influence injury severity. The project combines statistical analysis with supervised machine learning to inform road safety strategies.

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

**LLM & Agent Frameworks**: OpenAI API, Anthropic Claude API, DeepSeek API, Multi-Agent Systems, Agentic Workflows

**Data Science Tools**: Pandas, Matplotlib, Seaborn, Jupyter Notebook, Data Cleaning, Exploratory Data Analysis

**Development Tools**: Git, GitHub, uv, pip, Virtual Environments

**Deployment & Engineering**: End-to-End Pipeline Design, Code Generation, Automated Testing, Error Handling

---

## Contact

**GitHub**: [github.com/ZewenLu1008](https://github.com/ZewenLu1008)

Feel free to explore my projects and reach out for collaboration opportunities in Data Science, AI Engineering, or LLM Agent development.
