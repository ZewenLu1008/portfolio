# Domain-Aware Text Classifier

With the rise of large language models, distinguishing machine-generated text from human writing has become a critical challenge. This project addresses that issue by building a domain-adaptive classifier using two distinct datasets, each containing labeled text from different topical domains. The task was part of a Kaggle competition focused on text generation detection, where final predictions achieved 96% accuracy.

## Overview

The classifier operates on tokenized text samples from two separate domains. To handle domain shift and class imbalance, the workflow integrates TF-IDF feature extraction with a domain-aware classification pipeline. A logistic regression model first predicts the domain of each test sample, which is then routed to a domain-specific classifier for final label prediction. This approach improves recall for underrepresented classes and demonstrates the effectiveness of domain adaptation in sparse text classification.

 ## File Structure

```
domain_aware_text_classifier/
├── main.py
├── dataset/
│   ├── domain1_train_data.json
│   ├── domain2_train_data.json
│   └── test_data.json
└── README.md
```

## How to run

This project is implemented as a standalone Python script: `domain_adaptation_classifier.py`. It performs domain-aware text classification using TF-IDF features and logistic regression.

### Prerequisite

Make sure you have **Python 3.7+** installed, along with the following libraries:

- `pandas`
- `scikit-learn`
- `scipy`

## Output

After running the script, a file named `prediction.csv` will be generated in the project root directory. It contains the final classification results for the test set, structured as follows:

```
prediction.csv
├── id      
└── class   
```

The output is designed for submission to a Kaggle competition, where it will be evaluated against ground truth labels to assess model accuracy. In the final submission, the model achieved an impressive 96% accuracy, demonstrating the effectiveness of domain-aware feature extraction and logistic regression in handling cross-domain text classification.