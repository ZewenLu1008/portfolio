# Nutrition Prediction based on Convolution Network

Estimating physical quantities from images is a core aim of computer vision. In this research, we predict a meal’s calories from color and depth images, emphasizing design reasoning over reuse of pre-trained models (all networks are randomly initialized). We separate geometry and appearance via depth and RGB branches, then fuse their features to produce a single regression output. This separation is motivated by the different roles of volume (depth) and composition (RGB). We document the step-by-step experiments that shaped the architecture and regularization choices, and report clear metrics and visualizations. Finally, we analyze errors, especially underestimation for layered dishes—and outline practical improvements (better fusion, depth cleaning, and targeted augmentations).

## Overview

The **VolumeCompositionNet** model is designed to predict caloric content from visual input by combining depth-based volume estimation and RGB-based food composition recognition. 
The model architecture consists of:

- A **Volume Estimation Branch** that processes depth images to extract geometric and volumetric cues.  
- A **Composition Recognition Branch** that processes RGB images to learn color, texture, and ingredient-level information.  
- An **Adaptive Fusion Layer** that learns a weighting parameter α to dynamically balance the contributions of both branches.

The fused feature representation is passed through a multi-layer regression head to estimate total calories. This late-fusion design allows the model to capture both the physical and visual properties of food, improving prediction robustness.

---

## File Structure

```
submission/
├── main.py # Cuda is required for speedup
├── Nutrition_Prediction.pdf # Final report
├── requirements.txt # Python dependencies
├── output/
│   ├── submission_final.csv # Generated test predictions for Kaggle submission
│   ├── best_model.pth # Parameters of the best-performing model
│   ├── training_curves.png # Loss and metric visualization 
│   ├── val_scatter.png # Predicted vs. true calorie plot
│   ├── error_samples.png # Visualization of top/bottom error examples  
│   └── feature_importance.png # Correlation between feature norms and calorie labels 
└── README.md
```

## Dataset Setup

Download the **Nutrition5K** dataset from the link below:
[COMP90086_Nutrition5K | Kaggle](https://www.kaggle.com/competitions/comp-90086-nutrition-5-k/data)

## How to Run

Before running the Python script, update the dataset paths in `FinalProject_G104.py`:

```
data_dir = "path/to/Nutrition5K/train"
csv_file = "path/to/Nutrition5K/nutrition5k_train.csv"
test_dir = "path/to/Nutrition5K/test"
```

This project is implemented as a standalone Python script:  
`main.py`  

It trains the model, evaluates performance, and produces calorie predictions for the Nutrition5K test set.

### Prerequisite

Make sure you have **Python 3.8+** installed, along with the following libraries:

- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `Pillow`
- `tqdm`

You can install all dependencies with:
```bash
pip install -r requirements.txt
```

## Output

After execution, a folder named `outputs_x/` (e.g., `outputs_0/`) will be created with the following content:

```
outputs_x/
├── best_model.pth
├── training_curves.png
├── val_scatter.png
├── error_samples.png
├── feature_importance.png
└── submission_final.csv
```

The final predictions are saved in `submission_final.csv` for Kaggle submission.
 Example format:

```
submission.csv
├── ID
└── Value
```