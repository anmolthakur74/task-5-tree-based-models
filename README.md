# Task 5: Decision Trees and Random Forests

This repository contains my solution for **Task 5** of the AI & ML Internship. The focus of this task is to explore tree-based models—specifically Decision Trees and Random Forests—using the Heart Disease dataset. The goal is to understand model interpretability, overfitting control, ensemble methods, and feature importance.

---

## Objective

Learn tree-based models for classification & regression.

---

## Files Included

| File Name                  | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `heart.csv`               | Dataset used for classification                                  |
| `tree_based_models.ipynb` | Jupyter Notebook with all steps: training, tuning, visualization |
| `screenshots/`            | Folder containing plots of trees and feature importances         |
| `README.md`               | Project documentation                                            |

---

## 📌 What I Did

1. **Data Exploration**  
   Performed basic EDA using pandas to understand structure, check for null values, and examine the target distribution.

2. **Train/Test Split**  
   Divided data into features (X) and labels (y), with an 80/20 split for training and testing.

3. **Decision Tree Classifier**  
   - Trained a base decision tree using `DecisionTreeClassifier`.  
   - Visualized the tree using `plot_tree` to interpret splits and predictions.

4. **Controlling Overfitting**  
   - Tuned `max_depth` to limit the depth of the tree and avoid overfitting.  
   - Compared train and test accuracy across different depths.

5. **Random Forest Classifier**  
   - Trained a `RandomForestClassifier` to improve accuracy and reduce overfitting.  
   - Compared performance with the decision tree model.

6. **Feature Importance**  
   - Extracted feature importances using `.feature_importances_`  
   - Visualized top contributing features using bar plots.

7. **Model Evaluation**  
   - Used accuracy scores, confusion matrices, and cross-validation (`cross_val_score`) to assess model performance and robustness.

---

## Tools & Libraries Used

- Python 3.12  
- `pandas`, `numpy` — data handling  
- `matplotlib` — plotting  
- `scikit-learn` — model building and evaluation

---

## What I Learned

This task deepened my understanding of tree-based models. I learned:
- How decision trees make splits based on feature thresholds.
- How to visualize trees for better interpretability.
- Why random forests (via bagging) outperform single trees.
- How feature importance can guide insights in medical data.
- The trade-off between bias and variance when tuning tree depth.

---

## How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/anmolthakur74/task-5-tree-models.git
   cd task-5-tree-models
  ```

2. Install dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

3. Open the notebook:
  ```bash
  jupyter notebook tree_based_models.ipynb
  ```

## Author

**Anmol Thakur**

GitHub: [anmolthakur74]()
