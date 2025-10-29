# AI-ML-Internship-Task-6

##  Objective
To understand and implement the **K-Nearest Neighbors (KNN)** algorithm for classification problems using the **Iris dataset**.  
This project demonstrates data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.


##  Tools & Libraries Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**


##  Dataset
**Dataset Used:** [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
**File Name:** `Iris.csv`

| Feature | Description |
|----------|--------------|
| SepalLengthCm | Sepal length in cm |
| SepalWidthCm  | Sepal width in cm |
| PetalLengthCm | Petal length in cm |
| PetalWidthCm  | Petal width in cm |
| Species       | Target class (Iris-setosa, Iris-versicolor, Iris-virginica) |


##  Steps & Workflow
1. **Data Loading and Cleaning**
   - Loaded the `Iris.csv` file using Pandas.
   - Dropped unnecessary ID/index columns.
   - Encoded species labels numerically.

2. **Data Normalization**
   - Scaled features using `StandardScaler` for better performance.

3. **Model Training**
   - Used `KNeighborsClassifier` from `sklearn`.
   - Experimented with values of `K` (1–30).
   - Selected the best `K` based on accuracy.

4. **Evaluation Metrics**
   - Accuracy Score  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)

5. **Visualizations**
   - Accuracy vs K plot  
   - PCA-based Decision Boundaries  
   - Confusion Matrix Heatmap  
   - Multiclass ROC Curve


##  Results

| Metric | Value |
|--------|--------|
| Best K | 1 |
| Accuracy | 96.67% |
| Precision | 0.97 |
| Recall | 0.97 |
| F1-score | 0.97 |

**Confusion Matrix Example:**


[[10  0  0]
[ 0 10  0]
[ 0  1  9]]


##  Output Visualizations
Saved plots include:
- `accuracy_vs_k.png` – Accuracy comparison for different K values  
- `decision_boundaries_pca.png` – Visual decision boundaries after PCA projection  
- `confusion_matrix.png` – Heatmap of true vs predicted classes  
- `multiclass_roc.png` – ROC curves for each class


## How to Run the Project

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Run the Script

```bash
python "Knn Classification.py"
```

###  Output

All metrics and plots will be saved in your working directory.


##  Learning Outcomes

* Understood the working of **Instance-based Learning (KNN)**
* Learned to normalize data before training
* Explored how different K values affect model accuracy
* Practiced evaluating models using multiple metrics and visualizations


##  Conclusion

The KNN algorithm achieved **~97% accuracy** on the Iris dataset with **k = 1**.
This demonstrates that KNN performs effectively on small, well-separated datasets when features are properly normalized.



