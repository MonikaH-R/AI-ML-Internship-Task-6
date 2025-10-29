"""
K-Nearest Neighbors (KNN) Classification - CSV (fixed)
- Drops Id column automatically
- Encodes string labels to integers (LabelEncoder)
- Fixes decision-boundary plotting (numeric Z values)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from itertools import cycle

# ------------------------
# CONFIGURATION
# ------------------------
DATA_PATH = r"C:\Users\HP\PycharmProjects\pythonProject\AI & ML INTERNSHIP\AI-ML-Internship-Task 6\Iris.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ------------------------
# Load dataset
# ------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Columns:", list(df.columns))

# Drop Id-like column if present
id_cols = [c for c in df.columns if c.lower() in ("id", "index") or c.lower().startswith("unnamed")]
if id_cols:
    print("Dropping id/index columns:", id_cols)
    df = df.drop(columns=id_cols)

# Detect label column: prefer species/target/class else last column
label_candidates = [c for c in df.columns if any(k in c.lower() for k in ("species", "target", "class", "label"))]
if label_candidates:
    LABEL_COLUMN = label_candidates[0]
else:
    LABEL_COLUMN = df.columns[-1]

print("Using label column:", LABEL_COLUMN)

X = df.drop(columns=[LABEL_COLUMN]).values
y = df[LABEL_COLUMN].values
print("Raw feature shape:", X.shape, "Raw labels shape:", y.shape)

# ------------------------
# Encode labels (if strings)
# ------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)  # numeric labels 0..n_classes-1
label_names = le.classes_
print("Label mapping (index -> name):")
for idx, name in enumerate(label_names):
    print(f"  {idx} -> {name}")

# ------------------------
# Preprocessing + split
# ------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

# ------------------------
# Search k = 1..30 and evaluate
# ------------------------
def evaluate_k(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    return {'k': k, 'test_acc': acc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}

results = [evaluate_k(k) for k in range(1, 31)]
results_df = pd.DataFrame(results)
best_k = int(results_df.loc[results_df['test_acc'].idxmax(), 'k'])
print(f"\nBest k selected by test accuracy: {best_k}")

# ------------------------
# Train final model (with numeric labels)
# ------------------------
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_names))

# ------------------------
# Plot: Accuracy vs k
# ------------------------
plt.figure(figsize=(8,5))
plt.plot(results_df['k'], results_df['test_acc'], 'o-', label='Test accuracy')
plt.plot(results_df['k'], results_df['cv_mean'], 'x--', label='CV mean')
plt.fill_between(results_df['k'], results_df['cv_mean']-results_df['cv_std'], results_df['cv_mean']+results_df['cv_std'], alpha=0.15)
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs k')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_k.png')
print("Saved: accuracy_vs_k.png")
plt.close()

# ------------------------
# Decision boundaries (PCA -> 2D). Use numeric classes for Z.
# ------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

viz_knn = KNeighborsClassifier(n_neighbors=best_k)
viz_knn.fit(X_train_2d, y_train)  # y_train are numeric (encoded) labels

# mesh
x_min, x_max = X_train_2d[:,0].min()-1, X_train_2d[:,0].max()+1
y_min, y_max = X_train_2d[:,1].min()-1, X_train_2d[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = viz_knn.predict(grid_points)
Z = Z.reshape(xx.shape).astype(float)  # numeric -> safe for contourf

plt.figure(figsize=(8,6))
# choose cmap with discrete colors
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
colors = cycle(['red','blue','green','orange','purple'])
for cls_idx in np.unique(y_train):
    idx = (y_train == cls_idx)
    plt.scatter(X_train_2d[idx,0], X_train_2d[idx,1], label=f"{cls_idx}: {label_names[cls_idx]}", edgecolor='k')
plt.legend()
plt.title(f"Decision boundaries (k={best_k}) on PCA-projected data")
plt.tight_layout()
plt.savefig('decision_boundaries_pca.png')
print("Saved: decision_boundaries_pca.png")
plt.close()

# ------------------------
# Confusion matrix heatmap (numeric labels)
# ------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i,j], 'd'), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved: confusion_matrix.png")
plt.close()

# ------------------------
# Multiclass ROC (one-vs-rest) — requires predict_proba
# ------------------------
if hasattr(final_model, "predict_proba"):
    classes = np.unique(y_enc)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = final_model.predict_proba(X_test)  # shape (n_samples, n_classes)
    n_classes = y_test_bin.shape[1]
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    plt.figure(figsize=(8,6))
    plt.plot(fpr['micro'], tpr['micro'], label=f"micro-average (AUC = {roc_auc['micro']:.2f})", linestyle=':')
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{i}: {label_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC (one-vs-rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('multiclass_roc.png')
    print("Saved: multiclass_roc.png")
    plt.close()
else:
    print("Model doesn't support predict_proba; skipping ROC plot.")

print("\nDone. Plots saved and metrics printed.")
