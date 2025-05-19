# PHASE 2 AI PROJECT - DECISION TREE CLASSIFICATION ON STORES DATASET

import matplotlib
matplotlib.use('TkAgg')  # Fix for PyCharm plotting error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 1. Load Dataset
df = pd.read_csv("D:\\Downloads\\Chrome Downloads\\StoresData.csv")
df = df.drop(columns=["Unnamed: 26", "Unnamed: 27"], errors='ignore')
df.dropna(inplace=True)

# 2. Encode Categorical Variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Set Target and Features
target_col = "HomeDel (Num)"  # Target for classification
X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 5. Exploratory Plots
sns.pairplot(pd.concat([X_train[['Sales $m', 'No. Staff']], y_train], axis=1))
plt.suptitle("Pairplot of Selected Features and Target", y=1.02)
plt.show()

X_train.hist(figsize=(12, 8))
plt.suptitle("Histograms of Training Features")
plt.tight_layout()
plt.show()

# 6. MinMax Scaling
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 7. Fuzzy Feature Creation (Triangular Membership)
def triangular_membership(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)), 0)

fuzzy_features = ['Sales $m', 'No. Staff']
for feature in fuzzy_features:
    min_val, max_val = X_train_scaled[feature].min(), X_train_scaled[feature].max()
    mid_val = (min_val + max_val) / 2

    for label in ['low', 'medium', 'high']:
        if label == 'low':
            X_train_scaled[f'{feature}_{label}'] = triangular_membership(X_train_scaled[feature], min_val, min_val, mid_val)
            X_test_scaled[f'{feature}_{label}'] = triangular_membership(X_test_scaled[feature], min_val, min_val, mid_val)
        elif label == 'medium':
            X_train_scaled[f'{feature}_{label}'] = triangular_membership(X_train_scaled[feature], min_val, mid_val, max_val)
            X_test_scaled[f'{feature}_{label}'] = triangular_membership(X_test_scaled[feature], min_val, mid_val, max_val)
        else:
            X_train_scaled[f'{feature}_{label}'] = triangular_membership(X_train_scaled[feature], mid_val, max_val, max_val)
            X_test_scaled[f'{feature}_{label}'] = triangular_membership(X_test_scaled[feature], mid_val, max_val, max_val)

# 8. Hill-Climbing for max_depth
def hill_climb(X, y, max_iter=10):
    current_depth = 1
    best_score = 0
    path = []

    for _ in range(max_iter):
        scores = {}
        for d in [current_depth - 1, current_depth, current_depth + 1]:
            if d < 1 or d > 10:
                continue
            clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            clf.fit(X, y)
            score = clf.score(X, y)
            scores[d] = score

        best_candidate = max(scores, key=scores.get)
        path.append((best_candidate, scores[best_candidate]))

        if scores[best_candidate] > best_score:
            best_score = scores[best_candidate]
            current_depth = best_candidate
        else:
            break

    return current_depth, path

hill_best_depth, hill_path = hill_climb(X_train_scaled, y_train)

# 9. Grid Search for max_depth
grid_scores = {}
for d in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train_scaled, y_train)
    grid_scores[d] = clf.score(X_train_scaled, y_train)

grid_best_depth = max(grid_scores, key=grid_scores.get)

# 10. Train Final Model
final_model = DecisionTreeClassifier(max_depth=hill_best_depth, random_state=42)
final_model.fit(X_train_scaled, y_train)

# 11. Visualize Final Tree (Improved Appearance)
plt.figure(figsize=(16, 10))  # Adjusted figure size
plot_tree(
    final_model,
    filled=True,
    feature_names=X_train_scaled.columns,
    class_names=['No Delivery', 'Delivery'],
    rounded=True,
    fontsize=10  # Smaller font
)
plt.title("Final Decision Tree", fontsize=14)
plt.tight_layout()
plt.show()


# 12. Evaluate on Test Set
y_pred = final_model.predict(X_test_scaled)
print("=== Final Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 13. Show Search Results
print("\nHill-Climbing Path:", hill_path)
print("\nGrid Search Scores:", grid_scores)