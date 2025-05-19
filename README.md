# AI Project (Phase 2)

## 🎯 Objective
To analyze a retail dataset (`StoresData.csv`) and predict whether a store provides home delivery using supervised classification techniques. This is achieved by:
- Cleaning and preparing the dataset
- Engineering fuzzy logic-based features
- Training and tuning a decision tree classifier using hill-climbing and grid search
- Evaluating and visualizing the model performance

## 📂 Dataset Overview
- **Source**: FoodMart retail data  
- **Instances**: 151 records  
- **Features**: 28 columns (numerical, categorical, derived)  
- **Target**: `HomeDel (Num)` – a binary column indicating whether home delivery is offered (1 or 0)

## ⚙️ Preprocessing Steps
1. **Missing Value Handling**: Dropped all rows with missing values.  
2. **Categorical Encoding**: Applied LabelEncoder to columns like Location, State, etc.  
3. **Unnecessary Columns**: Removed `Unnamed: 26` and `Unnamed: 27` which contained only NaN.  
4. **Train-Test Split**: 70% for training and 30% for testing (stratified by class).  
5. **Feature Scaling**: Used MinMaxScaler on numerical features.

## 🤖 Fuzzy Feature Engineering
- Selected numerical features: `Sales $m` and `No. Staff`  
- Used triangular membership functions to create fuzzy logic variables (`low`, `medium`, `high`)  
- Appended these as new columns in the feature matrix

## 🌲 Model: Decision Tree Classifier

### ✅ Hill-Climbing (Local Search)
- Objective: Tune `max_depth` hyperparameter  
- Evaluated neighboring depth values and selected the one with highest training accuracy  
- Final depth chosen: **1**

### 🔍 Grid Search (Brute-Force)
- Explored `max_depth` values from 1 to 10  
- All values showed perfect training accuracy (likely due to data separability)

## 📊 Final Model Evaluation

| Metric     | Value |
|------------|-------|
| Accuracy   | 1.00  |
| Precision  | 1.00  |
| Recall     | 1.00  |

**Confusion Matrix:**
```
[[32, 0],
 [ 0, 13]]
```

## 🌳 Tree Visualization
- Used `plot_tree()` from `sklearn.tree`  
- Applied layout enhancements: `fontsize=10`, `figsize=(16,10)`, `rounded=True`, `tight_layout()`

## ✅ Highlights
- Demonstrated clean classification pipeline using a real-world dataset  
- Applied fuzzy logic for enhanced feature representation  
- Tuned model using both heuristic (hill-climbing) and exhaustive (grid search) approaches  
- Achieved interpretable results using decision trees with full visualization

## 📌 Recommendations
- Perform cross-validation to validate generalizability  
- Consider more advanced models (e.g., ensemble trees) for larger datasets  
- Explore additional fuzzy features for domain interpretability
