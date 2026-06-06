# Car Price Prediction & Classification Pipeline

An end-to-end Machine Learning pipeline developed to solve two distinct real-world business problems using a single dataset of used cars. This project transitions from pure data analysis to proactive predictive modeling.

## 📌 Business Problem & Objective
In the used car market, pricing vehicles and targeting the right customer segments often relies on intuition or unorganized historical data. Real-world market data is highly skewed, noisy, and full of outliers. 

For marketing and sales teams, relying on raw data leads to inaccurate budgeting, misplaced ads, and wasted ad spend. 

**The Goal:** Build a robust, end-to-end Machine Learning pipeline to:
1. Predict the exact selling price of a car (Regression) to guide pricing strategy.
2. Automatically categorize cars into balanced price segments (Classification) to optimize targeted marketing campaigns and maximize ROI without human bias.

---

## 🛠️ Project Workflow & Technical Challenges

### 1. Exploratory Data Analysis (EDA)
* **Dataset Scale:** 72,435 rows and 10 columns.
* **Target Distribution:** Car prices were highly **Right-Skewed**.
* **Missing Values:** Identified a uniform missingness pattern (~3,600 missing values per column across the dataset).
* **Feature Correlation:** Found that `engineSize` (0.63) and `year` (0.52) have the strongest positive correlation with price.

### 2. Data Preprocessing & Pipeline Engineering
To optimize model performance, the raw data underwent strict preprocessing steps:
* **Imputation:** Missing values in numerical features were filled using the `median`, while categorical features used the `mode` to preserve distributions.
* **The High Cardinality Challenge:** Categorical features (Make, Transmission, FuelType, Model) had no natural order, making **One-Hot Encoding** necessary. However, the `model` column suffered from high cardinality (too many unique values). Applying standard encoding would cause the *Curse of Dimensionality* and destroy model efficiency.
  * *Solution:* Analyzed the cumulative percentage of the data, grouped the rarest models representing the bottom 10% into a single category named `"other"`, and then safely applied One-Hot Encoding.
* **Feature Scaling & Outliers:** Applied `StandardScaler` to numerical features since their scales varied drastically (crucial for distance-based algorithms like KNN). Outliers were handled using statistical capping.

### 3. Model Training & Strategy
* **Regression (Price Prediction):** Trained a **Linear Regression** model to predict the continuous price target.
* **Classification (Targeted Segments):** Instead of manually picking arbitrary thresholds, used **Quantile Binning** (`pd.qcut`) to automatically split prices into three exact, perfectly balanced thirds: `['Cheap', 'Moderate', 'Expensive']`. This ensures fair data distribution for targeted campaigns.
* **Hyperparameter Tuning:** Trained a **K-Nearest Neighbors (KNN)** classifier. Utilized `GridSearchCV` combined with `K-Fold Cross-Validation` to automatically find the best combination of neighbors ($K$) and distance metrics (*Euclidean* vs. *Manhattan*).

---

## 📊 Evaluation & Results

The rigorous preprocessing and tuning directly translated into high-performing, reliable models:

### Linear Regression (Predicting Price)
* **$R^2$ Score:** `0.80` (Explains 80% of the variance in car prices).
* **MSE:** `0.13`

### KNN Classifier (Predicting Price Category)
* **Accuracy:** `0.85` (85% correct classifications).
* **Business Evaluation:** The evaluation focused heavily on the **Confusion Matrix** (Precision and Recall) to minimize *Class Bleeding*. Misclassifying a "Cheap" car as "Expensive" (*False Positive*) results in direct financial waste in ad targeting. This pipeline minimizes those errors to the lowest possible threshold.

---

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Google Colab

## 💡 Key Takeaway
This project demonstrates that Machine Learning is not just about importing libraries and running default algorithms. The true business value comes from data engineering choices—knowing how to mathematically handle high cardinality, scaling data safely, and aligning evaluation metrics with real-world business risks.
