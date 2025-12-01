# Three Generations Seeking Romance — Age & Generation Prediction using Regression & Classification

## Using Machine Learning & NLP on the OKCupid Dataset

---

## 📌 Project Overview

This project explores whether a user's online dating profile can predict:

1.  **Their age** (Regression)
2.  **Their generation** — *Millennial, Gen X, or Boomer* (Classification)

Using a dataset of $\sim 60,000$ anonymized OKCupid user profiles, we apply data cleaning, feature engineering, NLP (TF–IDF), and machine learning models to build predictive systems for both tasks.

At the time the dataset was created $(\sim 2011–2012)$, generations were defined as:
* **Millennial:** 18–32
* **Gen X:** 33–47
* **Boomer:** 48–70

---

## 📂 Dataset

The OKCupid dataset contains both structured and unstructured features:

### **Structured (categorical/numeric):**
* age, sex, body\_type, diet, drinks, drugs
* education, job, ethnicity, religion
* height, income, orientation, status
* location, sign, smokes

### **Unstructured text:**
* **essay0 – essay9** (long-form personal descriptions)

The text features are later merged into a single field: `essay_all`.

---

## 🧹 1. Data Preparation

### **1.1 Missing Values**
* Filled numerical missing values:
    * `height` $\rightarrow$ median
    * `income` $\rightarrow -1$ (unknown)
* Filled categorical missing values with `"Unknown"`
* Filled essay fields with empty strings
* Dropped users with missing age values

### **1.2 Essay Consolidation**
* Combined all `essay0`–`essay9` columns into a unified text column: `essay_all`
* Removed the original essay columns

### **1.3 Generation Creation**
Created a new classification target using age ranges:

| Age Range | Generation |
| :--- | :--- |
| 18–32 | Millennial |
| 33–47 | Gen X |
| 48–70 | Boomer |

Encoded as `generation_encoded` for modeling.

---

## 🛠️ 2. Feature Engineering & Encoding

### **TF-IDF for Text** The combined essay text (`essay_all`) was converted to a numerical representation via:  
```python TfidfVectorizer(max_features=500, stop_words="english")```python

### **Numeric Features (Passthrough)**
Numeric variables such as height, income, etc. were passed directly to the model without any scaling, as **tree-based models do not require normalization**.

### **ColumnTransformer Workflow**
A unified `ColumnTransformer` was built to process all feature types simultaneously:
* Numeric features $\rightarrow$ passthrough
* Categorical features $\rightarrow$ `OneHotEncoder`
* Text feature (`essay_all`) $\rightarrow$ TF-IDF with 500 features

This automated preprocessing ensures consistent transformation during training and prediction.

🤖 ## 3. Modelleme

### **3.1 Train–Test Split**
Hem regresyon (yaş tahmini) hem de sınıflandırma (kuşak tahmini) görevleri için 80/20 bölmesi kullanıldı:
```python train_test_split(X, y, test_size=0.2, random_state=42)```python

### **3.2 Regression Models (Age Prediction)**
Regression models trained:
* Gradient Boosting Regressor
* Random Forest Regressor
* Linear Regression
* Metrics Used:
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)

➡️ Gradient Boosting Regressor performed the best.

### **3.3 Classification Models (Generation Prediction)**
Classification models trained:
* Random Forest Classifier
* Gradient Boosting Classifier
* Logistic Regression

Metric Used:
* Accuracy

➡️ Logistic Regression achieved the highest accuracy $(\sim 69\%)$.

## 📊 Visualizations

The notebook also includes:
* Age histogram
* Generation countplot
* Confusion matrices
* Performance comparison tables

These visuals help interpret both the dataset and model performance.

---

## 📈 Key Insights

* **Text features (essays) add significant predictive value** when processed with TF-IDF.
* Age prediction achieves an average error of **5–6 years**, which is strong considering noisy user-generated text.
* Generation classification achieves **$\sim 68–69\%$ accuracy**, indicating moderate predictability.
* Ensemble models (Gradient Boosting, Random Forest) perform consistently well.
* **Logistic Regression surprisingly outperforms tree models** for generation classification.

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TF-IDF vectorization
* Matplotlib, Seaborn
