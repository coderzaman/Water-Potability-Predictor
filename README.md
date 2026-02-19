# 💧 Water Potability Predictor (End-to-End ML Project)

This repository contains an end-to-end Machine Learning project designed to predict whether water is safe for human consumption (potable) based on its chemical and physical properties. The project covers the entire Machine Learning lifecycle, from data preprocessing and pipeline creation to model training and deployment as an interactive web application.

## 🚀 Live Demo & Notebook
* **Live Web App:** [Try the Predictor on Hugging Face Spaces](https://huggingface.co/spaces/coderaktar/water-potability-predictor)
* **Google Colab Notebook:** [View the Training Process & Code](https://colab.research.google.com/drive/1c6hIs2I4lFz7UR3mwTNaNOsUUTim3hUZ?usp=sharing)

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Machine Learning Library:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Gradio
* **Deployment Platform:** Hugging Face Spaces

---

## 📊 Project Workflow
1. **Data Preprocessing:** Handled missing values using Mean Imputation and capped outliers using the Interquartile Range (IQR) method to ensure data quality.
2. **Feature Engineering:** Categorized the `pH` values into three distinct groups (Acidic, Neutral, Alkaline) and applied One-Hot Encoding to improve model comprehension.
3. **Model Selection:** Evaluated multiple algorithms (Logistic Regression, Decision Tree, AdaBoost, and XGBoost). **Random Forest Classifier** was selected as the final model due to its superior accuracy.
4. **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` to find the optimal hyperparameters for the Random Forest model.
5. **Machine Learning Pipeline:** Built a robust Scikit-Learn `Pipeline` combining `StandardScaler` and the tuned `RandomForestClassifier` to automatically scale and predict new incoming user data.
6. **Web Application Deployment:** Developed an interactive user interface using Gradio and successfully deployed the packaged pipeline to Hugging Face Spaces.

---

## 💻 How to Run Locally

If you want to run this project on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/coderzaman/Water-Potability-Predictor.git](https://github.com/coderzaman/Water-Potability-Predictor.git)
cd Water-Potability-Predictor

```

**2. Create and activate a virtual environment (Recommended):**

```bash
python -m venv venv
# For Linux/Mac:
source venv/bin/activate  
# For Windows:
# venv\Scripts\activate

```

**3. Install required dependencies:**

```bash
pip install -r requirements.txt

```

**4. Run the Gradio App:**

```bash
python app.py

```

*Click on the local URL provided in the terminal (e.g., `http://127.0.0.1:7860`) to view the app in your browser.*

---

## 📂 Directory Structure

* `app.py`: The main script for the Gradio web application.
* `water_pipeline.pkl`: The exported Scikit-Learn pipeline (contains both the Scaler and the Random Forest Model).
* `requirements.txt`: List of required Python libraries to run the project.
* `Water_Quality_Prediction.ipynb`: The primary notebook containing data analysis, model training, and evaluation steps.

---

**Author:** Aktaruzzaman

*B.Sc. in Computer Science & Engineering*
