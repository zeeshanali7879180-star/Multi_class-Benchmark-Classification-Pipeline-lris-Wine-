🌸 Iris Multiclass Classification Pipeline

A Machine Learning project that performs multiclass classification on the Iris dataset using Logistic Regression and Decision Tree classifiers.

The project includes Exploratory Data Analysis (EDA), data preprocessing, model training, performance evaluation, confusion matrices, and decision boundary visualization.

📌 Project Overview

The objective of this project is to build and benchmark two machine learning classification algorithms on the Iris dataset and compare their performance using different evaluation metrics.

Dataset: Iris Dataset

Problem Type: Multiclass Classification

Models Used:

Logistic Regression
Decision Tree Classifier
🎯 Project Objectives
Load and explore the Iris dataset.
Perform Exploratory Data Analysis (EDA).
Analyze feature relationships using pairplots.
Split data into training and testing sets.
Build classification models.
Train and evaluate both models.
Compare model performance.
Visualize confusion matrices.
Plot decision boundaries using two selected features.
Save evaluation results and visualizations.
📊 Dataset Information

The Iris dataset contains 150 samples with four numerical features and three target classes.

Features:

Sepal Length
Sepal Width
Petal Length
Petal Width

Target Classes:

Iris Setosa
Iris Versicolor
Iris Virginica
🛠️ Technologies & Libraries
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
⚙️ Machine Learning Workflow
Dataset Loading
Exploratory Data Analysis
Data Preprocessing
Train-Test Split (80% Training / 20% Testing)
Feature Scaling using StandardScaler
Model Building
Model Training
Model Evaluation
Confusion Matrix Visualization
Decision Boundary Visualization
Model Performance Comparison
🤖 Models Implemented
1. Logistic Regression
Uses StandardScaler for feature scaling.
Maximum iterations: 1000.
Performs multiclass classification.
2. Decision Tree Classifier
Maximum tree depth: 4.
Random state: 42.
Performs multiclass classification.
📈 Evaluation Metrics

Both models are evaluated using:

Accuracy
Weighted Precision
Weighted Recall
Weighted F1-Score
Classification Report
Confusion Matrix

Models are ranked according to their weighted F1-score.

📂 Project Structure
text
iris-classification-pipeline/
│
├── iris_classification_pipeline.py
│
├── outputs/
│   ├── 01_pairplot.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── decision_boundaries.png
│   └── classification_metrics.csv
│
└── README.md

🚀 Installation & Setup
1. Clone the Repository
bash
git clone https://github.com/YOUR-USERNAME/iris-classification-pipeline.git

2. Navigate to Project Directory
bash
cd iris-classification-pipeline

3. Install Required Libraries
bash
pip install numpy pandas matplotlib seaborn scikit-learn

4. Run the Project
bash
python iris_classification_pipeline.py

📁 Output Files

After successful execution, the project generates:

File	Description
01_pairplot.png	Feature relationship visualization
confusion_matrix_logistic_regression.png	Logistic Regression confusion matrix
confusion_matrix_decision_tree.png	Decision Tree confusion matrix
decision_boundaries.png	Decision boundary comparison
classification_metrics.csv	Model performance comparison
📊 Model Comparison

The project automatically generates a performance comparison table containing:

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	Generated at runtime	Generated at runtime	Generated at runtime	Generated at runtime
Decision Tree	Generated at runtime	Generated at runtime	Generated at runtime	Generated at runtime

Actual evaluation scores are saved in outputs/classification_metrics.csv.

🔮 Future Improvements
Implement Random Forest Classifier.
Add Support Vector Machine (SVM).
Perform cross-validation.
Apply hyperparameter tuning.
Compare additional classification algorithms.
Build an interactive prediction interface.
👨‍💻 Author

Your Name

Machine Learning | Python | Data Science

⭐ Acknowledgements
Scikit-learn for machine learning algorithms and evaluation tools.
Seaborn for the Iris dataset and data visualization.
Pandas and NumPy for data manipulation and numerical operations.

If you find this project useful, consider giving it a ⭐ on GitHub!
