# Student Grade Prediction Using Machine Learning

A Python machine-learning project that predicts a student's grade category from study behavior and classroom information.

The application uses a tuned **Random Forest** model and provides a simple **Streamlit** user interface for demonstrations.

---

## Project Overview

The model predicts one of five grade categories:

- A
- B
- C
- D
- F

It uses three input features:

- Weekly self-study hours
- Attendance percentage
- Class participation

This is a **supervised multiclass classification** problem.

- **Supervised:** the correct grades are available during training.
- **Classification:** the output is a category.
- **Multiclass:** there are five possible grade classes.

---

## Dataset

The dataset contains:

- 1,000,000 student records
- 6 original columns
- No missing values
- No duplicate rows
- 800,000 training records
- 200,000 testing records
- Stratified 80/20 train-test split
- `random_state = 42`

### Input Features

- `weekly_self_study_hours`
- `attendance_percentage`
- `class_participation`

### Target

- `grade`

### Removed Columns

- `student_id` was removed because it is only an identifier.
- `total_score` was removed to prevent target leakage because the final grade is derived from the total score.

---

## Models Compared

Four classification models were trained and tuned:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost

### Final Model

The selected final model is the **Tuned Random Forest**.

It was selected because it achieved the highest tuned Macro F1-score.

| Metric | Result |
|---|---:|
| Accuracy | 0.6325 |
| Macro Precision | 0.4411 |
| Macro Recall | 0.5535 |
| Macro F1 | 0.4552 |
| Grade-F Recall | 0.66 |
| Training Time | Approximately 7.4 seconds |

Logistic Regression remained a close and faster alternative.

---

## Application Workflow

```text
Student information
        ↓
Streamlit user interface
        ↓
Saved tuned Random Forest
        ↓
Predicted grade and grade probabilities
```

The user enters:

- Weekly self-study hours
- Attendance percentage
- Class participation

The application displays:

- Predicted grade
- Highest model probability
- Probability for grades A, B, C, D, and F
- Final model information

---

## Project Structure

```text
student-grade-prediction/
├── artifacts/
│   └── final_random_forest.joblib
├── data/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── metrics/
├── scripts/
│   └── train_final_model.py
├── src/
│   ├── final_model.py
│   ├── prediction_service.py
│   └── models/
├── tests/
├── ui/
│   └── streamlit_app.py
├── main.py
├── README.md
├── requirements.txt
└── requirements-colab.txt
```

The trained `.joblib` model is generated locally and should not be committed to Git.

---

# Requirements

Install these before running the project:

- Git
- Python 3.11 or newer
- Internet connection for cloning and installing packages

Check that Git and Python are installed:

```text
git --version
python --version
```

On some macOS systems, use:

```text
python3 --version
```

---

# Run on macOS

## 1. Clone the Repository

Open Terminal:

```bash
git clone https://github.com/Lyxing9999/student-grade-prediction.git
cd student-grade-prediction
```

## 2. Create a Virtual Environment

### Fish shell

```fish
python3 -m venv .venv
source .venv/bin/activate.fish
```

### Bash or Zsh

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal should show something similar to:

```text
(.venv)
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Train and Save the Final Model

```bash
python -m scripts.train_final_model
```

This command:

1. Loads the dataset.
2. Creates the same 80/20 train-test split.
3. Trains the tuned Random Forest.
4. Evaluates the model.
5. Saves the trained model.

The generated model is saved at:

```text
artifacts/final_random_forest.joblib
```

Expected results are approximately:

```text
Accuracy:        0.6325
Macro Precision: 0.4411
Macro Recall:    0.5535
Macro F1:        0.4552
```

## 5. Run the Streamlit Interface

```bash
python -m streamlit run ui/streamlit_app.py
```

Open this address in your browser:

```text
http://localhost:8501
```

Streamlit may open the browser automatically.

## 6. Stop the Application

Return to the terminal and press:

```text
Control + C
```

## 7. Deactivate the Virtual Environment

```bash
deactivate
```

---

# Run on Windows

The steps below use **PowerShell**.

## 1. Clone the Repository

Open PowerShell:

```powershell
git clone https://github.com/Lyxing9999/student-grade-prediction.git
cd student-grade-prediction
```

## 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

## 3. Activate the Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

After activation, PowerShell should show something similar to:

```text
(.venv)
```

### PowerShell Execution-Policy Error

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

This policy change applies only to the current PowerShell window.

### Windows Command Prompt Alternative

When using Command Prompt instead of PowerShell:

```bat
.venv\Scripts\activate.bat
```

## 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Train and Save the Final Model

```powershell
python -m scripts.train_final_model
```

The generated model is saved at:

```text
artifacts\final_random_forest.joblib
```

Expected results are approximately:

```text
Accuracy:        0.6325
Macro Precision: 0.4411
Macro Recall:    0.5535
Macro F1:        0.4552
```

## 6. Run the Streamlit Interface

```powershell
python -m streamlit run ui/streamlit_app.py
```

Open this address in your browser:

```text
http://localhost:8501
```

## 7. Stop the Application

Return to PowerShell and press:

```text
Ctrl + C
```

## 8. Deactivate the Virtual Environment

```powershell
deactivate
```

---

# Quick Start

## macOS with Fish

```fish
git clone https://github.com/Lyxing9999/student-grade-prediction.git
cd student-grade-prediction
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.train_final_model
python -m streamlit run ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Windows PowerShell

```powershell
git clone https://github.com/Lyxing9999/student-grade-prediction.git
cd student-grade-prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.train_final_model
python -m streamlit run ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

# Using the Interface

1. Enter weekly self-study hours.
2. Enter attendance percentage.
3. Enter class participation.
4. Select **Predict student grade**.
5. Review the predicted grade and probability chart.

Example input:

```text
Weekly self-study hours: 18
Attendance percentage: 90
Class participation: 7
```

The exact prediction depends on the trained model.

---

# Google Colab Notebook

The Google Colab notebook includes:

- Dataset inspection
- Exploratory data analysis
- Grade-distribution analysis
- Data preprocessing
- Four classification models
- Baseline model comparison
- Hyperparameter tuning
- Tuned model comparison
- Confusion matrix
- Final model selection
- Conclusion

---

# Main Findings

- The grade distribution is highly imbalanced.
- Grade A represents approximately 54.86% of the records.
- Grade F represents approximately 0.62% of the records.
- Weekly self-study hours is the strongest predictive feature.
- Attendance and class participation provide limited predictive information.
- Tuned Random Forest achieved the highest Macro F1-score.
- Most prediction mistakes occurred between neighboring grades.
- Random Forest detected approximately 66% of actual grade-F students.

---

# Troubleshooting

## `python` Is Not Found

### macOS

Try:

```bash
python3 --version
python3 -m venv .venv
```

### Windows

Install Python and enable the option:

```text
Add Python to PATH
```

Then reopen PowerShell.

## Streamlit Is Not Found

Run Streamlit through Python:

```text
python -m streamlit run ui/streamlit_app.py
```

Make sure the virtual environment is active and dependencies are installed:

```text
python -m pip install -r requirements.txt
```

## Final Model Is Missing

Run:

```text
python -m scripts.train_final_model
```

Confirm that this file exists:

```text
artifacts/final_random_forest.joblib
```

## Port 8501 Is Already in Use

Run Streamlit on another port:

```text
python -m streamlit run ui/streamlit_app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## Dataset Cannot Be Found

Confirm that the dataset exists in the expected project data directory and that the repository was cloned completely.

## Stop the Server

Press:

```text
Ctrl + C
```

in the terminal where Streamlit is running.

---

# Limitations

- The dataset is synthetic.
- The grade distribution is highly imbalanced.
- Grade-F precision remains low.
- Weekly study hours contains most of the predictive information.
- The prediction is an educational demonstration.
- The result should not be treated as an official academic assessment.

---

# Team Members

- Mr. Kaing Bunly
- Ms. Teng Nyka
- Mr. Yim EangVeng
- Mr. Choem Vanna

---

# Short Project Explanation

This project compares four machine-learning classification models for student-grade prediction. After baseline evaluation and hyperparameter tuning, Random Forest achieved the highest Macro F1-score and was selected as the final model. A Streamlit interface allows users to enter student information and view the predicted grade and grade probabilities.

