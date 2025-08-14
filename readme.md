# Predicting Rainfall in Perth with a Neural Network

This data science project demonstrates a complete, end-to-end machine learning workflow for predicting whether it will rain the next day in Perth, Australia. The project uses a neural network (Multi-layer Perceptron) and focuses on best practices, including detailed data pre-processing, robust feature engineering, in-depth model evaluation, and hyperparameter tuning.

The entire analysis is contained within the `predicting_weather.ipynb` Jupyter Notebook.

---

## Table of Contents
* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Technologies Used](#technologies-used)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [Project Workflow](#project-workflow)
* [Results and Analysis](#results-and-analysis)
* [Conclusion](#conclusion)
* [Future Improvements](#future-improvements)
* [License](#license)

---

## Project Overview

The goal of this project is to build a binary classification model that accurately predicts `RainTomorrow`. The analysis begins with raw weather data from Perth, proceeds through a rigorous cleaning and feature engineering phase, and culminates in the training and comparative evaluation of two neural network models.

A key focus of this project is to move beyond simple accuracy and perform a nuanced evaluation that considers the real-world implications of different types of prediction errors, especially in the context of an imbalanced dataset.

---

## Key Features

- **Detailed Data Pre-processing:** Includes scoping the dataset to a single location for a specialized model and removing irrelevant or data-leaking features.
- **Robust Missing Data Handling:** Uses `SimpleImputer` to handle missing values, preserving the dataset's size and integrity while carefully avoiding data leakage.
- **Advanced Feature Engineering:**
  - Converts boolean features to a binary format.
  - Applies a **sin/cos transformation** to cyclical features (wind direction) to preserve their cyclical nature for the model.
- **In-depth Model Evaluation:** Moves beyond accuracy to use a **Confusion Matrix** and **Classification Report** (calculating Precision, Recall, and F1-Score), which are critical for imbalanced datasets.
- **Hyperparameter Tuning:** Employs `GridSearchCV` to systematically search for an optimal neural network architecture, comparing its performance to an initial baseline model.
- **Clear Documentation:** The Jupyter Notebook is structured with detailed markdown cells that explain the "why" behind each step of the process.

---

## Technologies Used

- **Python 3.8+**
- **Libraries:**
  - `pandas` for data manipulation and analysis.
  - `numpy` for numerical operations.
  - `scikit-learn` for machine learning tasks (data splitting, imputation, scaling, modeling, and evaluation).
  - `matplotlib` for data visualization.
  - `Jupyter Notebook` for interactive development and analysis.

---

## Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alanspace/https://github.com/alanspace/Predicting_Weather_with_Neural_Networks.git
    cd Predicting_Weather_with_Neural_Networks
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file is included for easy setup.
    ```bash
    pip install -r requirements.txt
    ```
    *(If a `requirements.txt` file is not available, install the libraries manually: `pip install numpy pandas scikit-learn matplotlib jupyterlab`)*

---

## Usage

1.  Activate the virtual environment.
2.  Launch the Jupyter Notebook server:
    ```bash
    jupyter lab
    ```
3.  Open the `predicting_weather.ipynb` notebook and run the cells sequentially to reproduce the analysis.

---

## Project Workflow

The notebook follows a structured, end-to-end machine learning methodology:

1.  **Data Loading and Scoping:** The dataset is loaded and filtered to focus solely on the 'Perth' location to build a specialized model.
2.  **Pre-processing:** Unnecessary columns (`Date`, `RISK_MM`, `Location`) are removed.
3.  **Feature Engineering:** Boolean features (`RainToday`) are converted to 0/1, and cyclical features (`WindGustDir`, `WindDir9am`, `WindDir3pm`) are transformed into sine and cosine components to preserve their spatial relationships.
4.  **Data Splitting:** The data is split into training and testing sets *before* any imputation or scaling to prevent data leakage.
5.  **Imputation:** `SimpleImputer` is used to fill missing values, fitting only on the training data and transforming both train and test sets.
6.  **Scaling:** Features are scaled using `StandardScaler` to ensure the neural network treats all features equally regardless of their original scale.
7.  **Initial Modeling:** A baseline neural network with a `(50,50)` hidden layer architecture is trained and evaluated in detail.
8.  **Model Optimization:** `GridSearchCV` is used to test several different network architectures (`(2,)`, `(10,)`, `(50,50)`) and identify the best-performing one based on cross-validation.
9.  **Comparative Analysis:** The performance of the best model found by the grid search is compared against the initial baseline model using a full suite of classification metrics.

---

## Results and Analysis

The analysis yielded two key models with distinct performance characteristics on the test set:

-   **Initial Model `(50, 50)`:** A more complex model that proved to be better at correctly identifying rain events when they occurred (**higher recall** of 0.715).
-   **Optimized Model `(2,)`:** A much simpler model found via `GridSearchCV` that was more precise. When it predicted rain, it was correct more often (**higher precision** of 0.773).

The overall accuracy of both models was nearly identical (~89%), proving that accuracy is a misleading metric for this imbalanced dataset. The true difference lay in the **precision-recall trade-off**.

---

## Conclusion

The "best" model depends entirely on the business context and the cost of different errors:

-   For a use case where **missing a rain event is costly** (e.g., a concert organizer), the initial `(50, 50)` model with its higher recall is preferable.
-   For a use case where **falsely predicting rain is costly** (e.g., an automated irrigation system that shouldn't waste water), the optimized `(2,)` model with its higher precision is the better choice.

This project successfully demonstrates that effective data science is not just about building a model, but about deeply understanding its performance and translating those statistical metrics into actionable, context-driven insights.

---

## Future Improvements

-   **Experiment with other imputation strategies** (e.g., `KNNImputer` or `median` strategy) to see if it impacts model performance.
-   **Compare with other model types**, such as Logistic Regression, Gradient Boosting Machines (like XGBoost or LightGBM), which often perform very well on tabular data.
-   **Address class imbalance directly** using techniques like SMOTE (Synthetic Minority Over-sampling Technique) or by adjusting `class_weight` in the model parameters.
-   **Advanced Feature Engineering** by extracting cyclical features from the `Date` column (e.g., month, day of the year).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.