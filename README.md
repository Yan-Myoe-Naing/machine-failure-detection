# Machine Failure Detection

A machine learning project that classifies normal machine operation and failures using operating data.

The project covers data exploration, preprocessing, model comparison and hyperparameter tuning.

Failures make up only **3.39%** of the dataset. The evaluation focuses on **failure recall and F1 score**, since high accuracy alone can hide missed failures.

## Repository Files

| File | Description |
| --- | --- |
| [Machine Failure Detection.ipynb](Machine%20Failure%20Detection.ipynb) | Notebook with analysis, training code and saved results. |
| [Machine Failure Detection.pptx](Machine%20Failure%20Detection.pptx) | Presentation summarising the workflow and findings. |
| [factory_data.csv](factory_data.csv) | Dataset used for training and evaluation. |

## Dataset

The dataset contains **20,000 rows and 9 columns**.

- **Target:** `Machine Status` — `0` for normal operation and `1` for failure.
- **Predictors:** quality, ambient temperature, process temperature, rotation speed, torque and tool wear.
- **Excluded identifiers:** `Unique ID` and `Product ID`.

## Project Workflow

1. Explore class distribution, missing values, feature distributions and correlations.
2. Handle missing values using mode, forward-fill and mean imputation.
3. Encode quality as `L = 0`, `M = 1` and `H = 2`.
4. Create a stratified **80/20 train/test split** using `random_state=42`.
5. Explore scaling and PCA for baseline models.
6. Compare Dummy Classifier, Logistic Regression, Decision Tree, Gaussian Naive Bayes, Random Forest and Gradient Boosting.
7. Tune Gradient Boosting without quality using **Optuna**, with **50 trials** and **5-fold stratified cross-validation**.

Ensemble models use the original predictors or a subset without quality, rather than PCA inputs.

## Selected Model Results

The selected model is a **tuned Gradient Boosting Classifier without the quality feature**.

Results below are taken from the notebook's saved test outputs.

| Metric | Test Result |
| --- | ---: |
| Accuracy | 99.60% |
| Precision | 96.88% |
| Failure Recall | 91.18% |
| F1 Score | 0.9394 |

### Confusion Matrix

| Actual / Predicted | Normal | Failure |
| --- | ---: | ---: |
| Normal | 3,860 | 4 |
| Failure | 12 | 124 |

The model detects **124 of 136 failures**, misses **12 failures** and produces **4 false alarms**.

The majority-class baseline achieves **96.60% test accuracy** but detects no failures. This shows why recall and F1 matter for this task.

The best Optuna cross-validation F1 is **0.88435**. This is separate from the test F1 reported above.

## Tools Used

- Python
- pandas and NumPy
- scikit-learn
- Optuna
- Matplotlib and Seaborn
- Jupyter Notebook

## How to Run

1. Download or clone the repository.
2. Install the required packages:

   ```bash
   python -m pip install pandas numpy scikit-learn optuna matplotlib seaborn notebook
   ```

3. Keep `factory_data.csv` in the same folder as the notebook.
4. Replace the absolute Windows path in the data-loading cell with:

   ```python
   df = pd.read_csv("factory_data.csv")
   ```

5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `Machine Failure Detection.ipynb` and run the cells in order.

The Optuna search may take some time. New runs may produce different tuning results.

## Limitations and Future Improvements

- Fit imputation within the training set and each cross-validation fold. The current notebook fills missing values before splitting.
- Check whether row order supports forward-fill for process temperature.
- Use a fresh holdout set for final evaluation, since several model variants were compared on the same test set.
- Check performance across multiple training seeds.

## Author

**Yan Myoe Naing**  
Applied AI and Analytics, Singapore Polytechnic
