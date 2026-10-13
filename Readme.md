# Healthcare Insurance Cost Analysis and Prediction

## Project Overview

This project applies data analytics, statistical hypothesis testing, machine learning, and business intelligence to investigate healthcare insurance charges and predict individual insurance costs.

Using Python, Jupyter Notebook, and Power BI, the project follows an end-to-end analytical workflow: data inspection and cleaning, exploratory data analysis (EDA), statistical testing, regression modelling, model evaluation, and interactive dashboard development.

The analysis investigates how demographic and lifestyle characteristics—including age, body mass index (BMI), smoking status, sex, number of children, and geographic region—are associated with healthcare insurance charges. It also compares several regression algorithms to identify a suitable model for predicting charges.

A three-page Power BI report communicates the results through interactive views of customer characteristics, charge distributions, and machine-learning performance. Users can explore the results using filters for smoking status, BMI category, age category, and region.

The results describe patterns and associations in the available dataset. They do not establish causation, and predictive performance on this dataset does not guarantee equivalent performance on other populations.

## Project Objectives

The main objectives are to:

- Inspect, clean, validate, and prepare the healthcare insurance dataset.
- Explore distributions, potential outliers, and relationships between insurance charges and customer characteristics.
- Create BMI categories and investigate combined patterns involving BMI and smoking status.
- Apply statistical hypothesis tests to investigate differences and associations in insurance charges.
- Develop and compare machine-learning regression models.
- Evaluate predictive performance using a held-out test set and cross-validation.
- Examine prediction errors and permutation feature importance.
- Build interactive Power BI dashboards to communicate the findings.
- Discuss limitations, potential business implications, and responsible interpretation of the results.

## Dataset

The project uses the healthcare insurance dataset provided for the assessment, stored as `insurance.csv`.

The original dataset contains **1,338 observations and 7 columns**.

| Variable | Description |
|---|---|
| `age` | Age of the insured individual |
| `sex` | Sex category |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependants |
| `smoker` | Smoking status (`yes` / `no`) |
| `region` | Geographic region |
| `charges` | Healthcare insurance charges |

One exact duplicate record was removed from the analytical dataset, leaving **1,337 observations**. An additional BMI-category feature was created for exploratory analysis and reporting. The original raw dataset was retained unchanged.

Potential outliers were investigated but were not automatically removed because unusually high insurance charges may represent genuine observations.

## Project Structure

The project is organised into three Jupyter notebooks, supporting data files, and project documentation.

```text
Healthcare-Insurance-Cost-Analysis/
│
├── data/
│   └── insurance.csv
│
├── output/
│   ├── cleaned_insurance.csv
│   └── insurance_predictions.csv
│
├── notebooks/
│   ├── 01_data_cleaning_and_eda.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_machine_learning.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

This is the intended project structure. Confirm the actual folder names and file locations in the repository before submission. In particular, the notebooks should use consistent paths for the cleaned dataset and prediction output.

## Analytical Workflow

### 1. Data Cleaning and Exploratory Data Analysis

**Notebook:** `01_data_cleaning_and_eda.ipynb`

The first notebook inspects and prepares the dataset before exploring its characteristics. The workflow includes:

- Inspecting the dataset dimensions, column names, data types, and descriptive statistics.
- Checking missing values and duplicate records.
- Removing one exact duplicate record from the analytical dataset.
- Investigating potential outliers using the Interquartile Range (IQR) method and box plots.
- Retaining potential outliers that may represent valid observations.
- Creating BMI categories while retaining the continuous BMI variable.
- Examining the distribution of insurance charges.
- Comparing charges by age, BMI, smoking status, sex, number of children, and region.
- Exploring the combined relationship between BMI, smoking status, and insurance charges.
- Using descriptive statistics, scatter plots, box plots, distribution plots, and correlation visualisations to identify patterns.

#### Main EDA findings

- Insurance charges have a strongly right-skewed distribution, with a smaller number of observations showing very high charges.
- Smokers generally have substantially higher charges than non-smokers.
- Charges tend to increase with age.
- The obese BMI category has the highest median charges among the BMI categories examined.
- Differences by sex and region appear smaller than those associated with smoking status.
- The number of children does not show a clear monotonic pattern with charges.
- Investigating BMI and smoking status together provides additional insight into differences in charge distributions.

These findings are descriptive and do not establish causal relationships.

### 2. Statistical Analysis and Hypothesis Testing

**Notebook:** `02_statistical_analysis.ipynb`

The second notebook investigates three research questions using hypothesis tests. The significance level is **0.05** for all three tests.

#### Research question 1: Do insurance-charge distributions differ between smokers and non-smokers?

A two-sided Mann–Whitney U test compares the insurance charges of the two independent groups.

- **Null hypothesis (H₀):** The charge distributions are the same for smokers and non-smokers.
- **Alternative hypothesis (H₁):** The charge distributions differ between smokers and non-smokers.

#### Research question 2: Is age associated with insurance charges?

Spearman's rank correlation examines the direction and strength of the monotonic association between age and charges.

- **Null hypothesis (H₀):** There is no monotonic association between age and charges.
- **Alternative hypothesis (H₁):** There is a monotonic association between age and charges.

#### Research question 3: Is BMI associated with insurance charges?

Spearman's rank correlation examines the monotonic association between BMI and charges.

- **Null hypothesis (H₀):** There is no monotonic association between BMI and charges.
- **Alternative hypothesis (H₁):** There is a monotonic association between BMI and charges.

#### Statistical results

| Research question | Statistical test | Statistic / coefficient | p-value |
|---|---|---:|---:|
| Smoking status and charges | Mann–Whitney U | 7,403 | 5.75 × 10⁻¹³⁰ |
| Age and charges | Spearman's rank correlation | 0.5335 | 3.19 × 10⁻⁹⁹ |
| BMI and charges | Spearman's rank correlation | 0.1196 | 1.16 × 10⁻⁵ |

All three tests are statistically significant at the 0.05 level.

- **Smoking status:** The test identifies a statistically significant difference between the charge distributions. The observed median charge was approximately 34,456 for smokers and 7,346 for non-smokers.
- **Age:** The results indicate a moderately strong positive monotonic association between age and charges.
- **BMI:** The results indicate a statistically significant but weak positive monotonic association between BMI and charges.

Statistical significance does not necessarily imply a strong relationship. The tests do not demonstrate that these characteristics cause changes in insurance charges.

### 3. Machine Learning and Prediction

**Notebook:** `03_machine_learning.ipynb`

The third notebook develops regression models to predict healthcare insurance charges. The target variable is `charges`, and the selected demographic and lifestyle variables are used as input features.

The workflow includes:

- Validating the cleaned dataset before modelling.
- Separating input features from the target variable.
- Splitting the data into 80% training and 20% testing subsets.
- Creating a mean-based `DummyRegressor` baseline.
- Building preprocessing and regression pipelines.
- Comparing Linear Regression, Ridge Regression, Random Forest, Extra Trees, and Gradient Boosting.
- Using five-fold cross-validation to compare model performance.
- Selecting Gradient Boosting as the leading candidate based on cross-validation.
- Evaluating the selected model on the held-out test set.
- Examining actual-versus-predicted values and residuals.
- Investigating permutation feature importance.
- Exporting predictions to `output/insurance_predictions.csv`.

#### Model evaluation

The models are evaluated using:

- **Mean Absolute Error (MAE):** the average absolute difference between actual and predicted charges.
- **Root Mean Squared Error (RMSE):** a measure that gives greater weight to larger prediction errors.
- **R²:** a measure of the proportion of target variation explained by the model relative to a mean-based reference.

#### Recorded model results

| Evaluation stage | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline — test set | 9,861.80 | 13,612.43 | -0.0084 |
| Linear Regression — test set | 4,338.92 | 6,024.00 | 0.8025 |
| Gradient Boosting — five-fold cross-validation mean | 2,679.82 | 4,592.16 | 0.8433 |
| Gradient Boosting — final test set | 2,603.04 | 4,370.82 | 0.8960 |

Gradient Boosting was the leading candidate among the evaluated models based on five-fold cross-validation. On the held-out test set, it achieved an MAE of **2,603.04**, an RMSE of **4,370.82**, and an R² of **0.8960**.

The actual-versus-predicted plot and residual analysis indicate that the model predicts many observations reasonably well, but some high-charge cases are substantially underestimated. This is an important limitation when interpreting its overall performance.

Permutation feature importance identified **smoking status** as the most influential feature, followed by **age** and **BMI**. Other features, including children, region, and sex, had comparatively lower predictive importance in this fitted model. Feature importance describes how the model uses the available information; it does not establish causation.

## Power BI Dashboard

The project includes an interactive Power BI report titled **Insurance Charge Predictions**, organised into three pages.

### Page 1 — Customer Characteristics

This page presents customer and charge indicators, including the smoker percentage, average BMI, average charges, and total customers. Visualisations compare average charges by smoking status and region, show charges against age, and examine the combined patterns of BMI category, smoking status, and sex.

### Page 2 — Charge Distribution

This page investigates the distribution of insurance charges using charge bins and comparisons by smoking status. Additional visuals summarise average charges by BMI category, age category, and smoking status.

### Page 3 — Machine-Learning Performance

This page presents model-performance indicators, including R², MAE, and RMSE, alongside visualisations of prediction errors by BMI category, sex, smoking status, and charge range. An actual-versus-predicted-style scatter plot supports the visual investigation of model predictions.

### Interactive features

The report includes slicers for:

- Smoking status.
- BMI category.
- Age category.
- Geographic region.

The report also includes a **Clear all slicers** control and a prediction-accuracy category filter on the machine-learning performance page.

The Power BI pages support interactive investigation of subsets of the data. Values shown in filtered views depend on the active slicers and should not be interpreted as full-dataset results unless all filters are cleared. The model-performance figures displayed in Power BI should also be checked against the final evaluation in the machine-learning notebook before final submission, so that the evaluation dataset and metrics are clearly identified.

## Tools and Technologies

- **Python** — data analysis and machine learning.
- **Jupyter Notebook** — interactive analysis and documentation.
- **pandas** and **NumPy** — data manipulation and numerical operations.
- **Matplotlib** and **Seaborn** — statistical visualisation.
- **SciPy** — statistical hypothesis testing.
- **scikit-learn** — preprocessing, regression models, train–test splitting, cross-validation, evaluation, and permutation feature importance.
- **Power BI Desktop** — interactive reporting and dashboard development.
- **Git and GitHub** — version control and project management.

## How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook. The notebooks use the data-analysis, visualisation, statistical, and machine-learning libraries listed above. Power BI Desktop is required to open and interact with the Power BI report.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ali-Behdad/code-institute-_-Assessment-2-Healthcare-Insurance-ML.git
   ```

2. Navigate to the project directory:

   ```bash
   cd code-institute-_-Assessment-2-Healthcare-Insurance-ML
   ```

3. Optionally create and activate a virtual environment.

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

### Execution order

Run the notebooks in the following order:

1. `01_data_cleaning_and_eda.ipynb` — clean and explore the data.
2. `02_statistical_analysis.ipynb` — conduct the statistical tests.
3. `03_machine_learning.ipynb` — train and evaluate the prediction models.

Ensure that the cleaned dataset is saved to the path expected by the later notebooks. Check that the input and output paths are consistent with the actual repository structure before running the complete workflow.

To explore the interactive report, open the Power BI report file in Power BI Desktop. The report file should be included in the repository or linked from the README if it is stored separately.

## Limitations and Ethical Considerations

- The dataset contains a limited number of demographic and lifestyle variables.
- Medical history, diagnoses, treatment requirements, and other potential determinants of healthcare costs are not represented in the available features.
- Potential outliers were retained and may influence summary statistics and prediction errors.
- Statistical associations do not establish causal relationships.
- Predictive performance on this dataset does not guarantee generalisability to other populations or future observations.
- Some high-charge observations are substantially underestimated by the selected model.
- Feature importance describes predictive contribution within the fitted model and does not demonstrate causation.
- Insurance predictions should not be used as the sole basis for decisions that could unfairly disadvantage individuals.

Further validation, broader data, and appropriate fairness assessments would be needed before using such a model in real-world insurance decisions.

## Conclusion

This project combines data cleaning, exploratory data analysis, statistical hypothesis testing, machine-learning regression, and interactive Power BI reporting to investigate healthcare insurance charges.

The exploratory analysis identified substantial differences in charges between smokers and non-smokers and positive associations between charges and age. Statistical testing supported a significant difference between smoker groups, a moderately strong positive monotonic association between age and charges, and a weak positive monotonic association between BMI and charges.

Gradient Boosting was selected as the leading candidate following five-fold cross-validation and achieved an R² of 0.8960 on the held-out test set. However, residual analysis showed that some high-charge observations were substantially underestimated. Smoking status, age, and BMI were the most influential features in the permutation-importance analysis.

The Power BI report provides an interactive way to explore customer characteristics, charge distributions, and model errors. Together, the notebooks and dashboard demonstrate an end-to-end data analytics workflow while highlighting the importance of data quality, appropriate statistical interpretation, model evaluation, and responsible use of predictions.

## AI Tools and Acknowledgements

### AI-assisted support

ChatGPT and Copilot were used as supporting tools during project development for activities such as clarifying assessment requirements, reviewing code, improving documentation, and checking grammar and spelling. The project author is responsible for reviewing the code, analytical decisions, results, and interpretations.

### Acknowledgements

- **Code Institute** — course materials, project template, and assessment guidance.
- **Kaggle** — source of the healthcare insurance dataset used for the analysis.
- **Data Analytics course** — learning support and feedback.

## Author

**Ali Behdad**

GitHub: [Ali-Behdad](https://github.com/Ali-Behdad)
