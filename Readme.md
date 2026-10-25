# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)
# Healthcare Insurance Cost Analysis — Assessment 3

## Project Overview

This project develops an interactive data application for exploring healthcare insurance charges and communicating data-driven insights. It builds on work completed for Assessment 2, which used the `insurance.csv` dataset for data cleaning, exploratory data analysis (EDA), statistical hypothesis testing, machine-learning regression, and Power BI reporting.

**Assessment 3 reuses the previous project as a starting point.** The earlier analysis and its findings provide the foundation for this stage; they are not presented as if all the work were newly created. The main development in this stage is a new interactive dashboard built with **Streamlit**. Plotting issues identified in the earlier project were also reviewed and resolved, and code and visualisations were adjusted to improve how results are displayed.

The application explores how age, body mass index (BMI), smoking status, sex, number of children, and region are associated with healthcare insurance charges. These patterns are descriptive and do not establish causation. Model performance on this dataset does not guarantee equivalent performance for other populations.

## Project Rationale and Objectives

The project aims to make the analysis accessible to users who want to explore insurance-charge patterns without running the analysis notebooks themselves. Objectives are to:

- Inspect, clean, validate, and prepare the dataset.
- Explore charge distributions, potential outliers, and relationships with available characteristics.
- Investigate differences and associations using statistical tests.
- Develop and evaluate regression models.
- Present useful results through a Streamlit dashboard.
- Resolve plotting issues and improve the clarity of visual communication.
- Communicate limitations and recommendations responsibly.
- Document development decisions, ethical considerations, AI assistance, and future maintenance.

## Target Audience and Business Questions

The intended audience includes people who want to explore healthcare insurance charge patterns, including non-technical users. The dashboard should explain what each visualisation shows in clear language.

The project investigates questions such as:

- How do charges vary with age and BMI?
- How do observed charges differ between smokers and non-smokers?
- How do charges vary across the available demographic and regional categories?
- How well does the evaluated regression model predict charges in the test data?
- Which features contribute most to the fitted model's predictions?
- What limitations should users consider when interpreting the results?

## Dataset

The project uses the healthcare insurance dataset supplied for the earlier assessment, stored as `insurance.csv`. The original dataset contains **1,338 observations and 7 columns**.

| Variable | Description |
|---|---|
| `age` | Age of the insured individual |
| `sex` | Sex category |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependants |
| `smoker` | Smoking status (`yes` / `no`) |
| `region` | Geographic region |
| `charges` | Healthcare insurance charges |

One exact duplicate record was removed from the analytical dataset, leaving **1,337 observations**. A BMI-category feature was created for exploratory analysis. The raw dataset was retained unchanged. Potential outliers were investigated but not automatically removed because unusually high charges may be genuine observations.

## Reuse of Assessment 2 Work and Assessment 3 Changes

Assessment 3 builds on the earlier Assessment 2 project. The existing dataset, analytical workflow, statistical results, machine-learning work, and lessons learned provide a starting point. This reuse is acknowledged explicitly to make the development history transparent.

The main activities in this stage are:

- Developing a new interactive dashboard with **Streamlit**.
- Reviewing and resolving plotting problems encountered in the earlier project.
- Adjusting code and visualisations to improve presentation and interpretation.
- Improving the communication of findings for the intended audience.
- Giving greater attention to ethical, legal, and social implications.

The interactive Power BI report belonged to Assessment 2. The new Streamlit dashboard is the focus of the Assessment 3 application. 

## Previous Analytical Workflow and Findings

### Exploratory Data Analysis

The earlier workflow inspected data types, descriptive statistics, missing values, and duplicate records. It investigated potential outliers using the Interquartile Range (IQR) method and box plots, created BMI categories, and explored charges by age, BMI, smoking status, sex, number of children, and region.

The recorded exploratory findings were:

- Insurance charges have a strongly right-skewed distribution, with a smaller number of very high charges.
- Smokers generally have substantially higher charges than non-smokers.
- Charges tend to increase with age.
- The obese BMI category had the highest median charges among the BMI categories examined.
- Differences by sex and region appeared smaller than those associated with smoking status.
- The number of children did not show a clear monotonic pattern with charges.
- Examining BMI and smoking status together provided additional insight into charge distributions.

These are descriptive observations and do not demonstrate cause and effect.

### Statistical Analysis

The earlier project used a significance level of **0.05** for three tests.

| Research question | Test | Recorded result | Interpretation |
|---|---|---|---|
| Do charge distributions differ between smokers and non-smokers? | Two-sided Mann–Whitney U | U = 7,403; p = 5.75 × 10⁻¹³⁰ | Statistically significant difference |
| Is age associated with charges? | Spearman correlation | ρ = 0.5335; p = 3.19 × 10⁻⁹⁹ | Moderately strong positive monotonic association |
| Is BMI associated with charges? | Spearman correlation | ρ = 0.1196; p = 1.16 × 10⁻⁵ | Weak positive monotonic association |

All three tests were statistically significant at the 0.05 level. Statistical significance does not necessarily mean that a relationship is strong, and these tests do not establish causation.

### Machine Learning

The earlier project compared regression models for predicting `charges`, using an 80% training and 20% test split, a mean-based `DummyRegressor` baseline, preprocessing pipelines, and five-fold cross-validation. The evaluated algorithms included Linear Regression, Ridge Regression, Random Forest, Extra Trees, and Gradient Boosting.

| Evaluation stage | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline — test set | 9,861.80 | 13,612.43 | -0.0084 |
| Linear Regression — test set | 4,338.92 | 6,024.00 | 0.8025 |
| Gradient Boosting — cross-validation mean | 2,679.82 | 4,592.16 | 0.8433 |
| Gradient Boosting — final test set | 2,603.04 | 4,370.82 | 0.8960 |

Gradient Boosting was the leading candidate based on the recorded cross-validation results. Its held-out test-set R² was 0.8960. However, some high-charge cases were substantially underestimated. Permutation feature importance identified smoking status as the most influential feature, followed by age and BMI. Feature importance describes the fitted model's predictive behaviour; it does not establish causation.

These results are carried forward from Assessment 2. Before submission, verify that the current notebooks and Streamlit dashboard show consistent values and clearly distinguish cross-validation metrics from held-out test-set metrics.


## Streamlit Dashboard

The dashboard is intended to help users explore healthcare insurance charges, understand patterns across customer groups, and evaluate machine-learning predictions and their limitations.

Developed with **Streamlit** for Assessment 3, it builds on the analysis completed in Assessment 2 and contains three pages:

- **Customer Characteristics:** Compares charges by smoking status, region, age, BMI category, and sex.
- **Charge Distribution:** Uses histograms, box plots, violin plots, and scatter plots to explore charge distributions and relationships between variables.
- **Machine-Learning Performance:** Presents model metrics (R², MAE, and RMSE), prediction-error charts, and predicted-versus-actual charges.

Shared filters allow users to explore smoking status, BMI category, age category, and region. A *Clear all slicers* control resets the filters, while the machine-learning page includes a prediction-accuracy filter.

## Ethics, Privacy, Governance, and Responsible Use

Ethical practice is a central consideration in this project, particularly because it concerns healthcare-related data and insurance charges. The dataset includes demographic and lifestyle variables that may raise concerns about privacy, stereotyping, fairness, and the consequences of decisions made using predictions.

### Ethical issues and data governance

The analysis is limited to the variables in the supplied dataset. Data preparation should be documented, the original raw data should be retained unchanged, and unnecessary personal information should not be collected or exposed. The dashboard and README should explain the dataset's scope and limitations.

Observed associations between smoking status, age, BMI, and charges must not be presented as proof that these characteristics cause a particular charge. The dataset may not represent every population, insurance arrangement, or current circumstance.

### Legal and social implications

Insurance-related predictions can have financial and social consequences. Using demographic or lifestyle characteristics to make decisions about individuals could create or reinforce unfair outcomes. This project is an educational and exploratory application, not a validated system for determining premiums, eligibility, access to healthcare, or an individual's entitlement to support.

The project does not claim that the model has been approved for operational use or has passed a comprehensive fairness or legal-compliance assessment. Before real-world deployment, further work would be needed, including appropriate legal and data-protection review, checks of data representativeness, fairness testing across relevant groups, external validation, and human oversight.

### Responsible interpretation

The dataset contains a limited set of features and may omit important determinants of healthcare costs. Potential outliers were retained because they may be genuine observations. Model errors—particularly the underestimation of some high-charge cases—should be disclosed. Statistical significance and feature importance must not be confused with causation.

The dashboard should use clear language, avoid overstating predictive accuracy, and make limitations visible to technical and non-technical users. Predictions should not be the sole basis for decisions that could unfairly disadvantage individuals.

## AI Use and Acknowledgement

AI tools were used as supporting tools during development. Their use was concentrated mainly on **Streamlit development, code assistance, debugging, and code adjustments**, including help reviewing plotting problems and improving how the dashboard presents information. AI was also used to help clarify requirements and improve documentation and wording.

AI suggestions were not treated as automatically correct. The code, outputs, plots, and explanations were reviewed and tested against the actual data and intended behaviour. 


## Planning, Version Control, and Evaluation

The project plan and development history should show how tasks, progress, issues, and improvements were managed during development. Git and GitHub provide version control for tracking changes to the code and documentation. Clear commit messages—such as `Fix plot labels` or `Add Streamlit dashboard`—make individual changes easier to understand.

The application should be evaluated against its purpose and requirements. Relevant checks include dashboard functionality, plot clarity, consistency of displayed results, accessibility, usability, and whether the project hypotheses have been addressed. Maintenance considerations include dependency updates, error correction, and revalidation when data or code changes.

**Submission check:** Add the project-board link and report only testing and evaluation activities that were actually completed. The README alone does not provide evidence that these checks have been performed.

## Tools and Technologies

The earlier analysis used:

- **Python** — data analysis and machine learning.
- **Jupyter Notebook** — interactive analysis and documentation.
- **pandas** and **NumPy** — data manipulation and numerical operations.
- **Matplotlib** and **Seaborn** — statistical visualisation.
- **SciPy** — statistical hypothesis testing.
- **scikit-learn** — preprocessing, regression models, cross-validation, evaluation, and permutation feature importance.
- **Power BI Desktop** — reporting developed for Assessment 2.
- **Streamlit** — interactive dashboard developed for Assessment 3.
- **Git and GitHub** — version control and project management.

## Limitations and Future Improvements

The analysis is limited by the available dataset and selected features. It does not establish causal relationships, and performance on one test set does not guarantee performance on other populations or future data. Some high-charge observations were substantially underestimated.

Future improvements could include user testing, improved accessibility and responsive layout, clearer explanations for charts and metrics, fairness assessment across relevant groups, validation on appropriate external data, automated tests, and documented maintenance procedures. Plots and metrics should be rechecked whenever the data, code, or model changes.

## Conclusion

Assessment 3 builds on the healthcare insurance analysis developed for Assessment 2 and extends it into a new Streamlit-based data application. The earlier analysis provides the analytical foundation; this stage focuses on dashboard development, resolving plotting issues, improving communication, and documenting the development process.

The earlier analysis found substantially higher observed charges among smokers, a moderately strong positive association between age and charges, and a weak positive association between BMI and charges. Gradient Boosting achieved an R² of 0.8960 on the recorded held-out test set, although some high-charge cases were underestimated. These results should be communicated with appropriate qualifications.

A useful data application requires more than charts or a predictive model. It also requires a clear purpose, an appropriate audience, reliable implementation, iterative development, transparent acknowledgement of reused work and AI assistance, and careful consideration of ethical, legal, and social implications.

## References and Acknowledgements

- **Code Institute** — course materials, project template, and assessment guidance.
- **Kaggle** — source of the healthcare insurance dataset used in the analysis.
- **Data Analytics course** — learning support and feedback.
- **ChatGPT and Copilot** — supporting tools used for Streamlit development, code assistance, debugging, code adjustments, and documentation.

Acknowledge any external code, tutorials, libraries, or other material used in the final application in accordance with the course's academic-integrity guidance.

## Author

**Ali Behdad**

GitHub: [Ali-Behdad](https://github.com/Ali-Behdad)
