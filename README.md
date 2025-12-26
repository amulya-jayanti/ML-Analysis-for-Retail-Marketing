# Income Prediction and Customer Segmentation Project

### All Features provided in the dataset and their glossary:

1. Demographics:
- age: Person’s age in years.
- sex: Biological sex (Male / Female).
- race: Racial category as defined by Census (e.g., White, Black, Asian, etc.).
- hispanic origin: Whether the person identifies as Hispanic/Latino, independent of race.
- citizenship: Citizenship status (e.g., native-born, naturalized citizen, non-citizen).

2. Education & Enrollment:
- education: Highest level of education completed (e.g., high school, bachelor’s, master’s).
- enroll in edu inst last wk: Whether the person was enrolled in an educational institution during the previous week.

3. Employment & Work Status
- class of worker: Type of employer (private sector, government, self-employed, unpaid family worker, etc.).
- detailed industry recode: Fine-grained industry classification where the person works (numeric code).
- major industry code: Broader industry category (e.g., manufacturing, healthcare).
- detailed occupation recode: Fine-grained job role classification.
- major occupation code: Broader occupation category (e.g., professional, service, labor).
- full or part time employment stat: Whether the person works full-time or part-time.
- num persons worked for employer: Size of the employer (number of employees).
- own business or self employed: Whether the person owns a business or is self-employed.
- weeks worked in year: Number of weeks the person worked during the year.
- reason for unemployment: If unemployed, the reason (laid off, quit, new entrant, etc.).

4. Income & Financial Variables
- wage per hour: Hourly wage (if applicable).
- capital gains: Income from selling assets (stocks, property, etc.).
- capital losses: Losses from selling assets.
- dividends from stocks: Dividend income earned from investments.
- veterans benefits: Whether the person receives veterans’ benefits.

5. Tax & Filing Information
- tax filer stat: Tax filing status (single, joint, dependent, etc.).
- fill inc questionnaire for veteran's admin: Whether the income questionnaire was filled for Veterans Administration purposes.

6. Household & Family Structure
- detailed household and family stat: Detailed family role (e.g., spouse, child, householder).
- detailed household summary in household: Household composition summary (family vs non-family household).
- family members under 18: Number of family members under age 18 in the household.
- weight: Survey weight used to make the sample representative of the U.S. population; important for statistical analysis and usually not used as a predictive feature.

7. Migration & Residence
- region of previous residence: Region where the person lived previously.
- state of previous residence: State where the person lived previously.
- live in this house 1 year ago: Whether the person lived in the same house one year ago.
- migration code-change in msa: Whether the person moved between metropolitan statistical areas.
- migration code-change in reg: Whether the person moved between census regions.
- migration code-move within reg: Whether the person moved within the same region.
- migration prev res in sunbelt: Whether the previous residence was in the Sunbelt region.

8. Place of Birth & Background
- country of birth self: Country where the person was born.
- country of birth father: Father’s country of birth.
- country of birth mother: Mother’s country of birth.

9. Military / Labor Affiliations
- member of a labor union: Whether the person belongs to a labor union.
- veterans benefits: Indicates receipt of veterans’ benefits (binary or categorical).

10. Time
- year: Survey year, relevant for controlling temporal effects in analysis.

11. Target Variable
- label: Binary income indicator showing whether income is greater than $50K or less than or equal to $50K.

## Project Overview

This project delivers two machine learning solutions for retail marketing optimization:

1. **Binary Income Classifier**: Predicts whether individuals earn above or below $50,000 annually
2. **Customer Segmentation Model**: Groups customers into 5 actionable market segments

The analysis uses U.S. Census Bureau data from 1994-1995 containing 199,523 observations across 40 demographic and employment variables.

---

## 📁 Project Structure

```
.
├── README.md                              
├── Project_Report.docx                    # Comprehensive project report (10 pages)
│
├── datapreprocessing_EDA.ipynb           # Data cleaning, exploration, and feature engineering
├── predictive_modeling.ipynb             # Classification model training and evaluation 
├── predictive_modeling_refined.ipynb     # Classification model training and evaluation - refined (consider this as final modeling)
├── segmentation_modeling.ipynb           # Customer segmentation analysis

├── census-bureau.data                    # Raw data file
├── census-bureau.columns                 # Column definitions
└── final_data.csv                        # Preprocessed data (generated by datapreprocessing_EDA notebook)
```

---

## 🎯 Key Results

### Classification Performance
- **Best Model**: Gradient Boosting
- **ROC-AUC**: 0.948
- **PR-AUC**: 0.652
- **Precision**: 75.7% (minimizes wasted marketing spend)
- **Recall**: 41.2% (captures 2 in 5 high-income individuals)
- **F1-Score**: 0.534

### Customer Segments
1. **Established Professionals** (17%): High-income, dividend earners, age ~51
2. **Income-Constrained Workers** (93%): Largest group, moderate work intensity
3. **Dependents/Non-Earners** (51%): Children, students, non-working
4. **High-Wage Professionals** (10%): Full-time, highest hourly wages, age ~35
5. **Affluent Investors** (3%): Highest income rate (32.6%), significant capital gains

---

##  Setup Instructions

### Prerequisites

- **Python**: Version 3.8 or higher
- **Jupyter Notebook**: For running .ipynb files
- **RAM**: Minimum 4GB recommended
- **Disk Space**: ~500MB for data and dependencies

### Required Libraries

Install all dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

### Installation Steps

1. **Clone or download this project**
   ```bash
   cd /path/to/project
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy jupyter
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, sklearn, xgboost; print('All packages installed successfully!')"
   ```

---

## Execution Instructions

### Quick Start (Run All Notebooks)

Execute notebooks in the following order:

```bash
# 1. Data preprocessing and EDA
jupyter notebook datapreprocessing_EDA.ipynb

# 2. Classification modeling
jupyter notebook predictive_modeling_refined.ipynb

# 3. Customer segmentation
jupyter notebook segmentation_modeling.ipynb
```

### Detailed Execution Guide

#### Step 1: Data Preprocessing and Exploratory Data Analysis

**Notebook**: `datapreprocessing_EDA.ipynb`

**Purpose**: 
- Load and merge raw data files (census-bureau.columns, census-bureau.data)
- Handle missing values and data quality issues
- Perform exploratory data analysis
- Engineer features and create final preprocessed dataset

**Execution**:
```bash
jupyter notebook datapreprocessing_EDA.ipynb
```

**Key Outputs**:
- `final_data.csv`: Cleaned and preprocessed data (199,523 rows × 42 columns)
- EDA of every column and their interesting insights
- Visualization plots showing:
  - Target variable distribution
  - Age and education distributions
  - Correlation heatmaps
  - Employment patterns
  - Financial variable distributions

**Runtime**: ~3-5 minutes

**Important Notes**:
- Ensure `census-bureau.data` and `census-bureau.columns` are in the same directory
- The notebook uses survey weights for EDA and predictive modeling. 
- All visualizations are generated inline

#### Step 2: Predictive Modeling (Classification)

**Notebook**: `predictive_modeling_refined.ipynb`

**Purpose**:
- Train and evaluate multiple classification models
- Compare performance metrics
- Select optimal model and threshold
- Generate performance visualizations

**Prerequisites**:
- `final_data.csv` must exist (generated by Step 1)

**Execution**:
```bash
jupyter notebook predictive_modeling_refined.ipynb
```

**Key Outputs**:
- Trained models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- Performance comparison table (roc_auc, pr_auc, precision, recall, f1)
- Confusion matrix visualization
- Model selection recommendations

**Runtime**: ~10-15 minutes

**Models Trained**:
1. **Logistic Regression** (baseline)
   - Linear model with class_weight= balanced
   - Fast training, interpretable coefficients

2. **Random Forest**
   - 300 trees
   - Robust to outliers, handles non-linearity

3. **Gradient Boosting** (recommended)
   - 300 estimators, learning rate 0.05, max depth 3
   - Best precision-recall balance

4. **XGBoost**
   - 400 estimators, max depth 5, learning rate 0.05, logloss
   - Highest ROC-AUC, recall-focused

#### Step 3: Customer Segmentation

**Notebook**: `segmentation_modeling.ipynb`

**Purpose**:
- Create behavioral and demographic segments
- Compare clustering algorithms
- Profile and validate segments
- Generate marketing recommendations

**Prerequisites**:
- `final_data.csv` must exist (generated by Step 1)

**Execution**:
```bash
jupyter notebook segmentation_modeling.ipynb
```

**Key Outputs**:
- 5 customer segments with detailed profiles
- Segment population sizes and income distributions
- Behavioral characteristic heatmaps
- Marketing strategy recommendations
- Segment validation metrics

**Runtime**: ~5-8 minutes

**Clustering Algorithms Compared**:
- K-Means (selected)
- Gaussian Mixture Model
- Agglomerative Hierarchical (optional)

**Validation Metrics**:
- Silhouette Score: 0.275
- Davies-Bouldin Index: 1.06
- Calinski-Harabasz Score: 71,691

---

## Understanding the Outputs

### Classification Model Outputs

**Metrics Explanation**:
- **ROC-AUC** (0.948): Probability that model ranks a random high-income person above a random low-income person
- **Precision** (75.7%): Of predicted high-income individuals, 75.7% actually earn >$50K
- **Recall** (41.2%): Of actual high-income individuals, 41.2% are correctly identified
- **F1-Score** (0.534): Harmonic mean balancing precision and recall

**Confusion Matrix** (at 0.5 threshold):
```
                    Predicted
                 Low      High
Actual  Low     [TN]     [FP]
        High    [FN]     [TP]
```

**Business Implications**:
- High precision → Fewer wasted marketing resources
- Moderate recall → Misses some high-income individuals
- Adjustable threshold for different campaign types

### Segmentation Model Outputs

| Segment                                                                   | Weighted Population | High Income % | Key Characteristics                                                           | Marketing Focus                                                                    |
| ------------------------------------------------------------------------- | ------------------- | ------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **0 – Family-Centered Middle-Income Households**                          | **28M (~10%)**      | **23.8%**     | Age ~51, steady work history, married households, moderate dividend income    | Home improvement, insurance bundles, seasonal retail, retirement-adjacent services |
| **1 – Economically Active but Income-Constrained Workers**                | **185M (~64%)**     | **5.6%**      | Age ~44, low wages & investments, financially fragile, high price sensitivity | Value products, BNPL, loyalty programs, micro-financing, cost-saving bundles       |
| **2 – Non-Working / Economically Inactive Population (Dependents)**       | **102M (~35%)**     | **~0%**       | Avg age ~10, school-age dependents, minimal direct income                     | Indirect targeting via parents: apparel, education, entertainment, seasonal retail |
| **3 – Skilled Full-Time Earners (Labor-Driven Income)**                   | **19M (~7%)**       | **4.2%**     | Age ~36, highest wage intensity, longest work weeks, stable employment        | Automotive, tools, convenience services, buy-now/pay-later for durable goods       |
| **4 – Established High-Income Professionals (Investment-Wealth Earners)** | **12.9M (~4%)**     | **33.2%**     | Age ~48, high capital gains, stable employment, married, wealth accumulation  | Premium subscriptions, wealth management, luxury goods, travel packages            |


---

## Additional Resources

### Dataset Documentation
- **CPS Technical Documentation**: https://www2.census.gov/programs-surveys/cps/techdocs/cpsdec94.pdf
- **Variable Definitions**: See `census-bureau.columns` file
- **Survey Methodology**: U.S. Census Bureau Current Population Survey

### Machine Learning References
- **Scikit-learn Documentation**: https://scikit-learn.org/stable/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Imbalanced Learning**: https://imbalanced-learn.org/

### Business Analytics
- **Marketing Segmentation**: Best practices in retail analytics: https://www.investopedia.com/terms/m/marketsegmentation.asp

---

This project is delivered as a take-home assignment. All code, models, and documentation are provided for evaluation purposes.

**Attribution**:
- Dataset: U.S. Census Bureau (Public Domain)
- Libraries: Open-source Python ecosystem (BSD/MIT licenses)
- Analysis: Original work for this project

---

## Version History

**Version 1.0** (Current)
- Initial delivery with complete analysis
- 3 Jupyter notebooks covering EDA, classification, and segmentation
- Comprehensive 10-page project report
- Production-ready models with deployment guidelines

---

## Checklist for Project Evaluation

- [x] Classification model trained and validated
- [x] Segmentation model developed with 5 clusters
- [x] Code documented and executable
- [x] README with setup and execution instructions
- [x] Project report (≤10 pages) with:
  - [x] Data exploration and preprocessing approaches
  - [x] Model architecture and training algorithms
  - [x] Evaluation procedures and metrics
  - [x] Business insights and recommendations
  - [x] References to consulted resources
- [x] All deliverables tested and verified

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy jupyter

# 2. Run notebooks in order
jupyter notebook datapreprocessing_EDA.ipynb              # Generate final_data.csv
jupyter notebook predictive_modeling_refined.ipynb        # Train classification models
jupyter notebook segmentation_modeling.ipynb              # Create customer segments

# 3. Review outputs
# - Check final_data.csv was created
# - Review model performance metrics in notebook outputs
# - Examine segment profiles and visualizations

# 4. Read comprehensive report
# - Open Project_Report.pdf for full analysis and recommendations
```

---

**End of README**

*For detailed analysis, business recommendations, and implementation guidelines, please refer to Project_Report.pdf*
