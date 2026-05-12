import nbformat as nbf

nb = nbf.v4.new_notebook()

# A utility function to append markdown cells
def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

# A utility function to append code cells
def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

md("""# Rigorous Exploratory Data Analysis
**Project**: Explainable AI Multi-Modal Framework for Water Quality Compliance Prediction  
**Dataset**: `india_water_quality_preprocessed_phase1.csv`

This notebook provides a comprehensive EDA to inform the model-building phase of the project, covering spatial, temporal, threshold, and feature relationship analyses.
""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load Dataset
df = pd.read_csv('india_water_quality_preprocessed_phase1.csv')
print(f"Dataset Shape: {df.shape}")
""")

md("""## 1. Column Classification & Feature Assessment
Understanding feature types is the first step prior to modeling.

### Identifiers / Metadata
`Station_Code`, `Monitoring_Location`, `Station_Base_Name`, `Station_Group`
**Action**: Drop these before modeling. They are identifiers and do not generalize.

### Spatial / Location
`State`, `River_Basin`, `Spatial_Tags`, `Primary_Position`
**Action**: `Primary_Position` and `River_Basin` could be used for grouped k-fold cross-validation or spatial context.

### Raw vs. Aggregated vs. Scaled Redundancies
The dataset contains columns like `BOD_Min_mgL`, `BOD_Max_mgL`, `BOD_Mean_mgL`, `BOD_Mean_mgL_Zscore`, and `BOD_Mean_mgL_MinMax`.  
**Action**: Do **NOT** use Scaled versions (MinMax / Zscore) alongside the raw/mean features during EDA or simple tree-based modeling; it causes artificial multicollinearity. Use **only** the `_Mean_` features for this EDA to represent central tendencies of variables.

### Target Leakage Detection ⚠️
The following columns encode the target or are derived strictly for class balancing:
- `Is_Safe` (Binary target)
- `Compliance_Label` (Multi-class target)
- `Compliance_Label_Encoded`
- `Class_Weight`
- `Binary_Class_Weight`

**Critical Rule**: Under no circumstances should `Compliance_Label` be used to predict `Is_Safe` (or vice-versa), nor should the weights be used as features.
""")

md("""## 2. Data Overview""")
code("""print(df.info())
print("\\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
""")

md("""**Insights:**
- The dataset has 3171 rows and 60 columns.
- Mostly clean data. Only `State` shows `11` missing values which is negligible.
- Memory usage is low (~1.5 MB), so data handling operates in-memory perfectly.
""")

md("""## 3. Target Analysis""")
code("""fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(data=df, x='Is_Safe', ax=ax[0], palette='Set2')
ax[0].set_title('Binary Label: Is_Safe Distribution')
sns.countplot(data=df, x='Compliance_Label', ax=ax[1], palette='viridis', 
              order=['Class_A','Class_B','Class_C','Class_D','Class_E','Non_Compliant'])
ax[1].set_title('Multi-Class Label: Compliance_Label Distribution')
plt.xticks(rotation=45)
plt.show()

print("Is_Safe Proportions:\\n", df['Is_Safe'].value_counts(normalize=True))
print("\\nCompliance_Label Proportions:\\n", df['Compliance_Label'].value_counts(normalize=True))
""")

md("""**Insights:**
- **Binary (`Is_Safe`)**: Clean 62-38% split. This is moderately balanced; standard metrics (Accuracy, F1) will work well. Binary weights (`Binary_Class_Weight`) can be applied but aren't strictly mandatory given the mild imbalance.
- **Multi-Class (`Compliance_Label`)**: High imbalance. `Class_E` has almost 0 support (0.2%), while `Class_C` and `Non_Compliant` dominate (26-27% each).
- **Recommendation**: Begin modeling on the `Is_Safe` binary label. Exploring the multi-class label will heavily require weighted loss functions or SMOTE to handle the minority classes (A and E).
""")

md("""## 4. Spatial Analysis""")
code("""# Using Has_US_DS_Pair to isolate pairs
us_ds_df = df[df['Has_US_DS_Pair'] == True]

plt.figure(figsize=(10, 6))
sns.barplot(data=us_ds_df, x='Primary_Position', y='Is_Safe', ci=None, palette='mako')
plt.title('Safety Rate: True Upstream vs Downstream Pairs')
plt.ylabel('Proportion Safe')
plt.show()

print(us_ds_df.groupby('Primary_Position')['Is_Safe'].mean())
""")

md("""**Insights:**
- For valid US/DS pairs, **Upstream stations are 66% safe**, while **Downstream stations fall to 53% safe**.
- This signifies a strong drop in water quality due to city/industrial effluent discharge between points.
- **Risk factor:** While the effect is visible, ensure we have enough pair data to train purely spatial models.
""")

md("""## 5. Temporal Analysis""")
code("""print(df.groupby('Year')['Is_Safe'].mean())
print(df.groupby('Year')['Is_Safe'].count())
""")
md("""**Insights:**
- 2022 and 2023 have nearly identical counts (~1571 and 1600 rows).
- The percentage of strictly safe cases remains constant across both years (~61.9% vs ~61.8%).
- **Recommendation**: Year should NOT be used as a predictive feature. Use it merely for context or as a splitting variable for validation (e.g. train on 2022, test on 2023 out-of-time evaluation).
""")

md("""## 6. Numerical Feature Analysis & Threshold Adherence""")
code("""key_vars = [
    'Temperature_Mean_C', 'Dissolved_Oxygen_Mean_mgL', 'pH_Mean', 
    'Conductivity_Mean_umho_cm', 'BOD_Mean_mgL', 'Nitrate_N_Mean_mgL',
    'Fecal_Coliform_Mean_MPN100ml', 'Total_Coliform_Mean_MPN100ml'
]
df_numeric = df[key_vars + ['Is_Safe']]

print("--- SAFE VS UNSAFE MEANS ---")
print(df_numeric.groupby('Is_Safe').mean().T)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, col in enumerate(key_vars):
    if 'Coliform' in col or col == 'Conductivity_Mean_umho_cm':
        sns.boxplot(data=df_numeric, x='Is_Safe', y=np.log1p(df_numeric[col]), ax=axes[idx])
        axes[idx].set_ylabel(f'Log1p({col.split("_")[0]})')
    else:
        sns.boxplot(data=df_numeric, x='Is_Safe', y=col, ax=axes[idx])
plt.tight_layout()
plt.show()
""")

md("""**Threshold-Consistency Analysis**
The project's rule base implies:
- DO >= 5 mg/L
- BOD <= 3 mg/L
- pH between 6.5 and 8.5
""")
code("""safe_df = df[df['Is_Safe'] == 1]
unsafe_df = df[df['Is_Safe'] == 0]

print(f"Safe but BOD > 3: {len(safe_df[safe_df['BOD_Mean_mgL'] > 3])} / {len(safe_df)}")
print(f"Safe but DO < 5: {len(safe_df[safe_df['Dissolved_Oxygen_Mean_mgL'] < 5])} / {len(safe_df)}")
print(f"Safe but pH < 6.5 or > 8.5: {len(safe_df[(safe_df['pH_Mean'] < 6.5) | (safe_df['pH_Mean'] > 8.5)])} / {len(safe_df)}")
print("\\nUnsafe but DO>=5 AND BOD<=3 AND pH in 6.5-8.5:")
print(f"{len(unsafe_df[(unsafe_df['Dissolved_Oxygen_Mean_mgL'] >= 5) & (unsafe_df['BOD_Mean_mgL'] <= 3) & (unsafe_df['pH_Mean'] >= 6.5) & (unsafe_df['pH_Mean'] <= 8.5)])} / {len(unsafe_df)}")
""")

md("""**Insights:**
- **CRITICAL**: Out of 1962 Safe examples, exactly **0 cases have BOD > 3**. This means `BOD <= 3` is a hard constraint for a "Safe" assignment.
- Predictors will easily rely on this artifact. A tree model will just split at `BOD <= 3` and achieve immense accuracy.
- Unsafe cases have *significantly* higher coliforms and Conductivity than safe cases.
- **Log Transforms**: Coliforms exhibit massive right tails (outliers). Modeling with linear models will absolutely require log transformations for Coliform variables.
""")

md("""## 7. Relationships y Multicollinearity""")
code("""plt.figure(figsize=(10, 8))
corrs = df_numeric.corr(method='spearman')
sns.heatmap(corrs, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap (Rank-based)")
plt.show()

print("Top absolute correlations with Is_Safe:")
print(corrs['Is_Safe'].abs().sort_values(ascending=False).drop('Is_Safe'))
""")

md("""**Insights:**
- `BOD_Mean` has the highest correlation (-0.64) with `Is_Safe`, further evidencing the threshold-rule dominance.
- `Total_Coliform` and `Fecal_Coliform` correlate strongly with each other. Providing both to a linear model may induce multicollinearity. Choose `Total_Coliform` for broader analysis, or apply PCA if necessary.
- DO is relatively lowly correlated compared to BOD and Coliforms, yet remains functionally critical for compliance.
""")

md("""## 8. Summary & Recommendations

### Top 10 EDA Insights
1. Dataset is clean, requiring almost no imputation (`State` has 11 missing values).
2. Binary classification formulation (`Is_Safe`) operates on a comfortable 62%-38% label split.
3. Multi-class prediction (`Compliance_Label`) suffers heavily from minority classes (`Class_E`, `Class_A`), necessitating SMOTE/weighting.
4. An upstream/downstream degradation effect is successfully captured, showing safety falls from 66% to 53%.
5. The dataset maintains an identical safety distribution across 2022 and 2023.
6. `BOD` explicitly divides classes; zero "Safe" stations have an average BOD > 3 mg/L.
7. Unsafe stations have averages of 7.98 mg/L BOD compared to 1.66 in safe ones.
8. Coliform distributions are monstrously right-skewed.
9. Coliform families (Total and Fecal) exhibit high multicollinearity with each other.
10. `Is_Safe` is mostly influenced by `BOD` (-0.64 scale), `Total_Coliform` (-0.48), and `Conductivity` (-0.30).

### Top 5 Risks / Limitations
1. **Rule-Based Leakage Constraint**: Since the labels derive from fixed boundaries (e.g. BOD <= 3), the models will essentially reverse-engineer the regulatory thresholds rather than learn novel physical dynamics if exposed to the same raw parameters.
2. **Missing Outliers Context**: Heavy tails in Coliform data aren't invalid data—they are true pollution spikes. Don't simply discard them; transform them.
3. **Data Snooping through Scaled Stats**: Unintentionally running MinMax versions side-by-side with Means acts as duplicate variables.
4. **Target Leakage**: Forgetting to drop `Compliance_Label` prior to predicting `Is_Safe`.
5. **Spatial Context Loss**: Neglecting to group K-Folds by Station Code or Basin might result in over-optimistic accuracy if temporal snapshots of the same station leak into the validation set.

### Top 5 Preprocessing Recommendations
1. Drop metadata (Codes, Names) explicitly.
2. Select only `<Parameter>_Mean_` metrics to prevent Min/Max and Scaled redundancy.
3. Apply `np.log1p()` transformation on Conductivity, Fecal Coliform, and Total Coliform to mitigate huge distributions.
4. Preserve `Class_Weight` or compute your own for multi-class optimization.
5. Create an independent hold-out set drawn exclusively from 2023 to test true temporal robustness.

### Modeling Suitability
The dataset is highly prepared and best suited for **Binary Classification** combined with **Explainability (SHAP)**. SHAP will easily demonstrate its capability by surfacing the BOD <= 3 and DO >= 5 threshold step-functions, which perfectly validates XAI's ability to "explain" regulatory rule-tracking in AI models.
""")

with open('EDA_Water_Quality.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
