import nbformat as nbf
import os

def create_eda_notebook(output_path='notebooks/03_eda_model_readiness.ipynb'):
    nb = nbf.v4.new_notebook()

    nb.cells.append(nbf.v4.new_markdown_cell("# Phase 3: EDA & Model Readiness\nThis notebook analyzes the `model_ready_phase1.csv` dataset, evaluating target distributions, feature availability, and correlations before baseline ML training."))

    nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_palette('muted')

# Load the data
df = pd.read_csv('../data/processed/model_ready_phase1.csv')
print(f"Dataset Shape: {df.shape}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Target and Confidence Distributions"))
    nb.cells.append(nbf.v4.new_code_cell("""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df['strict_compliance_label'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Strict Compliance Label')
axes[0].set_ylabel('Count')

df['available_compliance_label'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Available Compliance Label')

df['label_confidence'].value_counts().plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('Label Confidence Distribution')

plt.tight_layout()
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Feature Availability and Missing Values\nFocusing on allowed features only."))
    nb.cells.append(nbf.v4.new_code_cell("""
allowed_features = [
    'dissolved_oxygen', 'bod', 'ph', 'temperature', 'conductivity', 
    'nitrate', 'fecal_coliform', 'total_coliform', 'fecal_streptococci', 
    'turbidity', 'cod', 'total_dissolved_solids', 'season'
]
# Filter to features actually in the dataframe
allowed_features = [f for f in allowed_features if f in df.columns]

missing_data = df[allowed_features].isnull().mean() * 100
missing_data = missing_data.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
missing_data.plot(kind='barh', color='orange')
plt.title('Percentage of Missing Values in Allowed Features')
plt.xlabel('% Missing')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Boxplots by Compliance Label\nComparing canonical parameters against the available compliance label."))
    nb.cells.append(nbf.v4.new_code_cell("""
plot_features = ['dissolved_oxygen', 'bod', 'ph', 'conductivity', 'nitrate', 'fecal_coliform', 'total_coliform']
plot_features = [f for f in plot_features if f in df.columns]

# Filter out 'Insufficient_Data' for clearer comparison
df_plot = df[df['available_compliance_label'] != 'Insufficient_Data']

fig, axes = plt.subplots(len(plot_features), 1, figsize=(10, 4 * len(plot_features)))
if len(plot_features) == 1:
    axes = [axes]

for ax, feature in zip(axes, plot_features):
    sns.boxplot(data=df_plot, x='available_compliance_label', y=feature, ax=ax)
    ax.set_title(f'{feature} Distribution by Compliance')
    ax.set_yscale('log' if feature in ['bod', 'conductivity', 'fecal_coliform', 'total_coliform'] else 'linear')

plt.tight_layout()
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Correlation Heatmap"))
    nb.cells.append(nbf.v4.new_code_cell("""
numeric_df = df[allowed_features].select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Allowed Numerical Features')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Class Imbalance Analysis\nMapping targets as requested for Phase 3 (Non-Compliant = 1, Compliant = 0) to assess imbalance."))
    nb.cells.append(nbf.v4.new_code_cell("""
# Experiment A: Strict
df_strict = df[(df['label_confidence'] == 'High') & (df['strict_compliance_label'].isin(['Compliant', 'Non-Compliant']))]
if not df_strict.empty:
    exp_a_target = (df_strict['strict_compliance_label'] == 'Non-Compliant').astype(int)
    print("Experiment A (Strict) Target Distribution (1 = Non-Compliant):")
    print(exp_a_target.value_counts(normalize=True) * 100)
else:
    print("Experiment A dataset is empty.")

print("\\n")

# Experiment B: Operational
df_oper = df[(df['label_confidence'].isin(['High', 'Medium'])) & (df['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))]
if not df_oper.empty:
    exp_b_target = (df_oper['available_compliance_label'] == 'Non-Compliant').astype(int)
    print("Experiment B (Operational) Target Distribution (1 = Non-Compliant):")
    print(exp_b_target.value_counts(normalize=True) * 100)
else:
    print("Experiment B dataset is empty.")
"""))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generated EDA notebook at {output_path}")

if __name__ == "__main__":
    create_eda_notebook()
