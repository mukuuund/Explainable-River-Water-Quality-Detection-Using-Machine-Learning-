# Phase 3.5: Model Sanity Audit and Leakage Check

## ❌ AUDIT FAILED
The audit detected severe issues that must be addressed before proceeding to Phase 4.

### Reasons for Failure:
- Found 28 leakage columns in training features.

---

### 1. Feature Leakage Audit
Suspicious columns found:
| Experiment     | Feature                        | Leakage_Warning     |
|:---------------|:-------------------------------|:--------------------|
| Full Model     | use_based_class                | General Leakage     |
| Ablation Model | dissolved_oxygen_min_mgl       | Hidden Core Leakage |
| Ablation Model | dissolved_oxygen_max_mgl       | Hidden Core Leakage |
| Ablation Model | ph_min                         | Hidden Core Leakage |
| Ablation Model | ph_max                         | Hidden Core Leakage |
| Ablation Model | bod_min_mgl                    | Hidden Core Leakage |
| Ablation Model | bod_max_mgl                    | Hidden Core Leakage |
| Ablation Model | dissolved_oxygen_mean_mgl      | Hidden Core Leakage |
| Ablation Model | ph_mean                        | Hidden Core Leakage |
| Ablation Model | bod_mean_mgl                   | Hidden Core Leakage |
| Ablation Model | total_dissolved_solids         | Hidden Core Leakage |
| Ablation Model | type_water_body                | Hidden Core Leakage |
| Ablation Model | name_of_water_body             | Hidden Core Leakage |
| Ablation Model | use_based_class                | General Leakage     |
| Ablation Model | use_of_water_in_down_stream    | Hidden Core Leakage |
| Ablation Model | odor                           | Hidden Core Leakage |
| Ablation Model | dissolved_o2                   | Hidden Core Leakage |
| Ablation Model | phenophelene_alkanity          | Hidden Core Leakage |
| Ablation Model | sulphate                       | Hidden Core Leakage |
| Ablation Model | phosphate                      | Hidden Core Leakage |
| Ablation Model | biochemical_oxygen_demand_mg_l | Hidden Core Leakage |
| Ablation Model | chemical_oxygen_demand_mg_l    | Hidden Core Leakage |
| Ablation Model | dissolved_oxygen_mg_l          | Hidden Core Leakage |
| Ablation Model | total_dissolved_solids_mg_l    | Hidden Core Leakage |
| Ablation Model | potential_of_hydrogen_ph       | Hidden Core Leakage |
| Ablation Model | sulphate_mg_l                  | Hidden Core Leakage |
| Ablation Model | total_phosphorus_mgp_l         | Hidden Core Leakage |
| Ablation Model | odour                          | Hidden Core Leakage |
### 2. Duplicate Contamination Audit
|   Total_Rows |   Full_Row_Duplicates |   Feature_Only_Duplicates |   Feature_Target_Duplicates |   Train_Test_Contamination_Count |   Label_Collisions_Count |
|-------------:|----------------------:|--------------------------:|----------------------------:|---------------------------------:|-------------------------:|
|         3171 |                     0 |                         0 |                           0 |                                0 |                        0 |

### 3. Shuffled-Label Sanity Result
| Experiment                     |   Balanced_Accuracy |   ROC_AUC |   F1_NonCompliant |
|:-------------------------------|--------------------:|----------:|------------------:|
| Shuffled Labels (DecisionTree) |            0.477285 |  0.477285 |          0.305164 |

(Values should be near ~0.5. If high, indicates leakage).

### 4. Group Split & Restricted Model Comparison
| Model                          |   Balanced_Accuracy |   ROC_AUC |   F1_NonCompliant | Group_Column   |
|:-------------------------------|--------------------:|----------:|------------------:|:---------------|
| DecisionTree_Unrestricted      |            1        |  1        |          1        | station_name   |
| DecisionTree_Restricted_d5_l10 |            1        |  1        |          1        | station_name   |
| RandomForest_Restricted_d5_l10 |            0.944849 |  0.992279 |          0.931116 | station_name   |

### 5. True Auxiliary-Only Performance
| Split_Type   | Model                   |   Balanced_Accuracy |   ROC_AUC |   F1_NonCompliant |
|:-------------|:------------------------|--------------------:|----------:|------------------:|
| Random_Split | DecisionTree_Restricted |            0.732424 |  0.807274 |          0.649299 |
| Group_Split  | DecisionTree_Restricted |            0.739174 |  0.787862 |          0.657258 |