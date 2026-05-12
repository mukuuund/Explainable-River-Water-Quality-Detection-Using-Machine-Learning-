# Phase 3.6: Clean Model Comparison

## Important Context: Phase 3 Leakage Correction
> [!WARNING]
> **Phase 3 perfect metrics were invalid due to hidden raw DO/BOD/pH leakage.**
> The Phase 3.5 audit detected that raw variants (like `dissolved_o2`, `bod_mean_mgl`) were inadvertently leaking into the features. Phase 3.6 corrects this by using explicit canonical allowlists. **Final model metrics should come only from Phase 3.6.**

## 1. Group Split Performance (Headline Metrics)
The models below were evaluated using GroupShuffleSplit to prevent spatial/station data leakage. This is the most defensible measure of real-world generalization.

| Variant                   | Split_Type   | Model                      |   Accuracy |   Balanced_Accuracy |   Precision_NonCompliant |   Recall_NonCompliant |   F1_NonCompliant |   ROC_AUC |
|:--------------------------|:-------------|:---------------------------|-----------:|--------------------:|-------------------------:|----------------------:|------------------:|----------:|
| Core Regulatory Model     | Group_Split  | DummyClassifier            |   0.662461 |            0.5      |                 0        |              0        |          0        |  0.5      |
| Core Regulatory Model     | Group_Split  | LogisticRegression         |   0.913249 |            0.90702  |                 0.859729 |              0.88785  |          0.873563 |  0.977815 |
| Core Regulatory Model     | Group_Split  | DecisionTree (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Core Regulatory Model     | Group_Split  | RandomForest (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Core Regulatory Model     | Group_Split  | GradientBoosting (depth=5) |   1        |            1        |                 1        |              1        |          1        |  1        |
| Extended Clean Model      | Group_Split  | DummyClassifier            |   0.662461 |            0.5      |                 0        |              0        |          0        |  0.5      |
| Extended Clean Model      | Group_Split  | LogisticRegression         |   0.903785 |            0.89644  |                 0.846154 |              0.873832 |          0.85977  |  0.97688  |
| Extended Clean Model      | Group_Split  | DecisionTree (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Extended Clean Model      | Group_Split  | RandomForest (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Extended Clean Model      | Group_Split  | GradientBoosting (depth=5) |   1        |            1        |                 1        |              1        |          1        |  1        |
| True Auxiliary-Only Model | Group_Split  | DummyClassifier            |   0.662461 |            0.5      |                 0        |              0        |          0        |  0.5      |
| True Auxiliary-Only Model | Group_Split  | LogisticRegression         |   0.62776  |            0.60445  |                 0.456    |              0.53271  |          0.491379 |  0.649427 |
| True Auxiliary-Only Model | Group_Split  | DecisionTree (depth=5)     |   0.679811 |            0.618525 |                 0.531792 |              0.429907 |          0.475452 |  0.677514 |
| True Auxiliary-Only Model | Group_Split  | RandomForest (depth=5)     |   0.70347  |            0.650134 |                 0.571429 |              0.485981 |          0.525253 |  0.712928 |
| True Auxiliary-Only Model | Group_Split  | GradientBoosting (depth=5) |   0.753943 |            0.698543 |                 0.672619 |              0.528037 |          0.591623 |  0.766795 |

## 2. Random Split Performance
Traditional 80/20 random stratified split.

| Variant                   | Split_Type   | Model                      |   Accuracy |   Balanced_Accuracy |   Precision_NonCompliant |   Recall_NonCompliant |   F1_NonCompliant |   ROC_AUC |
|:--------------------------|:-------------|:---------------------------|-----------:|--------------------:|-------------------------:|----------------------:|------------------:|----------:|
| Core Regulatory Model     | Random_Split | DummyClassifier            |   0.662992 |            0.5      |                 0        |              0        |          0        |  0.5      |
| Core Regulatory Model     | Random_Split | LogisticRegression         |   0.925984 |            0.920056 |                 0.881279 |              0.901869 |          0.891455 |  0.978212 |
| Core Regulatory Model     | Random_Split | DecisionTree (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Core Regulatory Model     | Random_Split | RandomForest (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Core Regulatory Model     | Random_Split | GradientBoosting (depth=5) |   1        |            1        |                 1        |              1        |          1        |  1        |
| Extended Clean Model      | Random_Split | DummyClassifier            |   0.662992 |            0.5      |                 0        |              0        |          0        |  0.5      |
| Extended Clean Model      | Random_Split | LogisticRegression         |   0.924409 |            0.920017 |                 0.873874 |              0.906542 |          0.889908 |  0.979144 |
| Extended Clean Model      | Random_Split | DecisionTree (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Extended Clean Model      | Random_Split | RandomForest (depth=5)     |   1        |            1        |                 1        |              1        |          1        |  1        |
| Extended Clean Model      | Random_Split | GradientBoosting (depth=5) |   1        |            1        |                 1        |              1        |          1        |  1        |
| True Auxiliary-Only Model | Random_Split | DummyClassifier            |   0.662992 |            0.5      |                 0        |              0        |          0        |  0.5      |
| True Auxiliary-Only Model | Random_Split | LogisticRegression         |   0.584252 |            0.57273  |                 0.410714 |              0.537383 |          0.465587 |  0.626179 |
| True Auxiliary-Only Model | Random_Split | DecisionTree (depth=5)     |   0.722835 |            0.67035  |                 0.605556 |              0.509346 |          0.553299 |  0.721485 |
| True Auxiliary-Only Model | Random_Split | RandomForest (depth=5)     |   0.744882 |            0.695019 |                 0.644444 |              0.542056 |          0.588832 |  0.766155 |
| True Auxiliary-Only Model | Random_Split | GradientBoosting (depth=5) |   0.80315  |            0.745854 |                 0.787097 |              0.570093 |          0.661247 |  0.809943 |

## 3. Ablation Conclusion
> [!NOTE]
> **The True Auxiliary-Only model represents the valid ablation result.** It confirms the exact predictive power of supporting parameters when core regulatory metrics are missing.
