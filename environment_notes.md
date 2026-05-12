# Environment Notes

## Dependency Pinning

| Package | Version | Note |
|---------|---------|------|
| numpy   | 1.26.4  | Required by pandas, pyarrow, numexpr, bottleneck (all compiled against numpy 1.x) |
| shap    | 0.49.1  | shap 0.51 requires numpy>=2 and broke the Anaconda environment |
| scikit-learn | (system) | Used for model pipeline, permutation importance |

> **Warning:** Do NOT run `pip install shap` without pinning.
> Use `pip install "shap<0.50" --no-deps` or `pip install shap==0.49.1`.
