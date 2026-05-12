# Live LIME Explanation Summary
Computed LIME explanations for 40 live station predictions.

During LIME perturbation, sklearn may warn that some features are all-NaN in the live background. This warning is suppressed only inside repeated LIME prediction calls to keep logs readable. It does not hide prediction failures.
