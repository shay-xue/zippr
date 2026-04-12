    # Model: XGBOOST

    ## Overview

    **Description:** XGBoost Gradient-Boosted Trees (MultiOutputRegressor, 100 estimators)

    **Why chosen:** Strong classical baseline for tabular spike-count data. Handles non-linear feature interactions without explicit engineering. One independent ensemble is trained per output channel.

    ---

    ## Data

    ### Preprocessing Summary

    | Parameter | Value |
    |-----------|-------|
    | Sampling rate | 32,000 Hz |
    | Neural channels | 64 (channels 0-63) |
    | Target channels | 12 (channels 64-75) |
    | Bin size | 20 ms = 640 samples |
    | Feature type | Spike count (\|z\| > k) |
    | Target aggregation | Mean per bin |
    | Train/test split | 80/20 chronological |

    ### k Values Tested

    k  { 2.5, 3.0, 3.5, 4.0 }

    ---

    ## Results

    ### R2 Summary Table

    | k | Train R2 | Test R2 |
    |---|---------|--------|
    | 2.5 | 0.4590 | 0.0281 |
| 3.0 | 0.4489 | 0.0349 |
| 3.5 | 0.4396 | 0.0319 |
| 4.0 | 0.4273 | 0.0326 |

    **Best k:** `3.0` -> Test R2 = **0.0349**

    ### Per-Channel R2 Breakdown (k = 3.0)

    | Channel | Test R2 |
    |---------|---------|
    | Ch 0 | 0.0990 |
| Ch 1 | 0.1429 |
| Ch 2 | 0.1189 |
| Ch 3 | 0.0321 |
| Ch 4 | 0.0000 |
| Ch 5 | 0.0000 |
| Ch 6 | 0.0000 |
| Ch 7 | 0.0000 |
| Ch 8 | 0.0000 |
| Ch 9 | 0.0000 |
| Ch 10 | 0.0087 |
| Ch 11 | 0.0172 |

    ---

    ## Plots

    ### R2 vs Threshold k

    ![R2 vs k](figures\xgboost_r2_vs_k.png)
*Mean train and test R2 across k values for XGBOOST*

    ### Per-Channel R2 (Best k = 3.0)

    ![Per-channel R2](figures\xgboost_per_channel_k3.0.png)
*Per-channel test R2 for k=3.0*

    ### Predicted vs Actual -- Channel 0 (Best k = 3.0)

    ![Pred vs Actual](figures\xgboost_pred_vs_actual_k3.0.png)
*Predicted vs actual for channel 0, k=3.0*

    ### Residuals -- Channel 0 (Best k = 3.0)

    ![Residuals](figures\xgboost_residuals_k3.0.png)
*Residual distribution for channel 0, k=3.0*

    ---

    ## Analysis

    ### Performance Trend vs k

    Test R2 **increases** as k increases from 2.5 to 4.0.
    At low k, the threshold is loose -- many sub-threshold noise fluctuations are
    counted as spikes, inflating feature values and adding noise.
    At high k, only strong deflections are counted -- fewer features, but higher
    SNR. The optimal k for this model is **3.0**.

    ### Overfitting / Underfitting

    The largest train-test gap is **0.4310** at k=2.5. This suggests meaningful overfitting -- the model has memorised training patterns that don't generalise. Consider adding dropout, reducing model capacity, or collecting more data.

    ### Strengths

    Excellent on tabular spike-count data. Robust to outlier counts. No normalisation of inputs required. Interpretable via feature importance.

    ### Weaknesses

    Trains independent regressors per channel -- misses cross-channel correlations. Slow with large datasets.

    ---

    ## Conclusion

    **XGBOOST** achieves weak decoding performance with a best test R2 of **0.0349** at k=3.0. We recommend using **k=3.0** for this model in the final BCI pipeline. This model is best suited for offline analysis rather than real-time on-device inference.
