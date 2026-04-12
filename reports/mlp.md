    # Model: MLP

    ## Overview

    **Description:** Multi-Layer Perceptron (2 hidden layers, 256 units, ReLU, Adam optimizer)

    **Why chosen:** Chosen as a universal approximator for non-linear mapping from spike-count features to continuous controller outputs. Simple to train and easily exportable to ONNX for on-device Synapse App inference.

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
    | 2.5 | 0.0723 | 0.0220 |
| 3.0 | 0.1068 | 0.0240 |
| 3.5 | 0.1068 | 0.0227 |
| 4.0 | 0.0953 | 0.0185 |

    **Best k:** `3.0` -> Test R2 = **0.0240**

    ### Per-Channel R2 Breakdown (k = 3.0)

    | Channel | Test R2 |
    |---------|---------|
    | Ch 0 | 0.1108 |
| Ch 1 | 0.1286 |
| Ch 2 | 0.1017 |
| Ch 3 | 0.0106 |
| Ch 4 | 0.0000 |
| Ch 5 | 0.0000 |
| Ch 6 | 0.0000 |
| Ch 7 | 0.0000 |
| Ch 8 | 0.0000 |
| Ch 9 | 0.0000 |
| Ch 10 | -0.0563 |
| Ch 11 | -0.0070 |

    ---

    ## Plots

    ### R2 vs Threshold k

    ![R2 vs k](figures\mlp_r2_vs_k.png)
*Mean train and test R2 across k values for MLP*

    ### Per-Channel R2 (Best k = 3.0)

    ![Per-channel R2](figures\mlp_per_channel_k3.0.png)
*Per-channel test R2 for k=3.0*

    ### Predicted vs Actual -- Channel 0 (Best k = 3.0)

    ![Pred vs Actual](figures\mlp_pred_vs_actual_k3.0.png)
*Predicted vs actual for channel 0, k=3.0*

    ### Residuals -- Channel 0 (Best k = 3.0)

    ![Residuals](figures\mlp_residuals_k3.0.png)
*Residual distribution for channel 0, k=3.0*

    ---

    ## Analysis

    ### Performance Trend vs k

    Test R2 **decreases** as k increases from 2.5 to 4.0.
    At low k, the threshold is loose -- many sub-threshold noise fluctuations are
    counted as spikes, inflating feature values and adding noise.
    At high k, only strong deflections are counted -- fewer features, but higher
    SNR. The optimal k for this model is **3.0**.

    ### Overfitting / Underfitting

    The largest train-test gap is **0.0841** at k=3.5. A moderate gap indicates mild overfitting. Regularisation or early stopping could close this gap.

    ### Strengths

    Fast training and inference. Simple architecture makes ONNX export straightforward. Handles non-linear channel interactions.

    ### Weaknesses

    Ignores temporal order across bins. Can overfit with limited data. Sensitive to learning rate.

    ---

    ## Conclusion

    **MLP** achieves weak decoding performance with a best test R2 of **0.0240** at k=3.0. We recommend using **k=3.0** for this model in the final BCI pipeline. For deployment in the Synapse App, this model is a good candidate due to its speed and ONNX compatibility.
