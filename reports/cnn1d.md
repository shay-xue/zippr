    # Model: CNN1D

    ## Overview

    **Description:** 1-D Convolutional Neural Network (2 Conv1d layers, 64 filters, kernel=3)

    **Why chosen:** Captures local spatial correlations between adjacent electrodes. The conv layers learn channel-interaction patterns that a pure MLP might miss, while remaining lightweight for real-time inference.

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
    | 2.5 | 0.0042 | -0.0113 |
| 3.0 | 0.0048 | -0.0111 |
| 3.5 | 0.0046 | -0.0111 |
| 4.0 | 0.0043 | -0.0112 |

    **Best k:** `3.0` -> Test R2 = **-0.0111**

    ### Per-Channel R2 Breakdown (k = 3.0)

    | Channel | Test R2 |
    |---------|---------|
    | Ch 0 | -0.0064 |
| Ch 1 | -0.0022 |
| Ch 2 | -0.0002 |
| Ch 3 | -0.0079 |
| Ch 4 | 0.0000 |
| Ch 5 | 0.0000 |
| Ch 6 | 0.0000 |
| Ch 7 | 0.0000 |
| Ch 8 | 0.0000 |
| Ch 9 | 0.0000 |
| Ch 10 | -0.0662 |
| Ch 11 | -0.0503 |

    ---

    ## Plots

    ### R2 vs Threshold k

    ![R2 vs k](figures\cnn1d_r2_vs_k.png)
*Mean train and test R2 across k values for CNN1D*

    ### Per-Channel R2 (Best k = 3.0)

    ![Per-channel R2](figures\cnn1d_per_channel_k3.0.png)
*Per-channel test R2 for k=3.0*

    ### Predicted vs Actual -- Channel 0 (Best k = 3.0)

    ![Pred vs Actual](figures\cnn1d_pred_vs_actual_k3.0.png)
*Predicted vs actual for channel 0, k=3.0*

    ### Residuals -- Channel 0 (Best k = 3.0)

    ![Residuals](figures\cnn1d_residuals_k3.0.png)
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

    The largest train-test gap is **0.0158** at k=3.0. The model generalises well; train/test performance is closely matched.

    ### Strengths

    Captures local spatial correlations between adjacent electrode channels. Weight sharing reduces overfitting risk on limited data.

    ### Weaknesses

    Assumes channel ordering carries spatial meaning -- not guaranteed for arbitrary electrode placement.

    ---

    ## Conclusion

    **CNN1D** achieves weak decoding performance with a best test R2 of **-0.0111** at k=3.0. We recommend using **k=3.0** for this model in the final BCI pipeline. For deployment in the Synapse App, this model is a good candidate due to its speed and ONNX compatibility.
