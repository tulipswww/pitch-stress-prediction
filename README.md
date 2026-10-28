# Pitch Stress Prediction

MATLAB scripts and finite element datasets for the manuscript **Rapid Prediction of von Mises Stress in Pitch Layers beneath Thick Optical Glass Components**.

The models estimate local von Mises stress in the pitch layer beneath the center and a corner of a square glass component, using geometric parameters and local settlement under short-term self-weight loading.

## Requirements

- MATLAB with Statistics and Machine Learning Toolbox.
- The scripts were tested in MATLAB R2025b.

## Files

The `center point` and `corner point` folders contain the corresponding scripts and datasets:

| File | Purpose |
| --- | --- |
| `fit_*_model.m` | Fit the stress prediction model. |
| `analyze_*_sensitivity.m` | Analyze sensitivity to individual regression coefficients. |
| `validate_*_model.m` | Evaluate the model on the validation datasets. |
| `*_training_data.xlsx` | Training data. |
| `*_validation_S1.xlsx` | Pitch-thickness interpolation data. |
| `*_validation_S2.xlsx` | Glass-thickness extrapolation data. |
| `*_validation_S3.xlsx` | Combined interpolation and extrapolation data. |

Here, `*` represents `center` or `corner`.

## Usage

Set the MATLAB current folder to `center point`, then run:

```matlab
fit_center_model
analyze_center_sensitivity
validate_center_model
```

For the corner model, set the current folder to `corner point` and run:

```matlab
fit_corner_model
analyze_corner_sensitivity
validate_corner_model
```

Keep the Excel files in the same folder as their scripts. Results are displayed in the Command Window and retained in the MATLAB workspace. Validation MAPE is reported as an equal-weight average across pitch-thickness groups.
