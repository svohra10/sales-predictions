
# Sales Predictions

Predicting retail item sales by outlet with scikit-learn. The project cleans a store sales dataset, explores what relates to sales, and compares a baseline, a linear regression, and a decision tree on held-out data.

**Author:** Sheel Vohra

## Business question

A retailer wants to know which item and store characteristics relate to sales, and whether sales at an item and outlet level can be predicted well enough to support decisions such as shelf placement and stocking.

## Data

- 8,523 rows and 12 columns, one row per item at an outlet
- **Target:** `Item_Outlet_Sales` (mean about $2,181, median about $1,794, max about $13,087, right-skewed)
- **Item features:** identifier, weight, fat content, visibility, type, MRP (list price)
- **Outlet features:** identifier, establishment year, size, location tier, outlet type

## Data cleaning

| Issue | Handling |
|---|---|
| No duplicate rows | None needed |
| `Item_Weight` missing for 1,463 rows (about 17%) | Imputed inside the modeling pipeline, so imputation values are learned from training data only |
| `Outlet_Size` missing for 2,410 rows (about 28%) | Filled using outlet location tier: Tier 2 outlets are all Small, so missing Tier 2 values were set to Small. Tier 3 outlets are Medium or High, so missing Tier 3 values were set to Medium, which is a judgment call |
| Inconsistent `Item_Fat_Content` labels (`LF`, `low fat`, `reg`) | Standardized to `Low Fat` (5,517) and `Regular` (3,006) |

## Exploratory analysis

- Most items have low visibility, with a tail of high-visibility outliers. Some items show a visibility of exactly 0, which is probably a missing-data placeholder because an item on a shelf cannot have zero visibility.
- `Item_MRP` is the only numeric variable with a moderate correlation to sales. The others are weakly correlated.
- Seafood shows the highest average sales and the largest variance across item types, with starchy foods second.
- Breakfast items are the most visible on average, with seafood second.
- The visibility observation suggests low-visibility items could be moved to more visible spots. This is a hypothesis for testing, not a result of the models in this project.

## Modeling

All models use a scikit-learn pipeline: mean imputation and standard scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns. Data was split 75/25 with `random_state=42`.

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Baseline (predict the mean) | 0.00 | -0.005 | 1,665 | 1,326 |
| Linear regression | 0.670 | invalid (see below) | invalid | invalid |
| Decision tree (default settings) | 1.00 | 0.231 | 1,457 | 991 |

## Findings

- **The decision tree beats the baseline on unseen data**, with test RMSE about 12% lower and MAE about 25% lower, but it explains only about 23% of the variance in sales.
- **The decision tree overfits.** A perfect training R² of 1.0 against a test R² of 0.23 means it memorized the training data. Training R² alone should not be used to choose a model.
- **The linear regression test result is invalid.** It fit the training data (R² 0.67) but produced absurd predictions on the test set (R² around -4.7e19). The likely cause is one-hot encoding the high-cardinality identifier columns (`Item_Identifier`, `Outlet_Identifier`), which creates thousands of near-redundant columns and unstable coefficients in an unregularized model. This has not been confirmed.
- The baseline comparison shows these models capture some signal, and `Item_MRP` appears to be the most informative feature.

## Limitations

- Models were not tuned and no cross-validation was used.
- Identifier columns were left in as features.
- The dataset has no price changes, promotions, seasonality, or competitor data, so it cannot answer pricing or promotion questions.
- Relationships in the exploratory analysis are associations in historical data, not causal effects.

## Next steps

- Drop or group the identifier columns
- Limit tree depth and tune hyperparameters with cross-validation
- Try regularized regression (Ridge or Lasso) and a random forest
- Report feature importances for the final model
- Check the zero-visibility values and decide how to treat them

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab

## How to run

1. Open `Final_Sales_Prediction_project.ipynb` in Google Colab or Jupyter.
2. Update the `sales_predictions` file path in the first cells to point to your copy of `sales_predictions.csv`. The notebook currently reads from Google Drive.
3. Run all cells.
