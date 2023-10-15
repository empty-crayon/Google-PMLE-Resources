# Class Transcript Notes

## BigQuery Preprocessing

- Involves two aspects: representation transformation and feature construction.
- **Representation Transformation**:
  - Converting numeric feature to categorical through bucketization.
  - Converting categorical features to numeric representation using methods like one-hot encoding, learning with counts, and sparse feature embeddings.
- Some models work with either numeric or categorical features, while others handle mixed types.
- Models can benefit from different representations (numeric and categorical) of the same feature.
- **Feature Construction**:
  - Creating new features using techniques like polynomial expansion, univariate mathematical functions, or feature crossing to capture interactions.
  - Features can also be constructed using business logic from the ML use case.

## BigQuery ML Feature Preprocessing

- Supports two types of feature preprocessing: automatic and manual.
- Automatic preprocessing occurs during training.
- `transform` clause allows custom preprocessing using manual preprocessing functions.
- Preprocessing functions can be used outside the `transform` clause.
- More details available in BigQuery ML reference documentation.

## Feature Engineering with BigQuery

- BigQuery aids feature engineering using SQL for common preprocessing tasks.
- Example: Preprocessing a dataset of taxi rides in New York City using SQL filtering operations to exclude certain data.
- SQL math and data processing functions are valuable for calculations and parsing common data formats (e.g., timestamps).

## Example Preprocessing with Dates in SQL

- Various preprocessing techniques for dates using SQL in BigQuery ML:
  - Extracting date parts into different columns (year, month, day, etc.).
  - Calculating time period between current date and columns in terms of years, months, days, etc.
  - Extracting specific features of the date (day of the week, weekend, holiday, etc.).
  - Visualization of date-related queries using SQL.

## One-Hot Encoding in BigQuery ML

- For non-numeric columns (except timestamp), BigQuery ML performs one-hot encoding transformation, generating a feature for each unique value.

## Incorporating Feature Engineering into BigQuery ML

- Demonstrating how to incorporate feature engineering into a BigQuery ML model using the New York taxi driver dataset.
- Regression problem to predict taxi fare price in New York City.
- Baseline model used to set a performance goal for error metrics.
- Evaluation metric: Root Mean Square Error (RMSE).

## Model Evaluation in BigQuery ML

- BigQuery automatically splits data for training and evaluation.
- Using `ML.EVALUATE` function to evaluate the performance of the regressor.
- RMSE is the primary evaluation metric, measuring the difference between model predictions and observed values.
- Lower RMSE indicates better model performance.

## Understanding RMSE

- RMSE measures the average error between model predictions and actual values in the units being measured.
- A lower RMSE implies a more accurate model prediction.

## Summary of Models in the Lab

- Table displaying various models in the lab, highlighting the differences in RMSE based on feature engineering.

