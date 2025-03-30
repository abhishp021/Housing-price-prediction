# Housing Price Prediction Model Report

## 1. Introduction

Aim of this project was to develop a predictive model for housing prices based on various features such as the number of rooms, population, income, and other related variables. The model's primary purpose is to predict the median house value for California districts, which is a regression problem. We have also created an API as well as a webpage where we feed data in `json` format and get the house price prediction. 

## 2. Dataset Overview

The dataset used for this project is the **California Housing Prices** dataset, which contains the following features:

- **MedInc**: Median income of the district
- **AveRooms**: Average number of rooms per household
- **AveOccup**: Average number of occupants per household
- **Population**: Population of the district
- **HouseAge**: Median age of the houses in the district
- **AveBedrms**: Average number of bedrooms per household
- **Latitude**: Latitude of the district
- **Longitude**: Longitude of the district
- **MedHouseVal**: Target variable, which represents the median house value (in thousands of dollars)

The data consists of approximately 20,000 rows with 9 features.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- **Handling Missing Values**: No missing values were there
- **Handling Outliers**: There were few data points where Average Occupancy per room was even greater than 100. Dropped such data points (15)
- **Feature Scaling**: Only scaled those features which were skewed. The continuous variables were scaled using different scalers, including **MinMaxScaler**, **StandardScaler**, and **Log Transformation**. This was done to normalize the data and improve the performance of the model.
- **Feature Engineering**: Created two new features called `OccupPerRoom` and `BedrmsPerRoom` representing number of occupants in each room and Number of bedrooms per room. 
- **Feature Selection**: Features with high skewness were identified, and the appropriate transformations were applied to them.
- **Train-Test Split**: The data was split into training and testing sets with a ratio of 80:20.

## 4. Model Selection

Several regression models were evaluated for the task, including:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **LightGBM Regressor**

Hyperparameter tuning was performed using **GridSearchCV** to optimize each model's performance.
Best model was selected based on combination of mode, GribSearchCV and Scaling method. 

## 5. Model Training

For each **combination of model and scaling** technique, the models were trained on the training dataset. The training process involved:

- **Model Fitting**: The models were fitted on the scaled training data.
- **Hyperparameter Tuning**: Grid search with cross-validation was used to find the best set of hyperparameters for each model.
  
The models were evaluated using **R²** (coefficient of determination), **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)**.

## 6. Model Evaluation

The performance of each model was evaluated on the testing set using the following metrics:

- **R² (R-squared)**: Measures the proportion of variance in the dependent variable explained by the model.
- **RMSE (Root Mean Squared Error)**: Measures the average error between the predicted values and the actual values.
- **MAE (Mean Absolute Error)**: Measures the average of the absolute errors between predicted and actual values.

The models were compared to identify the best-performing model for housing price prediction.

| scaler  | model           | best_params                                                  | r2      | rmse    | mae    |
|---------|-----------------|--------------------------------------------------------------|---------|---------|--------|
| minmax  | linear_regression| "default"                                                    | 0.652968| 0.678694| 0.496430|
| minmax  | decision_tree    | {"max_depth": 5}                                             | 0.599146| 0.729428| 0.532130|
| minmax  | random_forest    | {"max_depth": null, "n_estimators": 100}                     | 0.769472| 0.553161| 0.378074|
| minmax  | xgboost          | {"max_depth": 6, "n_estimators": 100}                        | 0.794697| 0.522020| 0.354119|
| minmax  | lightgbm         | {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100} | 0.803693| 0.510454| 0.348646|
| standard| linear_regression| "default"                                                    | 0.652968| 0.678694| 0.496430|
| standard| decision_tree    | {"max_depth": 10}                                            | 0.615477| 0.714415| 0.476317|
| standard| random_forest    | {"max_depth": null, "n_estimators": 100}                     | 0.769252| 0.553424| 0.377995|
| standard| xgboost          | {"max_depth": 6, "n_estimators": 100}                        | 0.794697| 0.522020| 0.354119|
| standard| lightgbm         | {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100} | 0.801901| 0.512779| 0.350019|
| log     | linear_regression| "default"                                                    | 0.653653| 0.678024| 0.508025|
| log     | decision_tree    | {"max_depth": 10}                                            | 0.615797| 0.714118| 0.475114|
| log     | random_forest    | {"max_depth": null, "n_estimators": 100}                     | 0.769938| 0.552602| 0.376578|
| log     | xgboost          | {"max_depth": 6, "n_estimators": 100}                        | 0.794697| 0.522020| 0.354119|
| log     | lightgbm         | {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100} | 0.804087| 0.509942| 0.348377|
| minmax  | linear_regression| "default"                                                    | 0.652968| 0.678694| 0.496430|
| minmax  | decision_tree    | {"max_depth": 5}                                             | 0.599070| 0.729498| 0.532187|
| minmax  | random_forest    | {"max_depth": null, "n_estimators": 100}                     | 0.769883| 0.552667| 0.377132|
| minmax  | xgboost          | {"max_depth": 6, "n_estimators": 100}                        | 0.794697| 0.522020| 0.354119|
| minmax  | lightgbm         | {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100} | 0.803693| 0.510454| 0.348646|
| standard| linear_regression| "default"                                                    | 0.652968| 0.678694| 0.496430|
| standard| decision_tree    | {"max_depth": 5}                                             | 0.596478| 0.731852| 0.533128|
| standard| random_forest    | {"max_depth": null, "n_estimators": 100}                     | 0.769148| 0.553550| 0.379298|
| standard| xgboost          | {"max_depth": 6, "n_estimators": 100}                        | 0.794697| 0.522020| 0.354119|
| standard| lightgbm         | {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100} | 0.801901| 0.512779| 0.350019|
| log     | linear_regression| "default"                                                    | 0.653653| 0.678024| 0.508025|
| log     | decision_tree    | {"max_depth": 5}                                             | 0.596702| 0.731649| 0.533042|
| log     | random_forest    | {"max_depth": null, "n_estimators": 100}                     | 0.770538| 0.551880| 0.376402|
| log     | xgboost          | {"max_depth": 6, "n_estimators": 100}                        | 0.794697| 0.522020| 0.354119|
| log     | lightgbm         | {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100} | 0.804087| 0.509942| 0.348377|
|---------|-----------------|--------------------------------------------------------------|---------|---------|--------|

## 7. Best Model: LightGBM Regressor

The **LightGBM Regressor** performed the best with an R² score of **0.804**, indicating that the model explained 80.4% of the variance in the housing prices. The RMSE of **0.51** and MAE of **0.35** further validate the model's strong predictive capabilities.

The key hyperparameters for the LightGBM model that contributed to its success were:
- **Learning Rate**: 0.1
- **Number of Estimators**: 100
- **Number of leaves**: 31
- **Maximum Depth**: 5

## 8. Model Deployment

The **LightGBM Regressor** model, along with the scaler used for feature scaling, was saved as a serialized file (`.pkl`). This can be used for predictions on new, unseen data.

- **Model Path**: `models/model_lightgbm_log.pkl`
- **Scaler Path**: `scalers/scaler_lightgbm_log.pkl`

These files can be loaded and used for real-time predictions in a production environment.

## 9. Conclusion

The predictive model for housing prices achieved a strong performance, with **LightGBM** being the top-performing model.

The following steps are recommended for further improvement:
- Incorporate additional external features (e.g., economic factors, local amenities).
- Use feature engineering techniques to create new features from existing ones.
- Evaluate the model's performance on more diverse datasets to test its robustness.

## 10. Running the Docker Container

To run the Docker container and use the model, follow these steps:

1. **Build the Docker Image**:
   ```bash
   docker build -t housing-price-prediction .
   ```
2. ```
    docker run -p 5000:5000 housing-price-prediction
    ```
3. Now the app can be accessed at `http://127.0.0.1:5000`

## 11. API Usage Guide

### 1. **Endpoint**: `/predict`
- **Method**: `POST`
- **Description**: Predicts the housing price based on input features.
  
#### Request Body:
```json
{
  "MedInc": 8.3252,
  "AveRooms": 6.984126984,
  "AveBedrms": 1.023923444,
  "Population": 322.0,
  "AveOccup": 2.555555556,
  "OccupPerRoom": 0.524031272,
  "BedrmsPerRoom": 0.143,
}
```


## 12. Future Work

- **Hyperparameter Optimization**: Further tuning of the hyperparameters using more advanced search techniques like **RandomizedSearchCV** or **Bayesian Optimization**.
- **Ensemble Models**: Combining multiple models into an ensemble (e.g., stacking) to potentially improve performance.