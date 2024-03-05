# NYCTaxiTripPrediction Using XGBoost

## About the model
XGBoost is a machine learning algorithm that belongs to the ensemble learning category, specifically the gradient boosting framework. It utilizes decision trees as base learners and employs regularization techniques to enhance model generalization. Known for its computational efficiency, feature importance analysis, and handling of missing values, XGBoost is widely used for tasks such as regression, classification, and ranking.

## Project overview
1. Haversine Distance Calculation: The haversine_distance function calculates the great-circle distance between two points on the Earth's surface, given their longitudes and latitudes. This is a crucial step for understanding the distance between pickup and dropoff locations, which can be a significant feature for predicting outcomes like fare or travel time.
2. Feature Engineering: The code then converts the pickup_datetime column into a datetime object and extracts features like the hour of the day (rush_nonrush) and whether the day is a weekday or weekend (weekday_weekend). These features can be very informative for predicting outcomes that might vary based on time of day or day of the week.
3. One-Hot Encoding: The pd.get_dummies function is used to perform one-hot encoding on the rush_nonrush and weekday_weekend columns. This step converts categorical variables into a format that can be better understood by machine learning algorithms. It creates binary columns for each category, which can help the model distinguish between different time periods and days.
4. Data Preparation: The code drops unnecessary columns (vendor_id, pickup_datetime, store_and_fwd_flag, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude) and renames the one-hot encoded columns to be more intuitive (rush_nonrush and weekday_weekend). This step is crucial for reducing the dimensionality of the dataset and focusing on the most relevant features for prediction.
5. Model Prediction: Finally, the code uses an XGBoost model to make predictions on the preprocessed dataset. The xgb.DMatrix function converts the dataset into a format that XGBoost can use for prediction. The model.predict function then generates predictions based on the trained model.

Kaggle Dataset- https://www.kaggle.com/c/nyc-taxi-trip-duration/overview
