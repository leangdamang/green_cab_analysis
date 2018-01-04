Green cabs are taxis that service the outer boroughs of NYC - while yellow cabs are more prevalent and well-known throughout the city, in the month of Sept 2015 alone Green Cabs accounted for 1.5 Million trips. In this notebook, I evaluate green cab data to create a tip predictor off an inherently noisy dataset. Through data wrangling, feature engineer, and a regression model built onto of a classifier, I was able to achieve a MSE of .0029 for the predictor. 


## Cleaning up the data ##
The publicly available dataset tracks every single trip through start/stop coordinates, start/stop times, passenger count, trip distance, payment type, cost data (tip amount, fees, base fare) etc. However, a quick evaluation of the data shows that it's rife with invalid data that needs to be corrected:

> 1) Dollars can never be below 0 (changed to absolute value)
> 2) 0 Coordinates are wrong (Drop these)
> 3) Fares should be at least $2.5 as that's the base (Change to median since these seem to be technical issues)
> 4) Distance should not be 0 (change to median)
> 5) RateCodeIDs should be 1-7 (change to the mode as these are technical issues)
> 6) Outliers for trip duration and trip_distance should be changed (within reason) (outliers beyond a threshold change to median or drop) 

To evaluate the outliers for #6, I created two new fields in the dataset - projected_fare and fare_difference from the Green Cab fare formula 
>> Fare = 2.5 + Distance/5*$.50(X) + $.50*Duration_Seconds(1-X) 

since cabs calculate via distance only when the cab is mobile and time when the cab is idle. From this, I was able to calculate the trips that far surpassed to projected cost and eliminated any outliers more than 3 STD DEVs from projection. 

I created other features including airport trips, determined by establishing bounding boxes around Newark, La Guardia, and John F. Kennedy Airports and evaluating start or stop coordinates that fall inside the bounding boxes, the Speed of the cab, any cabs that moved between boroughs, and time based features (day of week, day of month, hour).  

## Building the Model ## 
Sadly only about 40% of all rides tipped any amount. Knowing that, I tackled the model in 2 stages:
> 1) Predict if someone tipped at all
> 2) If they did tip, predict the amount 

I broke out the model this way so that I could train the regression solely on non-zero outcomes as the zero outcomes would be captured in the first stage, which could get me more accurate results. 

For the classification, I evaluated several models including Logistic Regression, Gaussian Naive Bayes, Random Forest Classifier, AdaBoost, and XGBoost Classifier through cross-validation with 5 folds. The Boosting models performed the best with AUCs over .954 so I opted for XGBoost because of its speed. I tuned the hyperparameter through grid searching the number of produced trees and the regularization lambda to optimize the model. 

Ultimately I found that payment method was the greatest indicator for the existance of tips - those paying by cash almost never tipped (or the cabbie never reported these tips), followed by the log(rate) of the taxi, and hour of day. 

For the regressor, I settled on XGB Regressor over linear regression, random forest, and gradient boosting regressors since it provided the best results. Similar to the classifier, I used grid search to optimize the hyperparameters and incorporated L2 regularization into the model to reduce overfitting. Overall, the regressor was great at minimizing MSE - it predicts within .25% away from the actual tip percent.

The top features of my regression model were all engineered - fare_difference, log(rate), and projected_fare. Logically this makes sense - these all relate to perceived service and expediency, which is what tip is designed to reward. 

In the future, I could improve this model through a variety of ways:

1) SVM & Neural Nets - SVM performs well with outlier data which I could incorporate into a future regressor. In addition, NNs are great at learning feature interactions so could detect a significant new feature that would improve the model. However, both are computationally taxing so I opted against them for this exercise.

2) Stacking models together - I could build out the various regressors tested and stack all the models together to create an ensemble model with lowered variance. In addition, I could build out a regression model without any outliers to stack into the main model, which should reduce the overestimation in my model.

3) Pull in relevant geo-data - I attempted this in the beginning but hit some roadblocks in terms of computation and rate limiting, but neighborhoods could have a significant impact on tip percentage. I would evaluate whether pickups/dropoffs at restaurant and bar-dense neighborhoods late at night have higher tipping rates. Hypothetically these exchanges might have higher credit card transaction rates (and there might be an inebriation effect), thus have increased tip percentages. 

## Anomaly Detection ##
One of the things I noticed in the dataset was the potential for fraud in the taxi data, especially when it came to cash transactions. I defined fraud as cash transactions with inordinately high fare_amounts compared to a low travel distance or time duration. I was able to manually identify over 6k transactions that lasted less than a minute, went 0 distance, but accounted for tens of thousands in revenue in just Sept alone. I tried building that out further by creating an anomaly detection model.

Working with a cash only dataset (payment_type == 2), I preserved the outliers purposefully and only cleaned data that was obviously erroneous, dropping bad ratecodeids, 0 coordinate trips, and fares that are less than $2.5.

My anomaly detector ultimately detected the long tail for lengthy and expensive cash trips, but not necessarily those far away. These can be proxies for very terrible traffic conditions or drivers who chose suboptimal routes and thus gouge their customers. I attempted to cluster anomalies together to derive some additional learnings, but all these anomalies were too similar.
