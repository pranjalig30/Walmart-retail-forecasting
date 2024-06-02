# Walmart-retail-forecasting

Walmart, the American giant leading in the retail industry, like all retail companies, is interested to gather insights on its historical data to forecast future unit sales of respective products to better mitigate opportunity losses and manage its inventory. 
In this project, our team worked with hierarchical sales data from Walmart, covering stores across California, Texas, and Wisconsin to train our prediction models and record the performance metrics. 
The extensive datasets include details at item, department, product category, and store levels, along with additional explanatory variables like price, promotions, day of the week, and special events. 
Leveraging the rich datasets by thoroughly examining the columns and diligently engineering the features is crucial for improving forecasting accuracy.

The solution involves exploring various machine learning models to forecast unit sales. 
It encompasses ensemble methods such as LightGBM, and CatBoost, alongside neural network architectures like LSTM, and linear regression models like Ridge. 
The choice of models aims to capture complex patterns in the sales data and enhance predictive accuracy. Our solution aim to use the most relevant features to accurately predict the sales amount for each product,
which brings proactive competitive advantage to Walmart in the realm of supply chain and operations like inventory management.

#### Tools Used
- GoogleColab for collaborative coding and computation
- Numpy (v 1.17.4) and Pandas (v 0.24.2) for data manipulation
- Matplotlib (v 3.0.3), CSV (v 1.0), and Seaborn (v 0.9.0) for analysis and visualization
- Scikit Learn (v 0.21.3), CatBoost (v 1.2.2) and LightGBM (v 2.3.0) for ensemble approach and linear regression machine learning algorithms
- TensorFlow (v 2.0.0)  and Keras (v 2.3.1) for neural network implementations

Cross-validation is key in building models by iteratively training and testing on different data subsets, ensuring robustness. However, in time series forecasting, this method can cause data leakage by using future data for training, making it unsuitable due to its temporal dependency.
In time series forecasting, predictions are inherently future-oriented.  The test sets are now always more forward in time than the training sets, therefore avoiding any data leakage when building our model.

### Models Explored

#### LightGBM (Ensemble Approach)
We used LightGBM with a uniform training-validation-test split and tuned hyperparameters via grid search. Key features included `['roll_mean_7', 'wday', 'sold_lag_7', 'week_num', 'roll_mean_14', 'd', 'event_name_1', 'sell_price', 'sold_lag_14', 'snap']`. The model's performance was evaluated using RMSE, with the final Kaggle submission yielding a WRMSSE of 4.97. We also plotted actual vs. predicted unit sales for comparison.

#### CatBoost (Ensemble Approach)
CatBoost, based on decision trees and gradient boosting, was used. Hyperparameters were tuned using grid search to minimize errors, resulting in a WRMSSE score of 4.94. This method iteratively combines weak models to form a robust predictive model.

#### Ridge Model (Linear Regression)
Ridge Regression adds a regularization term to prevent overfitting. We experimented with different α values using GridSearchCV, but the best performance (RMSE: 4.56331) was achieved with α=100, similar to the default setting (α=1.0), indicating effective regularization.

#### Random Forest (Ensemble Approach)
Using the `RandomForestRegressor` from scikit-learn, we constructed 77 decision trees with controlled depth and sample constraints to prevent overfitting. Despite limited hyperparameter tuning, we included it for exploration.

#### LSTM (Neural Network)
LSTM networks were implemented in two ways for time series forecasting. The first used past sales and oneDayBeforeEvent features, scoring 0.77 on Kaggle. The second included nine features such as sales, prices, and events, scoring 1.13. Both models involved data optimization, feature engineering, and sequence creation for effective training and prediction. 

Each model explored different aspects of predictive modeling, offering insights into their performance and applicability for time series forecasting.

WRMSSE (Provided by Kaggle Submission)
Light GBM : 4.97
CatBoost : 4.94
Ridge : 4.56
LSTM : 0.77

### Interpretation of Values of Our Analysis in Business Context 
Forecasts from an LSTM model can significantly aid Walmart's inventory management by predicting when certain products are likely to see increased demand, allowing for better stock preparation. This also benefits marketing strategies, as promotions can be timed with forecasted high-demand periods. Additionally, sales optimization is enhanced as the model can predict slow periods, enabling Walmart to plan strategies to boost sales during those times. The model's ability to predict peak sales periods and potential lulls helps in allocating resources efficiently and maximizing profitability.

The LSTM model, while effective, may have limitations in handling unusual market conditions or sudden changes in consumer behavior. To mitigate this, it's recommended that Walmart continuously refines the model, possibly integrating more dynamic variables and real-time data. This would enhance the model's responsiveness to market shifts and improve forecasting accuracy. Regular updates and adjustments to the model are crucial for maintaining its relevance and effectiveness in a rapidly changing retail environment
