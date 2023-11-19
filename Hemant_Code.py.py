

import pandas as pd
import joblib
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from tqdm import tqdm
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Regression:
    def __init__(self, path='C:/Users/ADMIN/Videos/ml/drive-download-20231105T071312Z-001/qgis_training_data.csv', regressor_opt='lr', no_of_selected_features=None):
        self.path = path
        self.regressor_opt = regressor_opt
        self.no_of_selected_features = no_of_selected_features
        if self.no_of_selected_features is not None:
            self.no_of_selected_features = int(self.no_of_selected_features)
        self.best_regressor = None

    def regression_model(self):
        if self.regressor_opt == 'lr':
            print('\n\t### Training Linear Regression ### \n')
            regressor = make_pipeline(PolynomialFeatures(degree=2),LinearRegression(fit_intercept=True, n_jobs=None, positive=False))
        elif self.regressor_opt == 'dt':
            print('\n\t### Training Decision Tree Regressor ### \n')
            regressor = DecisionTreeRegressor(criterion="squared_error",splitter='best',max_depth=None,min_samples_split=2, min_impurity_decrease=0.0, min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,random_state=42,ccp_alpha=0.0)
        elif self.regressor_opt == 'rf':
            print('\n\t ### Training Random Forest Regressor ### \n')
            regressor = RandomForestRegressor(random_state=42)
            # Hyperparameter tuning using GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
            return grid_search
            # Print the best parameters
            print("Best Parameters:", grid_search.best_params_)

        elif self.regressor_opt == 'lasso':
            print('\n\t### Training Lasso Regressor ### \n')
            regressor = make_pipeline(PolynomialFeatures(degree=5), Lasso(alpha=1.0, fit_intercept=True, random_state=42, selection='cyclic', tol=1e-4))
        elif self.regressor_opt == 'ridge':
            print('\n\t### Training Ridge Regressor ### \n')
            regressor = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0, fit_intercept=False, max_iter=100, tol=1e-3, solver='svd', random_state=42))

        elif self.regressor_opt=='svr': 
            print('\n\t### SVM Regressor ### \n')
            regressor = SVR()  
            regressor_parameters = {'rgr__C':(0.1,1,100),
            'rgr__kernel':('linear','rbf','poly','sigmoid'),
            }
        elif self.regressor_opt=="ab":
            print("\n\t### Training AdaBoostRegressor ### \n")
            regressor=AdaBoostRegressor(base_estimator='deprecated',n_estimators=40,learning_rate=2.0,loss='linear',random_state=42,)
        elif self.regressor_opt=='knn':
            print("\n\t### k-nearest neighbour ### \n")
            regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='brute', leaf_size=20, p=1, metric='euclidean', metric_params=None, n_jobs=-1)
        
        else:
            print('Select a valid regressor \n')
            return None
        return regressor

    def get_data(self):
        reader = pd.read_csv(self.path)
        data = reader.drop(['ADDRESS', 'Price','PIN'], axis=1)
        labels = reader['Price']
        training_data, validation_data, training_labels, validation_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
        return training_data, validation_data, training_labels, validation_labels

    def regression(self):
        training_data, validation_data, training_labels, validation_labels = self.get_data()
        regressor = self.regression_model()

        if regressor is not None:
            regressor.fit(training_data, training_labels)
            predicted = regressor.predict(validation_data)

            print('\n *************** Regression Metrics ***************  \n')
            mse = mean_squared_error(validation_labels, predicted)
            print(f'Mean Squared Error: {mse}')

            r2 = r2_score(validation_labels, predicted)
            print(f'R-squared: {r2}')
            rmse=np.sqrt(mse)
            print(f'Root Mean squared error:{rmse}')
            
            mae=mean_absolute_error(validation_labels,predicted)
            print(f'Mean Absolute error:{mae}')
            # Save the best regressor

            # Save the best regressor
            self.best_regressor = regressor
            joblib.dump(self.best_regressor, f"{self.regressor_opt}_best_regressor.joblib")
            print(f"Saved the best {self.regressor_opt} regressor to {self.regressor_opt}_best_regressor.joblib")




#Print Data set and plot it

data_path = 'C:/Users/ADMIN/Videos/ml/drive-download-20231105T071312Z-001/qgis_training_data.csv'
data = pd.read_csv(data_path)
data=data.drop(['PIN'],axis=1)

# Print the first few rows of the dataset
print("Dataset:")
print(data.head())
print(data.tail())



features = ['UNDER_CONS','RERA','BHK_NO.','SQUARE_FT','READY_TO_M','RESALE']

# Assuming 'Price' is the target variable for plotting
for feature in features:
    plt.scatter(data[feature], data['Price'])
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(f'Scatter Plot of {feature} against Price')
    plt.show()


fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
# Scatter plot
ax.scatter(data['LONGITUDE'], data['LATITUDE'], data['Price'], c='blue', marker='o')

# Set labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Price')

# Set title
ax.set_title('3D Scatter Plot of Latitude, Longitude, and Price')

plt.show()
# Run the regressors
regressors_list = ['lr','dt','rf','lasso','ridge','svr','ab','knn']

for regressor_opt in regressors_list:
    regression = Regression(regressor_opt=regressor_opt, no_of_selected_features=9)
    print(f"\nRunning Regressor: {regressor_opt}")
    regression.regression()

# Optional: Apply the best regressors on test data to generate predictions
test_data_path = 'C:/Users/ADMIN/Videos/ml/drive-download-20231105T071312Z-001/reformed_test_data.csv'  # Replace with the actual path to your test data
output_predictions_path = 'predictions.csv'

for regressor_opt in regressors_list:
    # Load the best regressor
    best_regressor = joblib.load(f"{regressor_opt}_best_regressor.joblib")

    # Load test data
    test_data = pd.read_csv(test_data_path)
    # Drop the "Address" column
    if 'ADDRESS' in test_data.columns:
        test_data = test_data.drop(['ADDRESS'], axis=1)

    # Predictions for test data
    predictions = best_regressor.predict(test_data)

    # Save predictions to a CSV file
    output_filename = f"{regressor_opt}_predictions.csv"
    pd.DataFrame(predictions, columns=['predictions']).to_csv(output_filename, index=False)
    print(f"Saved predictions for {regressor_opt} to {output_filename}")

# Combine the predictions into a single CSV file
combined_predictions = pd.concat([pd.read_csv(f"{regressor_opt}_predictions.csv") for regressor_opt in regressors_list], axis=1)
combined_predictions.to_csv(output_predictions_path, index=False)
print(f"Combined predictions saved to {output_predictions_path}")

