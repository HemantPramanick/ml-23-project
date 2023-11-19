from ml_final import Regression
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

