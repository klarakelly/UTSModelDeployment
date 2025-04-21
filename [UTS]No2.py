import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

# DataHandler Class
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def checkNullValue(self):
        if self.data is not None:
            print("Null value in each column:\n", self.data.isnull().sum())

    def encodeTarget(self):
        label_encoder = preprocessing.LabelEncoder()
        self.data['booking_status'] = label_encoder.fit_transform(self.data['booking_status'])
        self.data.drop('Booking_ID', axis=1, inplace=True)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

# XGBModelHandler Class
class XGBModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.scaler = RobustScaler()
        self.createModel()

    def split_data(self, test_size=0.2, random_state=0):
        x_train, x_test, y_train, y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        self.x_train = x_train.copy()
        self.x_test = x_test.copy()
        self.y_train = y_train
        self.y_test = y_test

    def convertToInt(self, column, dtype):
        self.x_train[column] = self.x_train[column].astype(dtype)
        self.x_test[column] = self.x_test[column].astype(dtype)

    def findMode(self, column):
        return self.x_train[column].mode()[0]
    
    def fillMissingValue(self, column, value):
        self.x_train[column].fillna(value, inplace=True)
        self.x_test[column].fillna(value, inplace=True)

    def check_outlier_boxplot(self, column):
        boxplot = self.x_train.boxplot(column=[column])
        plt.show()

    def findMedian(self, column):
        return self.x_train[column].median()
 
    def oneHotEncoding(self, columns):
        self.x_train = pd.get_dummies(self.x_train, columns=columns, drop_first=True)
        self.x_test = pd.get_dummies(self.x_test, columns=columns, drop_first=True)
        self.x_train, self.x_test = self.x_train.align(self.x_test, join='left', axis=1, fill_value=0)

    def labelEncoding(self, column):
        encoder = preprocessing.LabelEncoder()
        self.x_train[column] = encoder.fit_transform(self.x_train[column])
        self.x_test[column] = encoder.transform(self.x_test[column])

    def scaling(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def createModel(self, learning_rate=0.3, max_depth=6):
        self.model = xgb.XGBClassifier(n_estimators=100,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth,
                                       random_state=0)
    def trainModel(self):
        self.model.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test)
    
    def createReport(self):
        print("Accuracy:", accuracy_score(self.y_test, self.y_predict))
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_predict, target_names=['Not_Canceled', 'Canceled']))

    def tuningParameters(self):
        parameters = {'max_depth': [3, 5, 7],
                      'learning_rate': [0.01, 0.1, 0.2]}
        
        model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=0)
        model = GridSearchCV(model, 
                             param_grid=parameters,
                             scoring='accuracy',
                             cv=5)
        model.fit(self.x_train, self.y_train)
        print("Tuned Hyperparameters:", model.best_params_)
        print("Accuracy:", model.best_score_)
        self.createModel(learning_rate=model.best_params_['learning_rate'], max_depth=model.best_params_['max_depth'])
        
    def saveModel(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)


file_path = 'Dataset_B_hotel.csv'

data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.checkNullValue()
data_handler.encodeTarget()
data_handler.create_input_output('booking_status')

xgb_handler = XGBModelHandler(data_handler.input_df, data_handler.output_df)
xgb_handler.split_data()
xgb_handler.convertToInt('required_car_parking_space', 'Int64')
mode_meal = xgb_handler.findMode('type_of_meal_plan')
xgb_handler.fillMissingValue('type_of_meal_plan', mode_meal)
mode_parking = xgb_handler.findMode('required_car_parking_space')
xgb_handler.fillMissingValue('required_car_parking_space', mode_parking)
xgb_handler.check_outlier_boxplot('avg_price_per_room')
median_price = xgb_handler.findMedian('avg_price_per_room')
xgb_handler.fillMissingValue('avg_price_per_room', median_price)
xgb_handler.oneHotEncoding(['market_segment_type', 'type_of_meal_plan', 'room_type_reserved'])
xgb_handler.labelEncoding('arrival_year')
xgb_handler.scaling()

print("Before Tuning Parameter")
xgb_handler.trainModel()
xgb_handler.makePrediction()
xgb_handler.createReport()

print("After Tuning Parameter")
xgb_handler.tuningParameters()
xgb_handler.trainModel()
xgb_handler.makePrediction()
xgb_handler.createReport()

xgb_handler.saveModel('finalmodelxgb.pkl')