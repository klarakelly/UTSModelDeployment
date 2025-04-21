import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def preprocessingInput(input_dict):
    df = pd.DataFrame([input_dict])

    labelencode = preprocessing.LabelEncoder()
    df['arrival_year'] = labelencode.fit_transform(df['arrival_year'])

    df = pd.get_dummies(df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=True)

    columnOrders = [
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
        'required_car_parking_space', 'lead_time', 'arrival_year', 'arrival_month', 
        'arrival_date', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 
        'avg_price_per_room', 'no_of_special_requests', 'type_of_meal_plan_Meal Plan 1', 'type_of_meal_plan_Meal Plan 2', 
        'type_of_meal_plan_Meal Plan 3','room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4',
        'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7', 'market_segment_type_Aviation', 
        'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline']
    
    df = df.reindex(columns=columnOrders, fill_value=0)

    scaler = RobustScaler()
    dataScaled = scaler.fit_transform(df)
    return dataScaled

def main():
    st.title("Prediction for Hotel Booking Cancellation")

    input = {
        'no_of_adults': st.number_input('Number of Adults', 0, 10, value=0),
        'no_of_children': st.number_input('Number of Children', 0, 5, value=0),
        'no_of_weekend_nights': st.number_input('Weekend Nights', 0, 10, value=0),
        'no_of_week_nights': st.number_input('Week Nights', 0, 15, value=0),
        'type_of_meal_plan': st.selectbox('Meal Plan Type', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
        'required_car_parking_space': st.selectbox('Requires Car Parking? (0 = No, 1 = Yes)', [0, 1], index=0),
        'room_type_reserved': st.selectbox('Room Type Reserved', [f'Room_Type {i}' for i in range(1, 8)], index=0),
        'lead_time': st.number_input('Lead Time (days)', 0, 500, value=0),
        'arrival_year': st.selectbox('Arrival Year', [2017, 2018, 2019], index=0),
        'arrival_month': st.selectbox('Arrival Month', list(range(1, 13)), index=0),
        'arrival_date': st.selectbox('Arrival Date', list(range(1, 32)), index=0),
        'market_segment_type': st.selectbox('Market Segment Type', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'], index=0),
        'repeated_guest': st.selectbox('Is Repeated Guest? (0 = No, 1 = Yes)', [0, 1], index=0),
        'no_of_previous_cancellations': st.number_input('Previous Cancellations', 0, 20, value=0),
        'no_of_previous_bookings_not_canceled': st.number_input('Previous Successful Bookings', 0, 100, value=0),
        'avg_price_per_room': st.number_input('Average Room Price', 0.0, 10000.0, value=0.0),
        'no_of_special_requests': st.number_input('Number of Special Requests', 0, 5, value=0)
    }

    if st.button("Predict!"):
        model = load_model('finalmodelxgb.pkl')
        finalinput = preprocessingInput(input)
        result = model.predict(finalinput)
        output = 'Canceled' if result[0] == 1 else 'Not Canceled'
        st.success(f"Result: {output}")

if __name__ == '__main__':
    main()