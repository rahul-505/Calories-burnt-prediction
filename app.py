import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load datasets
exercise_data = pd.read_csv('exercise.csv')
calories_data = pd.read_csv('calories.csv')

# Merge datasets on User_ID
data = pd.merge(exercise_data, calories_data, on='User_ID')

# Convert categorical 'Gender' column to numerical (Male: 1, Female: 0)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Features (X) and Target (y)
X = data.drop(['User_ID', 'Calories'], axis=1)  # Drop unnecessary columns
y = data['Calories']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Calories Burned Prediction App")
st.write("""
This app predicts the number of calories burned during exercise based on your input.
""")

# Input fields
st.sidebar.header("User Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.slider("Age", 10, 100, 25)
    height = st.sidebar.slider("Height (cm)", 100, 250, 175)
    weight = st.sidebar.slider("Weight (kg)", 30, 200, 70)
    duration = st.sidebar.slider("Duration (minutes)", 1, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate", 50, 200, 100)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 40.0, 37.0)
    
    # Convert gender to numerical value
    gender = 1 if gender == "Male" else 0
    
    # Create a dictionary of input data
    input_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp
    }
    
    return pd.DataFrame(input_data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("User Input")
st.write(input_df)

# Predict calories burned
prediction = model.predict(input_df)

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted Calories Burned: **{round(prediction[0], 2)}**")