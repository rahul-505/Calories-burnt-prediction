# Import necessary libraries
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

# Function to predict calories burned
def predict_calories(gender, age, height, weight, duration, heart_rate, body_temp):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp]
    })
    # Predict calories burned
    predicted_calories = model.predict(input_data)
    return predicted_calories[0]

# User-friendly interface
print("Welcome to the Calories Burned Prediction Program!")
print("Please provide the following details:")

# Take input from the user
gender = input("Enter your gender (Male/Female): ").strip().capitalize()
age = int(input("Enter your age: "))
height = float(input("Enter your height (in cm): "))
weight = float(input("Enter your weight (in kg): "))
duration = float(input("Enter the duration of exercise (in minutes): "))
heart_rate = float(input("Enter your heart rate during exercise: "))
body_temp = float(input("Enter your body temperature during exercise (in °C): "))

# Convert gender to numerical value
gender = 1 if gender == 'Male' else 0

# Predict calories burned
calories_burned = predict_calories(gender, age, height, weight, duration, heart_rate, body_temp)

# Display the result
print("\nPredicted Calories Burned:", round(calories_burned, 2))