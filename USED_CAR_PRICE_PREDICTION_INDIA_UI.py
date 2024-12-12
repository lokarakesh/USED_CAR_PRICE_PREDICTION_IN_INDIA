
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Cache the dataset loading
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Cache the trained model
@st.cache_resource
def train_model(X, y):
    categorical_features = ["Make", "Model", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
    numerical_features = ["Year", "Kilometers_Driven", "Mileage(KMPL)", "Engine(CC)", "Power(BHP)", "Seats"]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))]
    )

    model_pipeline.fit(X, y)
    return model_pipeline

# Load the dataset
file_path = "Car_Data_Zscore.csv"  
data = load_data(file_path)

# Features and Target
X = data.drop("Price(Lakhs)", axis=1)
y = data["Price(Lakhs)"]

# Train the model and cache it
model = train_model(X, y)


# Custom CSS for a modern UI
st.markdown(
    """
    <style>
        .stApp {
            background: #f7f7f9;
        }
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
        }
        .section-title {
            color: #007bff;
            font-size: 20px;
            font-weight: bold;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        .predict-button button {
            background-color: #28a745;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.title("🚗 Predict the Price, Drive the Decision. 🏎️")
st.write("Fill in the details below to get the estimated price of the used 🚗.")

# Input fields for car details
make = st.selectbox("Make", options=data["Make"].unique())

# Filter models dynamically based on selected make
filtered_models = data[data["Make"] == make]["Model"].unique()
model_name = st.selectbox("Model", options=filtered_models)

location = st.selectbox("Location", options=data["Location"].unique())
year = st.number_input("Year", min_value=1980, max_value=2023, step=1)
fuel_type = st.selectbox("Fuel Type", options=data["Fuel_Type"].unique())
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
transmission = st.selectbox("Transmission", options=data["Transmission"].unique())
owner_type = st.selectbox("Owner Type", options=data["Owner_Type"].unique())
mileage = st.number_input("Mileage (KMPL)", min_value=0.0, step=0.1)
engine = st.number_input("Engine (CC)", min_value=500.0, step=100.0)
power = st.number_input("Power (BHP)", min_value=20.0, step=5.0)
seats = st.number_input("Seats", min_value=2, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "Make": [make],
        "Model": [model_name],
        "Location": [location],
        "Year": [year],
        "Fuel_Type": [fuel_type],
        "Kilometers_Driven": [kilometers_driven],
        "Transmission": [transmission],
        "Owner_Type": [owner_type],
        "Mileage(KMPL)": [mileage],
        "Engine(CC)": [engine],
        "Power(BHP)": [power],
        "Seats": [seats],
    })
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f} Lakhs")
# Footer
st.markdown("---")
st.write("Made with ❤️ by Rakesh Loka")
st.markdown("""
    <div style="background-color: #f9f9f9; border-left: 6px solid #007bff; padding: 10px; margin: 20px 0;">
        <h3 style="color: #007bff;">💡 Note to Users:</h3>
        <p style="color: #333; font-size: 16px;">
            This tool is designed to provide an estimated price for your used car based on the details you enter. 
            Please ensure that the inputs are as accurate as possible to get the best prediction. 
        </p>
        <ul style="color: #555; font-size: 14px;">
            <li>Use the dropdowns to select the car make, model, and other details.</li>
            <li>Fields like engine size and power will be auto-filled when possible based on the model.</li>
            <li>Prices are indicative and may vary based on additional factors not included in the dataset.</li>
        </ul>
        <p style="color: #333; font-size: 16px;">
            Your feedback is valuable to us. Feel free to share your thoughts to help us improve!
        </p>
    </div>
""", unsafe_allow_html=True)
