from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# --------------------- Load Pickle Files ---------------------

# Heart Model
with open('models/model_heart.pkl', 'rb') as model_file:
    model_heart = pickle.load(model_file)
with open('models/scaler_heart.pkl', 'rb') as scaler_file:
    scaler_heart = pickle.load(scaler_file)
with open('models/encoder_heart.pkl', 'rb') as encoder_file:
    encoder_heart = pickle.load(encoder_file)

# Diabetes Model
with open('models/model_diabetes.pkl', 'rb') as model_file:
    model_diabetes = pickle.load(model_file)
with open('models/scaler_diabetes.pkl', 'rb') as scaler_file:
    scaler_diabetes = pickle.load(scaler_file)

# Brain Model
with open('models/model_brain.pkl', 'rb') as model_file:
    model_brain = pickle.load(model_file)
with open('models/encoder_brain.pkl', 'rb') as encoder_file:
    encoder_brain = pickle.load(encoder_file)

# --------------------- Helper Function ---------------------
def preprocess_data(data, numeric_cols, categorical_cols=None, scaler=None, encoder=None):
    """
    Preprocess the input data:
    - Scale numeric features
    - Encode categorical features (if encoder is provided)
    """
    input_df = pd.DataFrame([data])

    # Scale numeric columns
    if scaler and numeric_cols:
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Encode categorical columns if encoder exists
    if encoder and categorical_cols:
        encoded_categorical = encoder.transform(input_df[categorical_cols])
        encoded_categorical = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))
        return pd.concat([input_df[numeric_cols], encoded_categorical], axis=1)

    return input_df[numeric_cols]

# --------------------- Routes for Pages ---------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/brain')
def brain():
    return render_template('brain.html')

# --------------------- Heart Disease Prediction ---------------------
@app.route('/predict_heart', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        data = {
            "Age": float(request.form["Age"]),
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingBP": float(request.form["RestingBP"]),
            "Cholesterol": float(request.form["Cholesterol"]),
            "FastingBS": float(request.form["FastingBS"]),
            "RestingECG": request.form["RestingECG"],
            "MaxHR": float(request.form["MaxHR"]),
            "ExerciseAngina": request.form["ExerciseAngina"],
            "Oldpeak": float(request.form["Oldpeak"]),
            "ST_Slope": request.form["ST_Slope"],
        }

        # Preprocess input data
        numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
        categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
        input_preprocessed = preprocess_data(data, numeric_cols, categorical_cols, scaler_heart, encoder_heart)

        # Make prediction
        prediction = model_heart.predict(input_preprocessed)
        output = "at risk of a heart attack" if prediction[0] == 1 else "not at risk of a heart attack"

        return render_template("heart.html", prediction_text=f"The person is {output}.")
    except Exception as e:
        return render_template("heart.html", prediction_text=f"An error occurred: {str(e)}")

# --------------------- Diabetes Prediction ---------------------
@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    try:
        # Collect data from the form
        data = {
            "Pregnancies": float(request.form["Pregnancies"]),
            "Glucose": float(request.form["Glucose"]),
            "BloodPressure": float(request.form["BloodPressure"]),
            "SkinThickness": float(request.form["SkinThickness"]),
            "Insulin": float(request.form["Insulin"]),
            "BMI": float(request.form["BMI"]),
            "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
            "Age": float(request.form["Age"]),
        }

        # Preprocess input data
        numeric_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        input_preprocessed = preprocess_data(data, numeric_cols, scaler=scaler_diabetes)

        # Make prediction
        prediction = model_diabetes.predict(input_preprocessed)
        output = "at risk of diabetes" if prediction[0] == 1 else "not at risk of diabetes"

        return render_template("diabetes.html", prediction_text=f"The person is {output}.")
    except Exception as e:
        return render_template("diabetes.html", prediction_text=f"An error occurred: {str(e)}")

# --------------------- Brain Disease Prediction ---------------------
@app.route("/predict_brain", methods=["POST"])
def predict_brain():
    try:
        # Collect data from the form (ensure consistent naming)
        data = {
            "gender": request.form["Gender"],  # Match case and underscore
            "age": float(request.form["Age"]),
            "hypertension": int(request.form["Hypertension"] == 'yes'),  # Convert 'yes'/'no' to 1/0
            "heart_disease": int(request.form["HeartDisease"] == 'yes'),  # Same for 'HeartDisease'
            "ever_married": request.form["EverMarried"],  # Adjusted name
            "work_type": request.form["WorkType"],  # Adjusted name
            "Residence_type": request.form["ResidenceType"],  # Adjusted name
            "avg_glucose_level": float(request.form["GlucoseLevel"]),
            "bmi": float(request.form["BMI"]),
            "smoking_status": request.form["SmokingStatus"],  # Adjusted name
        }

        # Preprocess input data
        numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
        categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        input_preprocessed = preprocess_data(data, numeric_cols, categorical_cols, encoder=encoder_brain)

        # Make prediction
        prediction = model_brain.predict(input_preprocessed)
        output = "at risk of brain disease" if prediction[0] == 1 else "not at risk of brain disease"

        return render_template("brain.html", prediction_text=f"The person is {output}.")
    except Exception as e:
        return render_template("brain.html", prediction_text=f"An error occurred: {str(e)}")


# --------------------- Run Flask App ---------------------
if __name__ == '__main__':
    app.run(debug=True)
