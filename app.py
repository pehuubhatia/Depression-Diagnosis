from flask import Flask, request, render_template
import pandas as pd
import pickle  # Use if you save your model with pickle

app = Flask(__name__)

# Load the trained model (assuming it's saved as 'rf_model.pkl')
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Depression level categories based on encoded values
depression_labels = {
    0: "No depression",
    1: "Mild depression",
    2: "Moderate depression",
    3: "Severe depression",
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from form
        user_input = {
            "gender": request.form["gender"],
            "age": int(request.form["age"]),
            "afftype": request.form["afftype"],
            "melanch": request.form["melanch"],
            "inpatient": request.form["inpatient"],
            "edu": request.form["edu"],
            "marriage": request.form["marriage"],
            "work": request.form["work"]
            # Add all other necessary input fields here
        }

        # Convert input to DataFrame for model prediction
        input_df = pd.DataFrame([user_input])

        # Predict and get label
        prediction_encoded = rf_model.predict(input_df)
        prediction_encoded = round(prediction_encoded[0])  # Round to nearest int
        prediction_label = depression_labels.get(prediction_encoded, "Unknown")

        # Pass prediction to template
        return render_template("index.html", prediction=prediction_label)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
