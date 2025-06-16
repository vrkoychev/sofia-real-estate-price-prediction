from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask import render_template

app = Flask(__name__)
model = joblib.load("best_real_estate_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            'location': request.form['location'],
            'property_type': request.form['property_type'],
            'construction_type': request.form['construction_type'],
            'floor_type': request.form['floor_type'],
            'square_meters': float(request.form['square_meters']),
            'construction_year': int(request.form['construction_year']),
            'floor_number': int(request.form['floor_number']),
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        return jsonify({"predicted_price": round(prediction, 0)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
