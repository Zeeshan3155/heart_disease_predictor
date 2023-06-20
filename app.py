from flask import Flask, request, jsonify, render_template
from src.pipeline.prediction_pipe import PredictPipeline,CustomData
import json

app = Flask(__name__,static_folder="static")


@app.route("/", methods=['GET'])
def template():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    # Process the form data and generate a prediction
    data = list(map(float,data.values()))
    custom_data = CustomData(*data)
    df = custom_data.create_dataframe()
    predictor = PredictPipeline()
    prediction = predictor.predict(df)
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)