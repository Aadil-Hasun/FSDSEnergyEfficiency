from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            x1=float(request.form.get('X1')),
            x2=float(request.form.get('X2')),
            x3=float(request.form.get('X3')),
            x4=float(request.form.get('X4')),
            x5=float(request.form.get('X5')),
            x6=float(request.form.get('X6')),
            x7=float(request.form.get('X7')),
            x8=float(request.form.get('X8')),
        )
        data_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred_result = predict_pipeline.predict(data_df)

        return render_template('result.html', result=pred_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)