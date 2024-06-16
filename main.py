import pandas as pd
from pipeline import PIPELINE
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)


def predict(df):
    ppl = PIPELINE()

    if 'processed_text' not in df:
        return None
    labels = ppl.inference(df['processed_text'].values, is_lemmatize=False)
    df['label'] = labels
    df.to_csv('test_test.csv', index=False)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        df = pd.read_csv(file)

        df_result = predict(df)

        output_file = 'output.csv'
        df.to_csv(output_file, index=False)

        return send_file(output_file, as_attachment=True)


if __name__ == '__main__':
    # TODO Расскомментировать для обычного инференса
    # df = pd.read_csv('gt_test.csv')
    # df_result = predict(df)
    # output_file = 'submission.csv'

    app.run(debug=True, host='0.0.0.0', port=5000)
