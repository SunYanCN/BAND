from flask import Flask, render_template, request
import requests
from band import utils
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    """
    Home Page.

    URL: /
    POST HTTP Method: Renders page along with Keras Model's output
    GET HTTP Method: Renders page without any computation.
    """
    if request.method == 'POST':
        # Retrive review and get rating from model
        endpoint = "http://127.0.0.1:8501"
        review = request.form["review"]
        processor = utils.load_processor(model_path='saved_model/blstm/1')
        x = list(review)
        tensor = processor.process_x_dataset([x])
        json_data = {"model_name": "default", "data": {"input:0": tensor.tolist()}}
        result = requests.post(endpoint, json=json_data)
        preds = dict(result.json())['dense/Softmax:0']
        label_index = np.array(preds).argmax(-1)
        labels = processor.reverse_numerize_label_sequences(label_index)
        confidence = preds[0][label_index.tolist()[0]]

        # Open results file to save output for analysis.
        with open("results.csv", "a") as f:
            f.write("{},{}\n".format(review, confidence))
        
        # Same IP address and browser information of user
        with open("usage.csv", "a") as f:
            f.write("{}, {}\n".format(request.user_agent.string, request.remote_addr))
        
        # Default message that will overwritten if no error occurs
        result = ["Unexpected Error occured.", "You may have entered a lot of unkonwn words", ""]
        if confidence:
            result = ["Review: {}".format(review[:50]),
                        "\nUser has given a {} type Text".format(labels),
                        "Confidence: {:.3f} %".format(confidence*100),"",""]

        # Rendering page with result
        return render_template('index.html', result=result)
    else :
        # Rendering page without result
        return render_template('index.html')


@app.route('/ner', methods=['GET', 'POST'])
def ner():
    """
    Home Page.

    URL: /
    POST HTTP Method: Renders page along with Keras Model's output
    GET HTTP Method: Renders page without any computation.
    """
    if request.method == 'POST':
        # Retrive review and get rating from model
        endpoint = "http://127.0.0.1:8500"
        review = request.form["review"]
        processor = utils.load_processor(model_path='saved_model/bilstm/1')
        x = list(review)
        tensor = processor.process_x_dataset([x])
        json_data = {"model_name": "default", "data": {"input:0": tensor.tolist()}}
        result = requests.post(endpoint, json=json_data)
        preds = dict(result.json())['activation/truediv:0']
        label_index = np.array(preds).argmax(-1)
        labels = processor.reverse_numerize_label_sequences(label_index)
        labels = labels[0][:len(x)]

        # Open results file to save output for analysis.
        with open("ner_results.csv", "a") as f:
            f.write("{},{}\n".format(review, labels))

        # Same IP address and browser information of user
        with open("ner_usage.csv", "a") as f:
            f.write("{}, {}\n".format(request.user_agent.string, request.remote_addr))

        # Default message that will overwritten if no error occurs
        result = ["Unexpected Error occured.", "You may have entered a lot of unkonwn words", ""]
        if labels:
            result = ["","","",\
                      "Original Text: " + ' '.join(x),"NER Result:    "+' '.join(labels)]

        # Rendering page with result
        return render_template('index.html', result=result)
    else:
        # Rendering page without result
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)