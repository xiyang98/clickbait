import flask

import clickbait_test_real
import subprocess

app = flask.Flask(__name__)

@app.route("/")
def hello_route():
    return flask.send_from_directory("templates/", "clickbaitHome.html")


@app.route("/api/v1/classify", methods=["POST"])
def classify_api():
    request = flask.request.get_json(silent=True)
    if isinstance(request, str):
        result = subprocess.check_output(['python', 'clickbait_test_real.py', '-u=' + request])
        prediction = result.decode('ascii')
        print("app",prediction)
        response = flask.jsonify(predictions=prediction)
        return response
        
    else:
        response = {
            "error": "Bad input"
        }

    #return prediction from backend
    return flask.jsonify(request)
