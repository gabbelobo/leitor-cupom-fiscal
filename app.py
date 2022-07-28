from fileinput import filename
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import contours

app = Flask(__name__)
cors = CORS(app,  resources={r"/*": {"origins": "*"}})
app.run(debug=True)

@app.route("/receive", methods=['POST'])
def receive():
    file = request.files["file"] 
    filename = file.filename 
    file.save(filename)
    image = cv2.imread(filename)
    img_contours = contours.get_contours(image)
    return jsonify(contours = img_contours.tolist())

@app.route("/csv_file", methods=['POST'])
def csv_file():
    data = request.form['contours']
    data_split = data.split(',')
    final_contours = np.ndarray(shape=(4,2))
    for i in range(4):
        for j in range(2):
            final_contours[i][j] = data_split[i*2 + j]
    # return jsonify(data = final_contours.tolist())
    file = request.files["file"] 
    filename = file.filename 
    file.save(filename)
    image = cv2.imread(filename)
    df = contours.get_csv(image, final_contours)
    return jsonify(df.to_dict())


@app.route("/")
def hello_world():
    return jsonify("hi")