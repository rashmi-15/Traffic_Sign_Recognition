from flask import Flask, render_template, request
from PIL import Image

import mlengine
from mlengine import load_model, load_transforms, predict

MODEL_PATH = 'models/model_A99.8469778.pt'

app = Flask(__name__)

model = load_model(MODEL_PATH)
preprocess = load_transforms()

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def get_input():
	if request.method == 'POST':
		img_file = request.files['image']
		image = Image.open(img_file)
		label, prob = predict(model, preprocess, image)
		return render_template('result.html', label=label, prob=prob)
	else:
		return render_template('index.html')


if __name__ == '__main__':
 	app.run(debug=True, port=2222) 
