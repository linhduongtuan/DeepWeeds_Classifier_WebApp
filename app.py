from flask import Flask, render_template, request
from models import get_model_fruits, get_model_weeds, get_tensor, get_weed_name, get_fruit_name
import os 
from math import floor

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

model_weeds = get_model_weeds()
model_fruits = get_model_fruits()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')
   

@app.route('/weed_prediction')
def weed_prediction():
    return render_template('weed_prediction.html')

@app.route('/inference_weeds', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        top_probs, top_labels, top_weeds = get_weed_name(image_bytes=image)
        return render_template('inference_Weeds.html', weeds=top_weeds, name_weeds=top_labels, probabilities_weeds=top_probs)

@app.route('/fruit_prediction')
def fruit_prediction():
    return render_template('fruit_prediction.html')

@app.route('/inference_fruits', methods = ['POST'])
def archive():
    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        top_probs, top_labels, top_fruits = get_fruit_name(image_bytes=image)
        return render_template('inference_Fruits.html', fruits=top_fruits, name_fruits=top_labels, probabilities_fruits=top_probs)


if __name__== '__main__':
    print('Loading website')
    print('To open local website, please click on an address at below')
    app.run(debug=True)
 
