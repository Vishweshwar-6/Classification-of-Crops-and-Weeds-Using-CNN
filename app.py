from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


import os, sys, glob, re

app = Flask(__name__)

model_path = "try_one.h5"



classes = {0:"Crops:-{ About Crops }",1:"weeds:-{ about weeds} "}

def model_predict(image_path):
    print("Predicted")
    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    model = load_model(model_path)
    result = np.argmax(model.predict(image))
    prediction = classes[result]
    
    
    if result == 0:
        print("crop.html")
        
        return "Crops","crops.html"
    elif result == 1:
        print("weeds.html")
        
        return "weeds", "weeds.html"    
    


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    if request.method == 'POST':
        print("Entered here")
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = model_predict(file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    


if __name__ == '__main__':
    app.run(debug=True,threaded=False)
    
