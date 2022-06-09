from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np 
from keras.models import load_model
from tensorflow_addons.optimizers import AdamW

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

image_model = load_model('ml_model/model.h5', custom_objects = {'AdamW' : AdamW})
image_cascade = cv2.CascadeClassifier('ml_model/haarcascade_frontalface_alt2.xml')

def image_predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = image_cascade.detectMultiScale(gray,1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = image[y:y+h, x:x+w]

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'after.jpg'), image)
    try:
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'cropped.jpg'), cropped)

    except:
        pass

    try:
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'cropped.jpg'), 0)

    except:
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'after.jpg'), 0)

    img = cv2.resize(img, (48,48))
    img = img/255

    img = img.reshape(1,48,48,1)

    pred = image_model.predict(img)

    label_emotion =  ['Angry', 'Fear', 'Happy', 'Neutral','Sad','Suprise','disgust']
    pred = np.argmax(pred)
    final_pred = label_emotion[pred]

    return final_pred

@app.route('/predict', methods=['POST'])
def predict():
	if 'file' not in request.files:
		return jsonify({
			'status': 'failed',
			'message': 'no image files'
		})
	else:
		files = request.files.getlist("file")
		prediction = []

		for file in files:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			prediction.append(image_predict(image))

			try:
				os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'after.jpg'))
				os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'cropped.jpg'))
			except:
				pass
		
		return jsonify(prediction)