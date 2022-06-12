from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import jwt
import cv2
import numpy as np 
from keras.models import load_model
# Import libraries
import pandas as pd

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files.
import librosa
import librosa.display

from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

load_dotenv()

def image_predict(image_model, image_cascade, filename, image):
    faces = image_cascade.detectMultiScale(image,1.1, 3)

    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

        cropped = image[y:y+h, x:x+w]

    cv2.imwrite(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename + '_after.jpg'), image)
    try:
        cv2.imwrite(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename + '_cropped.jpg'), cropped)

    except:
        pass

    try:
        img = cv2.imread(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename + '_cropped.jpg'), 0)

    except:
        img = cv2.imread(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename + '_after.jpg'), 0)

    img = cv2.resize(img, (392,384), 3)

    img = img/255

    img = img.reshape(1,224,224,3)

    pred = image_model.predict(img)

    label_emotion =  ['Angry', 'Fear', 'Happy', 'Neutral','Sad','Suprise','disgust']

    pred = np.argmax(pred)

    final_pred = label_emotion[pred]

    return final_pred

# augmentation audio data
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# extraction audio data


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result

# Main function (get extracted audio data)


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

def audio_predict(audio_model, filename):
    # As this is a multiclass classification problem onehotencoding our Y.
    encoder = OneHotEncoder()

    result = get_features(filename)  # path predict
    result = pd.DataFrame(result)

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    result = scaler.fit_transform(result)

    # making our data compatible to model.
    result = np.expand_dims(result, axis=2)

    # Predicting audio data
    pred_test = audio_model.predict(result)
    pred_test = np.argmax(pred_test[0])

    label_emotion = ['Angry', 'Fear', 'Happy',
                    'Neutral', 'Sad', 'Suprise', 'disgust']
    final_pred = label_emotion[pred_test]
    return final_pred

    # pred_test = encoder.fit_transform(
    #     np.array(pred_test).reshape(-1, 1)).toarray()
    # y_pred = encoder.inverse_transform(pred_test)

    # print(y_pred)


@app.route('/api/images/predict', methods=['POST'])
def predictImage():
    authHeader = request.headers.get('Authorization')

    try:
        token = authHeader.split(' ')[1]
    except:
        return jsonify({'Status': 'Forbidden', 'message': 'No Credentials'}), 401

    try:
        data = jwt.decode(token, os.environ.get('ACCESS_TOKEN_SECRET'), algorithms="HS256")
    except:
        return jsonify({
            'status': 'Forbidden',
            'message': 'You don\'t have permission'
        }), 403

    if 'file' not in request.files:
        return jsonify({
            'status': 'failed',
            'message': 'no image files'
        })
    else:
        files = request.files.getlist("file")
        prediction = []
        image_model = load_model('ml_model/image_model.h5')
        image_cascade = cv2.CascadeClassifier('ml_model/haarcascade_frontalface_alt2.xml')

        for file in files:
            filename = data.get('username') + '_' + secure_filename(file.filename)
            file.save(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename))

            image = cv2.imread(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename))

            prediction.append(image_predict(image_model,image_cascade, filename, image))

            try:
                os.remove(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename))
                os.remove(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename + '_after.jpg'))
                os.remove(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename + '_cropped.jpg'))
            except:
                pass
        
        return jsonify({'response': prediction})

@app.route('/api/audio/predict', methods=['POST'])
def predictAudio():
    authHeader = request.headers.get('Authorization')
    
    try:
        token = authHeader.split(' ')[1]
    except:
        return jsonify({'Status': 'Forbidden', 'message': 'No Credentials'}), 401
        
    try:
        data = jwt.decode(token, os.environ.get('ACCESS_TOKEN_SECRET'), algorithms="HS256")
    except:
        return jsonify({
            'status': 'Forbidden',
            'message': 'You don\'t have permission'
        }), 403

    if 'file' not in request.files:
        return jsonify({
            'status': 'failed',
            'message': 'no audio files'
        })
    else:
        file = request.files['file']
        filename = data.get('username') + '_' + secure_filename(file.filename)
        file.save(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename))
        audio_model = load_model('ml_model/audio_model.h5')
        prediction = audio_predict(audio_model, os.path.join(os.environ.get('UPLOAD_FOLDER'), filename))
        try:
            os.remove(os.path.join(os.environ.get('UPLOAD_FOLDER'), filename))
        except:
            pass
        return jsonify({'response': prediction})
