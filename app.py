from flask import request, Flask, render_template
import os
import sys
import numpy as np
import librosa




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['POST'])
def index():
        audio = request.files['file']
        file_path = '/tmp/' + audio.filename
        y,sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y, sr=sr)
        key = np.argmax(chroma)
        return render_template('index.html',key=key)


if __name__=='__main__':
    app.run(debug=True)