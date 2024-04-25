from flask import request, Flask, render_template
import os
import sys
import numpy as np
import librosa
import requests
from bs4 import BeautifulSoup as bs



app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Continue with file processing
        path = os.getcwd()
        if not os.path.isdir(path+'/audio_files'):
             os.mkdir("audio_files")
        
        file_path = os.path.join(path+'/audio_files', file.filename)
        file.save(file_path)
        
        y, sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = np.argmax(mean_chroma)
        key_scale = chroma_to_key[key]
        
        return render_template('home.html', key=key, key_scale=key_scale)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)