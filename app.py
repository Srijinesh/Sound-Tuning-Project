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
        path = os.getcwd()
        if os.path.isdir(path+'/audio_files')==False:
             os.mkdir("audio_files")
        file = request.files['file']
        path = path+'/audio_files'
        file_path = os.path.join(path, file.filename)
        file.save(file_path)
        y,sr = librosa.load(file_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        #key = librosa.beat.tempo(y=y, sr=sr)

        mean_chroma = np.mean(chroma, axis=1)
        chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = np.argmax(mean_chroma)
        key_scale = chroma_to_key[key]
        return render_template('index.html',key=key,key_scale = key_scale)


if __name__=='__main__':
    app.run(debug=True)