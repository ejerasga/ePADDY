from flask import Flask, render_template, request, redirect, url_for, make_response
import os
import numpy as np
import librosa
import matplotlib
import csv
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import librosa.display
from keras.models import load_model
import json


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['GRAPH_FOLDER'] = 'static/graphs/'
model = load_model('model.h5') 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)

# extract MFCC features from the wav file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    spectrogram = librosa.util.fix_length(spectrogram, size=128, axis=1)
    
    # match the model input (128, 128, 1)
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    return spectrogram

# function to handle model inference
def predict_species(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    r_dominica, t_castaneum, s_oryzae = prediction[0]

    # the predicted species
    predicted_species = ''
    if r_dominica > max(t_castaneum, s_oryzae):
        predicted_species = 'R_dominica'
    elif t_castaneum > max(r_dominica, s_oryzae):
        predicted_species = 'T_castaneum'
    else:
        predicted_species = 'S_oryzae'

    return {
        'r_dominica': round(r_dominica * 100, 1),
        's_oryzae': round(s_oryzae * 100, 1),
        't_castaneum': round(t_castaneum * 100, 1),
        'predicted_species': predicted_species
    }

# generate bar graph for predictions
def generate_graph(prediction, filename):
    species = ['$\it{R\_dominica}$', '$\it{S\_oryzae}$', '$\it{T\_castaneum}$']
    values = [prediction['s_oryzae'], prediction['r_dominica'], prediction['t_castaneum']]
    
    plt.figure(figsize=(5, 4))
    plt.bar(species, values, color=['#1F78B4', '#1F78B4', '#1F78B4'])
    plt.ylabel('Prediction (%)')
    plt.title(f'Prediction for {filename}')
    
    graph_path = f"{filename}_bar.png"
    plt.savefig(os.path.join(app.config['GRAPH_FOLDER'], graph_path))
    plt.close()
    return graph_path

# generate spectrogram
def generate_spectrogram(file_path, filename):
    audio, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(5, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for {filename}')
    
    spectrogram_path = f"{filename}_spectrogram.png"
    plt.savefig(os.path.join(app.config['GRAPH_FOLDER'], spectrogram_path))
    plt.close()
    return spectrogram_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return 'No files uploaded', 400

    files = request.files.getlist('files')
    if not files:
        return 'No selected files', 400

    results = []
    
    for file in files:
        if file and file.filename.endswith('.wav'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Predict the species
            prediction = predict_species(filepath)
            
            # Generate graphs
            graph_url = generate_graph(prediction, file.filename)
            spectrogram_url = generate_spectrogram(filepath, file.filename)
            
            results.append({
                'filename': file.filename,
                'result': prediction,
                'graph_url': graph_url,
                'spectrogram_url': spectrogram_url
            })
    
    return render_template('result.html', results=results)

@app.route('/generate_csv', methods=['POST'])
def generate_csv():
    results = request.form.getlist('results')[0]  # Get results passed from the form
    results = json.loads(results)  # Load results as JSON

    csv_output = [["File Name", "R_dominica", "S_oryzae", "T_castaneum"]]
    
    for item in results:
        csv_output.append([
            item['filename'],
            f"{item['result']['s_oryzae']}%",
            f"{item['result']['r_dominica']}%",
            f"{item['result']['t_castaneum']}%"
        ])
    
    output = '\n'.join(','.join(row) for row in csv_output)
    response = make_response(output)
    response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
