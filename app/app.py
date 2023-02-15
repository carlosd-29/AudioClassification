from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
#import IPython.display as ipd
import librosa
import librosa.display
import json
import soundfile as sf
from moviepy.editor import VideoFileClip
from dataset import *

app = Flask(__name__)

sample_rate = 44100

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(6,6))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6,6))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(6,6))
        self.conv4 = nn.Conv2d(64,128, kernel_size=(6,6))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(6,6))
        self.conv_drop = nn.Dropout2d()
        #self.batch_norm24 = nn.BatchNorm2d(24)
        #self.batch_norm48 = nn.BatchNorm2d(48)
       # self.batch_norm64 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 200)
        self.fc2 = nn.Linear(200, 20)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

def create_spec(wavFile):
    sample_rate = 44100
    waveform, sample_rate = librosa.load("static/"+wavFile, sr=sample_rate)
    plt.axis('off')
    X, _ = librosa.effects.trim(waveform)
    XS = librosa.feature.melspectrogram(X, sr=sample_rate)
    Xdb = librosa.amplitude_to_db(XS, ref=np.max)
    librosa.display.specshow(Xdb, sr=sample_rate)

    imgFile = wavFile.split('.')[0]+".png"
    plt.savefig("static/"+imgFile, pad_inches=0.0, bbox_inches='tight')
    return imgFile

def run_prediction(imageFile):
    image = Image.open("static/"+imageFile)
    transform=transforms.Compose([transforms.Resize((288,432)),transforms.ToTensor()])

    transformed_image = transform(image)[:3].unsqueeze(0)

    loaded_model = CNNet()
    loaded_model.load_state_dict(torch.load('trained_for_20_76_7.pth'))
    loaded_model.eval()
    pred = loaded_model(transformed_image)
    _, predicted = torch.max(pred.data, 1)

    
    #Read json file with classes
    with open('classes.json') as json_file:
        class_map  = json.load(json_file)

    for key, value in class_map.items():
        if predicted.item() == value:
            print("The predicted value is:", key)
            return key
    return "Class is not supported"

#Handle audios that are longer than 5 seconds
def parseAudio(wavFile):
    waveform, sample_rate = librosa.load("static/"+ wavFile, sr=None)
    segment_dur = 5
    segment_length = sample_rate * segment_dur
    num_sections = int(np.ceil(len(waveform) / segment_length))
    imgFile = ""

    split = []
    split_filename = []
    for i in range(num_sections):
        segment = waveform[ i * segment_length: (i + 1) * segment_length]
        split.append(segment)
    
    for i in range(num_sections):
        recording_name = os.path.basename(wavFile[:-4])
        out_file = f"{recording_name}_{str(i)}.wav"
        split_filename.append(out_file)
        sf.write(os.path.join("static", out_file), split[i], sample_rate)
    predictions = []
    for i in split_filename:
        imgFile = create_spec(wavFile)
        prediction = run_prediction(imgFile)
        predictions.append(prediction)
    final_prediction = max(set(predictions), key=predictions.count)
    return final_prediction, imgFile
        
def video_to_audio(videoFile):
    filename, ext = os.path.splitext(videoFile)
    clip = VideoFileClip("static/"+videoFile)
    fileout = os.path.join("static",f"{filename}.mp3")
    clip.audio.write_audiofile(fileout)

@app.route("/")
def index():
    return render_template('main.html')

@app.route('/classify', methods = ['POST']) 
def classify():
    uploaded_file = request.files['file']
    uploaded_file.save("static/"+uploaded_file.filename)
    wavFile = uploaded_file.filename

    #check if video file
    ext = wavFile.split(".")[1]
    if ext != ".wav" or ext != '.mp3':
        video_to_audio(wavFile)
        wavFile = wavFile.split('.')[0] + '.mp3'

    prediction, imgFile = parseAudio(wavFile)
    return render_template("classify.html",imgFile=imgFile,wavFile=wavFile, prediction=prediction)
            
if __name__ == "__main__":
    app.run(debug=True)