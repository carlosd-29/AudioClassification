import argparse
import random
import pandas as pd
import IPython.display as ipd
import librosa
import librosa.display
from dataset import *
#from train import *

parser = argparse.ArgumentParser(description = "Run inference on trained model")

parser.add_argument('-l', '--list', help= 'Gives list of files that can be used to test the model', action='store_true')
parser.add_argument('-c', '--category', help='Specify which class you want to test', default='airplane')
parser.add_argument('-f', '--filename', help='Random file to be used to test model')


data_path   = 'ESC-50-Master/audio/'
meta_df     = pd.read_csv('ESC-50-Master/meta/esc50.csv')
wav_file    = ""
category    = ""
sample_rate = 44100
args        = parser.parse_args()

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(6,6))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6,6))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(6,6))
        self.conv4 = nn.Conv2d(64,128, kernel_size=(6,6))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(6,6))
        self.conv_drop = nn.Dropout2d()
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


if args.filename:
    wav_file = args.filename

if args.category:
    category = args.category

if args.list:
    with open("classes.txt") as fp:
        wav_options = fp.read().splitlines()

    category = random.choice(wav_options)
    
filtered     = meta_df[meta_df["category"] == category]
random_index = random.randint(0, len(filtered)-1)
wav_file     = filtered.values.tolist()[random_index][0]

print("You've gotten class: {} and file: {}".format(category, wav_file))
waveform, sample_rate = librosa.load(data_path+wav_file, sr=sample_rate)
#ipd.Audio(waveform, rate=sample_rate) can run in jupyter notebook only

print("Here is the generated Spectrogram: {}".format(category))

plt.axis('off')
X, _ = librosa.effects.trim(waveform)
XS = librosa.feature.melspectrogram(X, sr=sample_rate)
Xdb = librosa.amplitude_to_db(XS, ref=np.max)
librosa.display.specshow(Xdb, sr=sample_rate)
plt.savefig("testing_wav.png", pad_inches=0.0, bbox_inches='tight')
image = Image.open("testing_wav.png")
transform=transforms.Compose([transforms.Resize((288,432)),transforms.ToTensor()])

transformed_image = transform(image)[:3].unsqueeze(0)

loaded_model = CNNet()
loaded_model.load_state_dict(torch.load('trained_for_20_76_7.pth'))
loaded_model.eval()
print("Size of transformed image", transformed_image.size())
pred = loaded_model(transformed_image)
_, predicted = torch.max(pred.data, 1)
print(predicted)
#print(class_map)
matched = False
for key, value in class_map.items():
    if predicted.item() == value:
        print("The predicted value is:", key)
        matched = True
        break
if not matched:
    print("The predicted value does not match")


