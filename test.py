import os
import json

import wave as wav
import sounddevice as microphone

from io import BytesIO
from scipy.io import wavfile
from vosk import SetLogLevel
from vosk import Model, KaldiRecognizer

SetLogLevel(-1)


# Set duration to 5 seconds
duration = 5

# Set Sample Rate to 16000Hz
rate = 16000

# Record audio from the microphone
print('Recording Now...')

_raw = microphone.rec(
    int(duration * rate),
    samplerate=rate,
    channels=1,
    dtype='int16'
)

microphone.wait()

_wav = BytesIO(bytes())
wavfile.write(_wav, 16000, _raw)

print('Finished Recording')


# Load Recongition Software

if not os.path.exists("model"):
    print('Please download a model from https://alphacephei.com/vosk/models')
    exit(1)


model = Model("model")
engine = KaldiRecognizer(model, 16000)
engine.SetWords(True)


raw = wav.open(_wav, 'rb')

while True:
    data = raw.readframes(4000)

    if len(data) == 0:
        break
    if engine.AcceptWaveform(data):
        result = json.loads(engine.Result())
        if len(result['text']) == 0:
            print('No Voice Activity Detected')
        else:
            print(result['text'])
    else:
        pass
