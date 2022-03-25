import wave as wav
import sounddevice as microphone

from vosk import Model, KaldiRecognizer


# Set duration to 5 seconds
duration = 5

# Set Sample Rate to 16000Hz
rate = 16000

# Record audio from the microphone
print('Recording Now...')

_raw = microphone.record(
    int(duration * rate),
    samplerate=rate,
    channels=1,
    dtype='int16'
)

raw = wav.open(_wav, 'rb')

print('Finished Recording')


# Load Recongition Software

if not os.path.exists("model"):
    return print('Please download a model from https://alphacephei.com/vosk/models')

model = Model("model")
engine = KaldiRecognizer(model, 16000)
engine.setWords(True)

# Read the File and print results

while True:
    data = raw.readframes(4000)

    if len(data) == 0:
        break
    if engine.AcceptWaveForm(data):
        print(engine.result())
    else:
        print(engine.PartialResult())

print(engine.FinalResult())
