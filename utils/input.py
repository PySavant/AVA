import os
import json
import time
import queue
import threading

import wave as wav
import numpy as np
import sounddevice as microphone

from io import BytesIO
from scipy.io import wavfile
from datetime import datetime

from vosk import SetLogLevel
from vosk import Model, KaldiRecognizer

from logger import getLogger
from timer import timed, FunctionTimer

SetLogLevel(-1)

mode = "production"

class InputManager():
    ''' Insert Description Here '''

    def __init__(self):
        self.duration = 4
        self._exit = False
        self.thread = None
        self.queue = queue.Queue()
        self.logger = getLogger('TRACE', filepath='input')

        microphone.default.channels = 1
        microphone.default.dtype = 'int16'
        microphone.default.samplerate = 16000

    @timed
    def load_engine(self):
        if not os.path.exists(f"data/models/{mode}"):
            self.logger.critical("input", "Please Download a model from https://alphacephei.com/vosk/mdoels")
            exit(1)

        model = Model(f"data/models/{mode}")

        self.engine = KaldiRecognizer(model, 16000)
        self.engine.SetWords(True)

    def _worker(self):
        while not self._exit:
            try:
                _wav = self.queue.get(block=True, timeout=10)
            except queue.Empty:
                self.logger.warn('input', 'Microphone Stream Paused. Closing Background Thread...')
                return
            with FunctionTimer() as timer:
                raw = wav.open(_wav['data'], 'rb')

                while True:
                    _data = raw.readframes(4000)

                    if len(_data) == 0:
                        self.logger.trace('input', 'No Data Found in Chunk')
                        break

                    if self.engine.AcceptWaveform(_data):
                        result = json.loads(self.engine.Result())

                        if len(result['text']) == 0 or result['text'] == 'the':
                            self.logger.trace('input', 'No Voice Activity Detected')
                            break

                        else:
                            self.logger.info('result', f'Speech Recognized: {result["text"]}')

                            break
                    else:
                        pass

            self.logger.debug('input', f'Data Processed - Elapsed Time: {(datetime.now()-_wav["timestamp"]).total_seconds()}')



    def start_worker(self):
        if self.thread or self.queue.empty():
            return

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.logger.debug('input', 'Background Thread Established. Awaiting Data Stream...')
        self.thread.start()

    def _queueStream(self, indata: np.ndarray, frames: int, time, status: microphone.CallbackFlags):
        _wav = BytesIO(bytes())
        wavfile.write(_wav, 16000, indata)

        data = {
            'data': _wav,
            'timestamp': datetime.now()
        }

        self.queue.put(data)

        if not self.thread:
            self.start_worker()

    def run(self):
        with microphone.InputStream(callback=self._queueStream, blocksize=80000):
            try:
                self.logger.info('input', 'Microphone Connected. Now Streaming Audio Data...')
                input('(Press Any Key to Stop)')

                self.logger.info('input', 'Shutting Down...')
            except KeyboardInterrupt:
                self.logger.critical('input', 'Interpreter Override Detected. Shutting Down...')
                self._exit = True
                self.thread.join()

        return

    # Interface Controls

    def start(self):
        self.logger.debug('input', f'Loading AI Speech Recognition Models - Mode: {mode}')
        self.load_engine()

        self.logger.debug('input', 'Setting Up Background Threads...')
        self.start_worker()

        self.logger.debug('input', 'Connecting To Microphone...')
        self.run()

if __name__ == '__main__':
    manager = InputManager()
    manager.start()
