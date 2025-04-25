import unittest
import torch
import os
from util.inference import get_speech_timestamps, load_audio
from util.convert import convert_weight_from_jit

model_path = None


class TestEval(unittest.TestCase):
    def setUp(self):
        jit_model = torch.jit.load(os.environ["SILERO_JIT_MODEL_PATH"])
        self._audio_8k = load_audio(
            os.path.join(
                os.path.split(__file__)[0],
                "test_audio.wav"
            ),
            output_sr=8000
        )
        self._audio_16k = load_audio(
            os.environ["TEST_AUDIO_PATH"],
            output_sr=16000
        )
        self._model_8k, self._model_16k = convert_weight_from_jit(jit_model)
        self._model_8k.eval()
        self._model_16k.eval()

    def test_model_eval(self):
        vad_8k_result = get_speech_timestamps(
            self._audio_8k, self._model_8k, sampling_rate=8000
        )
        vad_16k_result = get_speech_timestamps(
            self._audio_16k, self._model_16k, sampling_rate=16000
        )
        self.assertTrue(
            len(vad_8k_result) > 0,
            "The VAD (8k) detect nothing in the test file."
        )
        self.assertTrue(
            len(vad_16k_result) > 0,
            "The VAD (16k) detect nothing in the test file."
        )


if __name__ == '__main__':
    unittest.main()
