import os
import torch
import numpy as np
import unittest
from util import convert_weight_from_jit
from model.silero_vad_params import silero_8k_params, silero_16k_params


class TestModelSimilarity(unittest.TestCase):
    def setUp(self):
        jit_model = torch.jit.load(os.environ["SILERO_JIT_MODEL_PATH"])

        self._tolerate = 1e-3
        self._src_model_8k, self._src_model_16k = jit_model._model_8k, jit_model._model

        self._model_8k, self._model_16k = convert_weight_from_jit(jit_model)

        self._src_model_8k.eval()
        self._src_model_16k.eval()
        self._model_8k.eval()
        self._model_16k.eval()

    def compare_tensor_similarity(self, t1, t2, tolerance: float):
        np_output1 = t1.numpy()
        np_output2 = t2.numpy()
        diff = np.abs(np_output1 - np_output2)
        are_close = np.all(diff < tolerance)
        self.assertTrue(
            are_close,
            f"The tensor difference is greater than the allowed tolerance of {tolerance}."
        )

    def test_model_8k(self):
        context_size = silero_8k_params["context_size"]
        num_samples = silero_8k_params["window_size"]

        with torch.no_grad():
            x_size = (10, 5 * (context_size + num_samples))
            x = torch.randn(x_size)
            # padding x
            x = torch.nn.functional.pad(x, (context_size, 0), value=0.0)
            frame_size = context_size + num_samples
            x = torch.nn.functional.pad(x, (0, frame_size - ((x.shape[-1] - context_size) % frame_size)), value=0.0)
            # eval models
            outs1 = list()
            outs2 = list()
            state1 = torch.zeros(0)
            state2 = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                frame = x[:, i - context_size:i + num_samples]
                out1 = self._src_model_8k.stft(frame)
                out2 = self._model_8k.stft(frame)
                self.compare_tensor_similarity(out1, out2, 1e-3)

                out1 = self._src_model_8k.encoder(out1)
                out2 = self._model_8k.encoder(out2)
                self.compare_tensor_similarity(out1, out2, 1e-3)

                out1, state1 = self._src_model_8k.decoder(out1, state1)
                out2, state2 = self._model_8k.decoder(out2, state2)
                self.compare_tensor_similarity(out1, out2, 1e-3)
                self.compare_tensor_similarity(state1, state2, 1e-3)
                outs1.append(out1)
                outs2.append(self._model_8k(frame))
            # test SileroVADNet.forward
            outs1 = torch.cat(outs1, dim=2).squeeze(1)
            outs2 = torch.cat(outs2, dim=1)
            self.compare_tensor_similarity(outs1, outs2, 1e-3)

    def test_model_16k(self):
        context_size = silero_16k_params["context_size"]
        num_samples = silero_16k_params["window_size"]

        with torch.no_grad():
            x_size = (10, 5 * (context_size + num_samples))
            x = torch.randn(x_size)
            # padding x
            x = torch.nn.functional.pad(x, (context_size, 0), value=0.0)
            frame_size = context_size + num_samples
            x = torch.nn.functional.pad(x, (0, frame_size - ((x.shape[-1] - context_size) % frame_size)), value=0.0)
            # eval models
            outs1 = list()
            outs2 = list()
            state1 = torch.zeros(0)
            state2 = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                frame = x[:, i - context_size:i + num_samples]
                out1 = self._src_model_16k.stft(frame)
                out2 = self._model_16k.stft(frame)
                self.compare_tensor_similarity(out1, out2, 1e-3)

                out1 = self._src_model_16k.encoder(out1)
                out2 = self._model_16k.encoder(out2)
                self.compare_tensor_similarity(out1, out2, 1e-3)

                out1, state1 = self._src_model_16k.decoder(out1, state1)
                out2, state2 = self._model_16k.decoder(out2, state2)
                self.compare_tensor_similarity(out1, out2, 1e-3)
                self.compare_tensor_similarity(state1, state2, 1e-3)
                outs1.append(out1)
                outs2.append(self._model_16k(frame))
            # test SileroVADNet.forward
            outs1 = torch.cat(outs1, dim=2).squeeze(1)
            outs2 = torch.cat(outs2, dim=1)
            self.compare_tensor_similarity(outs1, outs2, 1e-3)


if __name__ == '__main__':
    unittest.main()
