import torch
import torch.nn as nn
from model.stft import STFT


class SileroSTFT(nn.Module):
    def __init__(self, layer_params):
        super(SileroSTFT, self).__init__()
        self.stft = STFT(
            **(layer_params[0])
        )

    def forward(self, x):
        # [batch, 129 | 65, 5]
        stft_result = self.stft(x)
        # [batch, 129 | 65, 1-5]
        return stft_result[:, :, 1:]


class SileroEncoder(nn.Module):
    def __init__(self, layer_params: list):
        super(SileroEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(**(layer_params[0])),
            nn.ReLU(),
            nn.Conv1d(**(layer_params[1])),
            nn.ReLU(),
            nn.Conv1d(**(layer_params[2])),
            nn.ReLU(),
            nn.Conv1d(**(layer_params[3])),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class SileroDecoder(nn.Module):
    def __init__(self, layer_params):
        super(SileroDecoder, self).__init__()
        self.lstm = nn.LSTMCell(**(layer_params[0]))
        self.conv = nn.Sequential(
            nn.Dropout(**(layer_params[1])),
            nn.ReLU(),
            nn.Conv1d(**(layer_params[2])),
            nn.Sigmoid()
        )

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.lstm(x, (state[0], state[1]))
        else:
            h, c = self.lstm(x)
        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.conv(x)
        return x, state


class SileroVADNet(nn.Module):
    def __init__(self, layer_params):
        super(SileroVADNet, self).__init__()
        self.stft = SileroSTFT(layer_params["stft_params"])
        self.encoder = SileroEncoder(layer_params["encoder_params"])
        self.decoder = SileroDecoder(layer_params["decoder_params"])
        self._state = torch.zeros(0)

    def reset(self):
        device = next(self.parameters()).device
        self._state = torch.zeros(0, device=device)

    def forward(self, x):
        out = self.stft(x) if len(x.shape) == 2 else self.stft(x.view(1, x.shape[-1]))
        out = self.encoder(out)
        out, self._state = self.decoder(out, self._state)
        return out.squeeze(1) if len(x.shape) == 2 else out.squeeze(1)[0]
