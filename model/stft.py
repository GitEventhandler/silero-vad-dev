import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center


class STFT(torch.nn.Module):
    def __init__(
            self,
            filter_length=1024,
            hop_length=512,
            win_length=None,
            window='hann'
    ):
        """
        This code is a simplified version of the STFT implementation from the torch-stft repository.
        (https://github.com/pseeth/torch-stft/blob/master/torch_st_ft/stft.py)
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        assert (filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())

    def forward(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)

        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect'
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        return magnitude
