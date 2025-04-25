silero_stft_16k_params = [
    # STFT
    {"filter_length": 256, "hop_length": 128, "win_length": 256, "window": "hann"}
]

silero_stft_8k_params = [
    # STFT
    {"filter_length": 128, "hop_length": 64, "win_length": 128, "window": "hann"}
]

silero_encoder_16k_params = [
    # Conv1d
    {"in_channels": 129, "out_channels": 128, "kernel_size": 3, "bias": True, "stride": 1, "padding": 1},
    # Conv1d
    {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "bias": True, "stride": 2, "padding": 1},
    # Conv1d
    {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "bias": True, "stride": 2, "padding": 1},
    # Conv1d
    {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "bias": True, "stride": 1, "padding": 1}
]

silero_encoder_8k_params = [
    # Conv1d
    {"in_channels": 65, "out_channels": 128, "kernel_size": 3, "bias": True, "stride": 1, "padding": 1},
    # Conv1d
    {"in_channels": 128, "out_channels": 64, "kernel_size": 3, "bias": True, "stride": 2, "padding": 1},
    # Conv1d
    {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "bias": True, "stride": 2, "padding": 1},
    # Conv1d
    {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "bias": True, "stride": 1, "padding": 1}
]

silero_decoder_16k_params = [
    # LSTM
    {"input_size": 128, "hidden_size": 128, "bias": True},
    # Dropout
    {"p": 0.1},
    # Conv1d
    {"in_channels": 128, "out_channels": 1, "kernel_size": 1, "bias": True}
]

silero_decoder_8k_params = [
    # LSTM
    {"input_size": 128, "hidden_size": 128, "bias": True},
    # Dropout
    {"p": 0.1},
    # Conv1d
    {"in_channels": 128, "out_channels": 1, "kernel_size": 1, "bias": True}
]

silero_16k_params = {
    # "context_size": 64,
    "window_size": 512,
    "stft_params": silero_stft_16k_params,
    "encoder_params": silero_encoder_16k_params,
    "decoder_params": silero_decoder_16k_params
}

silero_8k_params = {
    # "context_size": 32,
    "window_size": 256,
    "stft_params": silero_stft_8k_params,
    "encoder_params": silero_encoder_8k_params,
    "decoder_params": silero_decoder_8k_params
}
