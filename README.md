# Silero-VAD-Dev

[Silero-VAD](https://github.com/snakers4/silero-vad) is a pre-trained, enterprise-grade Voice Activity Detector (VAD) developed by the Silero team. This repository provides an dev(eager) mode implementation of silero-vad, featuring the following improvements and capabilities:

- A self-contained implementation of the Silero VAD architecture within the codebase, independent of external model libraries.
- The implementation includes support for fine-tuning the `encoder` layers along with the decoder, whereas the official repository is limited to fine-tuning only the decoder.
- Validation via comparative analysis with baseline models to ensure numerical consistency in the implementation.
- If you have already successfully finetune your model with official repo, you can seamlessly use the prepared data with this repo's training script.
---

## 1. Basic Useage

### 1.1 Prepare Environment

First, Clone this repository.

```shell
git clone https://github.com/GitEventHandler/silero-vad-dev && cd silero-vad-dev
```

The requirements.txt file for this project does not include PyTorch. You should install PyTorch based on their specific environment requirements (version 2.4 is recommended). Once PyTorch is installed, run the following commands to install the necessary dependencies:
```shell
pip install -r requirements.txt
```

### 1.2 Converting Original Model Weights

Original Silero-VAD model weights are provided in `.jit` formats. To convert them to `.pt` format, use the following command.

```shell
python util/convert.py jit2pt \
    --input '.jit model file path' \
    --output 'output .pt model file path' 
```

### 1.3 Use Tuned Silero-VAD In Your Code

> [!IMPORTANT]
> If your inference produces an unexpected result, check that your audio's sample rate is either 8 kHz or 16 kHz, and ensure you're using the correct model.

You have two ways to use the fine-tuned Silero-VAD:

1. After training, convert the `.pt` model back to `.jit` format with `util/convert.py`. And use the `get_speech_timestamps` function from official silero-vad package to load the jit file. 
* convert from `.pt` to `.jit` 
```shell
python util/convert.py pt2jit \
    [--input_8k '8khz .pt model file path'] \
    [--input_16k '16khz .pt model file path'] \
    --template_jit 'official .jit model file path' \
    --output 'output .jit model file path' 
```
* use model with official silero-vad
```python
from silero_vad.utils_vad import get_speech_timestamps, read_audio

audio = read_audio("AUDIO FILE")
model = torch.jit.load("TORCH JIT FILE")

get_speech_timestamps(audio=audio, model=model, sampling_rate=16000)
```

2. Use `get_speech_timestamps` implemented by this repo in `util.inference`, and you will be able to inference with the `model.SileroVADNet` model.

```python
import torch
from util.inference import get_speech_timestamps, load_audio

audio = load_audio("AUDIO FILE", output_sr=16000)
model = torch.load("TORCH PT FILE")

get_speech_timestamps(audio=audio, model=model)
```

---

## 2. Train

### 2.1 Pre-Process Dataset

You have two options: using the feather format required by the official repository or using a JSONL file.

```shell
python util/dataset.py \
    --input 'feather or jsonl path' \
    --output 'pre-processed feather file path' \
    --thread THREAD \
    --show-tqdm \
    --target-sr [8000 | 16000]
```

### 2.2 Run SFT

```shell
python train.py \
    --train-data-path 'pre-processed train dataset path' \
    --test-data-path 'pre-processed train dataset path' \
    --sampling-rate [16000 | 8000] \
    --output 'output model (.pt) file dir'
    --epochs 10 \
    --learning-rate 0.001 \
    --device cuda:0 \
    --checkpoint-path 'pretraining model weight (.pt)' \
    --batch-size 100 \
    [--freeze-encoder] \
    [--freeze-decoder]
```

