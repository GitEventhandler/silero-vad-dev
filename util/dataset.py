from model import silero_16k_params

if __name__ == '__main__':
    import sys
    import os
    from model.silero_vad_params import silero_8k_params, silero_16k_params

    sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))

import argparse
import json
import concurrent.futures
import numpy as np
import tqdm
import logging
import os.path
import feather
import librosa
import pandas as pd
from typing import Generator, Tuple, List

logger = logging.getLogger("Pre-Processor")


def _read_jsonl_dataset(jsonl_path: str, tqdm_pbar: tqdm.tqdm = None):
    with open(jsonl_path, 'r') as jsonl_file:
        lines = jsonl_file.readlines()
    if tqdm_pbar is not None:
        tqdm_pbar.total = len(lines)
        tqdm_pbar.refresh()
    result = list()
    for line in lines:
        item = json.loads(line)
        result.append({
            "audio_path": item["audio_path"],
            "timestamps": [{"start": ts["start"], "end": ts["end"]} for ts in item["speech_ts"]]
        })
        if tqdm_pbar is not None:
            tqdm_pbar.update(1)
    return result


def _read_raw_dataset(feather_path: str, tqdm_pbar: tqdm.tqdm = None):
    df = feather.read_dataframe(feather_path)
    result = list()
    if tqdm_pbar is not None:
        tqdm_pbar.total = len(df["audio_path"])
        tqdm_pbar.refresh()

    for audio_path, timestamps in zip(df["audio_path"], df["speech_ts"]):
        result.append({
            "audio_path": audio_path,
            "timestamps": [{"start": ts["start"], "end": ts["end"]} for ts in timestamps]
        })
        if tqdm_pbar is not None:
            tqdm_pbar.update(1)
    return result


def _filter_raw_dataset(
        raw_dataset: list,
        do_detailed_check: bool = False,
        min_duration: float = 1.0,
        tqdm_pbar: tqdm.tqdm = None
):
    if tqdm_pbar is not None:
        tqdm_pbar.total = len(raw_dataset)
        tqdm_pbar.refresh()

    filtered_dataset = list()
    for item in raw_dataset:
        audio_path = item["audio_path"]
        timestamps = item["timestamps"]
        # check if an audio file exists locally
        if not os.path.isfile(audio_path):
            logger.warning(f"{audio_path} skipped: audio file not found")
            continue
        if do_detailed_check:
            duration = librosa.get_duration(filename=audio_path)
            # filter out audio that is too short
            if duration < min_duration:
                logger.warning(f"{audio_path} skipped: audio too short (less than {min_duration} seconds)")
                continue
            # check whether if the provided timestamps are valid to the file
            filtered_ts = list()
            for ts in timestamps:
                if ts["start"] >= ts["end"]:
                    logger.warning(
                        logger.warning(
                            f"{audio_path}, {ts['start']}→{ts['end']} skipped: start before end"
                        )
                    )
                    continue
                if ts["start"] >= duration:
                    logger.warning(
                        logger.warning(
                            f"{audio_path}, {ts['start']}→{ts['end']} skipped: start timestamp overflow"
                        )
                    )
                    continue
                if ts["end"] > duration:
                    logger.warning(
                        logger.warning(
                            f"{audio_path}, {ts['start']}→{ts['end']} corrected: end timestamp overflow"
                        )
                    )
                    ts["end"] = duration
                filtered_ts.append({"start": ts["start"], "end": ts["end"]})
        else:
            filtered_ts = timestamps
        filtered_dataset.append({
            "audio_path": audio_path,
            "timestamps": filtered_ts
        })
        if tqdm_pbar is not None:
            tqdm_pbar.update(1)

    return filtered_dataset


def _process_audio(
        audio_path: str,
        timestamps: list,
        dataset_sr: int,
        window_size: int = 512,
        seq_win_size: int = 100,
        seq_stride: int = 44800,
        padding_value: float = 0.0
):
    wave, _ = librosa.load(audio_path, sr=dataset_sr)
    timestamps_sr = [
        {"start": int(ts["start"] * dataset_sr), "end": int(ts["end"] * dataset_sr)} for ts in timestamps
    ]
    # split audio into several segments
    audio_size = wave.shape[-1]
    seq_size = seq_win_size * window_size
    # slide frame
    winds, labels = list(), list()
    for seq_head_index in range(0, audio_size, seq_stride):
        sub_seqs = wave[seq_head_index: seq_head_index + seq_size]
        if len(sub_seqs) < seq_size:
            sub_seqs = np.pad(
                sub_seqs,
                (0, seq_size - len(sub_seqs)),
                'constant',
                constant_values=(0, padding_value)
            )
        # split into windows
        sub_winds, sub_labels = list(), list()
        for wind_head_index in range(0, seq_size, window_size):
            sub_winds.append(sub_seqs[wind_head_index:wind_head_index + window_size])
            sample_index = seq_head_index + wind_head_index
            label = 0.0
            for ts in timestamps_sr:
                start, end = ts["start"], ts["end"]
                if start <= sample_index <= end:
                    label = 1.0
                    break
            sub_labels.append(label)

        winds.append(sub_winds)
        labels.append(sub_labels)
    return winds, labels


def _get_dataframe(
        raw_dataset: list,
        dataset_sr: int = 16000,
        window_size: int = 512,
        seq_win_size: int = 100,
        seq_stride: int = 44800,
        padding_value: float = 0.0,
        n_thread: int = 4,
        tqdm_pbar: tqdm.tqdm = None
) -> pd.DataFrame:
    df_dict = {"sequences": list(), "labels": list()}
    if tqdm_pbar is not None:
        tqdm_pbar.total = len(raw_dataset)
        tqdm_pbar.refresh()

    def _pp_audio(item):
        audio_path, timestamps = item["audio_path"], item["timestamps"]
        winds, labels = _process_audio(
            audio_path,
            timestamps,
            dataset_sr,
            window_size,
            seq_win_size,
            seq_stride,
            padding_value
        )
        if tqdm_pbar is not None:
            tqdm_pbar.update(1)
        return winds, labels

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_thread) as executor:
        futures = executor.map(_pp_audio, raw_dataset)
        for s, l in futures:
            df_dict["sequences"].extend(s)
            df_dict["labels"].extend(l)

    return pd.DataFrame(df_dict)


def load_dataset(
        processed_feather_path: str,
        batch_size: int = 100
):
    full_dataset = pd.read_feather(processed_feather_path)
    full_dataset_size = full_dataset.shape[0]
    loaded_dataset_length = 0
    for i in range(0, full_dataset_size, batch_size):
        # features and labels are first horizontally concatenated, then vertically concatenated
        batch_size = min(batch_size, full_dataset_size - loaded_dataset_length)
        seqs_data = full_dataset["sequences"].iloc[i: i + batch_size].tolist()
        seq_len = len(seqs_data[0])
        seq_batch = [
            # vertical concatenation of elements on the same sequence
            np.stack([seqs_data[j][i] for j in range(batch_size)], axis=0)
            for i in range(seq_len)
        ]

        labels_data = full_dataset["labels"].iloc[i: i + batch_size].tolist()
        label_len = len(labels_data[0])
        label_mat = np.stack(labels_data, axis=0)
        label_batch = [label_mat[:, i].reshape((batch_size, 1)) for i in range(label_len)]

        if len(seq_batch) != len(label_batch):
            raise ValueError()

        loaded_dataset_length += batch_size
        yield seq_batch, label_batch, (loaded_dataset_length, full_dataset_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data using specified parameters")
    parser.add_argument('--input', type=str, required=True, help="Input feather file path")
    parser.add_argument('--thread', type=int, default=4, help="Number of threads to use for pre-process")
    parser.add_argument('--output', type=str, required=True, help="Output feather file path")
    parser.add_argument('--show-tqdm', action='store_true', help="Show tqdm progress bar")
    parser.add_argument('--strict-filter', action='store_true', help="Use more strict filter")
    parser.add_argument('--target-sr', type=int, choices=[8000, 16000], default=16000)
    parser.add_argument('--stride', type=int, default=44800)
    parser.add_argument('--seq-win-len', type=int, default=100)
    parser.add_argument('--padding-value', type=float, default=0.0)
    args = parser.parse_args()

    # dataset split info
    if args.target_sr == 8000:
        window_size = silero_8k_params["window_size"]
    else:
        window_size = silero_16k_params["window_size"]

    # load dataset
    loading_pbar = tqdm.tqdm(desc="Loading Dataset: ") if args.show_tqdm else None
    if args.input.endswith(".jsonl"):
        raw_set = _read_jsonl_dataset(args.input, loading_pbar)
    elif args.input.endswith(".feather"):
        raw_set = _read_raw_dataset(args.input, loading_pbar)
    else:
        raise ValueError()

    # filter dataset
    filter_pbar = tqdm.tqdm(desc="Filtering Dataset: ") if args.show_tqdm else None
    filtered_raw_dataset = _filter_raw_dataset(raw_set, args.strict_filter, tqdm_pbar=filter_pbar)

    # convert dataset
    process_pbar = tqdm.tqdm(desc="Processing Raw Dataset: ") if args.show_tqdm else None
    df_dataset = _get_dataframe(
        filtered_raw_dataset,
        dataset_sr=args.target_sr,
        window_size=window_size,
        seq_win_size=args.seq_win_len,
        seq_stride=args.stride,
        n_thread=args.thread,
        padding_value=args.padding_value,
        tqdm_pbar=process_pbar
    )
    df_dataset.to_feather(args.output)
