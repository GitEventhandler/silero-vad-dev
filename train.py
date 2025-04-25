import argparse
import os
import torch
from torch import nn, optim
from model.silero_vad import SileroVADNet
from model.silero_vad_params import silero_16k_params, silero_8k_params
from util.dataset import load_dataset


def progress_bar(percent: float, char_len: int = 10, fill: str = "#", blank: str = "-"):
    filled_len = min(int(char_len * percent), char_len)
    return fill * filled_len + blank * (char_len - filled_len)


def accuracy(pred_y_mat: torch.Tensor, y_mat: torch.Tensor, threshold: float):
    pred_y_binary = (pred_y_mat > threshold).float()
    correct = (pred_y_binary == y_mat).sum().item()
    total = y_mat.numel()
    acc = correct / total
    return acc


def main():
    os.makedirs(args.output, exist_ok=True)

    # Determine model configuration
    model_config = silero_16k_params if args.sampling_rate == 16000 else silero_8k_params

    # Initialize model
    model = SileroVADNet(model_config).to(args.device)

    # Load pre-trained model if specified
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint.state_dict())
        print(f"Loaded pre-trained weights from: {args.checkpoint_path}")
        print('-' * 20)

    # Freeze components after initialization
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad_(False)

    # Define training components
    criterion = nn.BCELoss()
    optimizer = optim.Adam([
        p for p in model.parameters() if p.requires_grad
    ], lr=args.learning_rate)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):

        tot_train_loss = 0.0
        batch_index = 0
        """
        Train Area
        """
        model.train()
        for xs, ys, (loaded, total) in load_dataset(args.train_data_path, args.batch_size):
            xs = [torch.FloatTensor(w).to(args.device) for w in xs]
            ys = [torch.FloatTensor(l).to(args.device) for l in ys]
            # forward model
            model.reset()
            pred_y = list(model(x) for x in xs)

            pred_y_mat = torch.cat(pred_y, dim=1)
            y_mat = torch.cat(ys, dim=1)

            # calculate loss
            loss = criterion(pred_y_mat, y_mat)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_train_loss += loss.item()
            # output_logs
            batch_index += 1
            progress = loaded / total
            print(
                f"Train epoch {epoch} / {args.epochs} [{progress_bar(progress, char_len=20)}] {progress * 100:>4.1f}%\t",
                f"train_loss = {float(loss):.4f}, train_acc = {accuracy(pred_y_mat, y_mat, args.acc_threshold):.4f}"
            )

        """
        Test Area
        """
        with torch.no_grad():
            model.eval()
            loss_list = list()
            acc_list = list()
            for xs, ys, (_, _) in load_dataset(args.test_data_path, args.batch_size):
                xs = [torch.FloatTensor(w).to(args.device) for w in xs]
                ys = [torch.FloatTensor(l).to(args.device) for l in ys]
                # forward model
                model.reset()
                pred_y = list(model(x) for x in xs)

                pred_y_mat = torch.cat(pred_y, dim=1)
                y_mat = torch.cat(ys, dim=1)
                loss = criterion(pred_y_mat, y_mat)
                loss_list.append(float(loss))
                acc_list.append(accuracy(pred_y_mat, y_mat, args.acc_threshold))

            avg = lambda arr: sum(arr) / len(arr)
            print('-' * 20)
            print(f"Test")
            print(f"\tResult: test_loss = {avg(loss_list):.4f}, test_acc = {avg(acc_list):.4f}")
            if float(avg(loss_list)) < best_loss:
                print(f"\tNew best loss, {avg(loss_list):.4f} < {best_loss:.4f} (best_loss)")
                best_loss = avg(loss_list)
                save_path = os.path.join(args.output, f"model_{args.sampling_rate}_epoch_{epoch}.pt")
                torch.save(model, save_path)
                print(f"\tSaving model to {save_path}")
            else:
                print(f"\t{avg(loss_list):.4f} ≥ {best_loss:.4f} (best_loss)")
            print('-' * 20)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    # ArgumentParser object that contains all the arguments needed for VAD training.

    parser = argparse.ArgumentParser(description="Silero VAD Training")

    # Required parameters for data paths
    parser.add_argument("--train-data-path", required=True, help="Path to the training dataset")
    parser.add_argument("--test-data-path", required=True, help="Path to the validation dataset")

    # Hyperparameters and settings related to accuracy threshold
    parser.add_argument(
        "--acc-threshold",
        type=float,
        default=0.5,
        help="Accuracy threshold for validation data"
    )

    # Sampling rate for audio processing
    parser.add_argument(
        "--sampling-rate",
        type=int,
        choices=[8000, 16000],
        required=True,
        help="Sampling rate of the audio data (8000 or 16000 Hz)"
    )

    # Flags to freeze encoder and decoder weights
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights")
    parser.add_argument("--freeze-decoder", action="store_true", help="Freeze decoder weights")

    # Path to pre-trained model weights if provided
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to pre-trained model weights (if any)")

    # Training settings
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    parser.add_argument("--output", required=True, help="Output directory for saving models and logs")
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Learning rate for the optimizer during training"
    )
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument("--device", default="cuda", help="Device to use for training (cpu or cuda)")

    # Print a preview of the parameters
    print('-' * 20)
    print("Parameter Preview:")
    # Parse arguments and print them in an easy-to-read format
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"\t{arg}: {value}")
    print('-' * 20)
    
    main()
