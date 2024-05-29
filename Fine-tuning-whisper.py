import os

import time

import torch

import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim

import torchaudio

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluate import load

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

class SubDataset(Dataset):

    def __init__(self, data_path):

        self.data = torch.load(data_path)

        self.waveforms = [x['waveform'] for x in self.data]

        self.labels = [x['label'] for x in self.data]

    def __len__(self):

        return len(self.waveforms)

    def __getitem__(self, idx):

        return self.waveforms[idx], self.labels[idx]


def preprocess_entry(entry, max_length):

    waveform, sample_rate, label = entry

    waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.size(-1)), mode='constant', value=0)

    return {'waveform': waveform.cpu(), 'label': label}

def preprocess_data(dataset_path, output_path):

    dataset = torchaudio.datasets.LIBRISPEECH(dataset_path, url="dev-clean", download=True)

    max_length = 96000  # Define maximum length

    print("Starting parallel preprocessing using joblib...")

    data = Parallel(n_jobs=10, verbose=10)(delayed(preprocess_entry)((waveform, sample_rate, label), max_length) for waveform, sample_rate, label, _, _, _ in dataset)

    torch.save(data, output_path)

    print("Data preprocessing and saving completed.")

    return output_path

def load_data(data_path):

    if not os.path.exists(data_path):

        print("Preprocessing and saving dataset...")

        data_path = preprocess_data("", data_path)

    else:

        print("Loading preprocessed data...")

    return SubDataset(data_path)

def train(rank, world_size, train_dataset, whisper_model, whisper_processor, wer_metric, epochs=10):

    if world_size > 1:

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        print(f"Process group for DDP initialized on GPU {rank}.")

    device = torch.device(f"cuda:{rank}" if world_size > 1 else "cuda")

    torch.cuda.set_device(device)

    print(f"Training is set to run on {device}.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler if world_size > 1 else None)

    whisper_model.to(device)

    ddp_model = DDP(whisper_model, device_ids=[rank]) if world_size > 1 else whisper_model

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-5)

    loss_fn = torch.nn.CrossEntropyLoss()

    mean_losses = []

    mean_wers = []

    start_time = time.time()

    for epoch in range(epochs):

        total_loss = torch.tensor(0.0, device=device)

        total_wer = torch.tensor(0.0, device=device)

        num_batches = 0

        for waveforms, labels in train_loader:

            optimizer.zero_grad()

            waveforms = waveforms.to(device)

            input_features_list = []

            for waveform in waveforms:

                input_features = whisper_processor(waveform.squeeze(0).to('cpu'), sampling_rate=16000, return_tensors="pt").input_features

                input_features = input_features.to(device)

                input_features_list.append(input_features)

            labels_encoded = [whisper_processor.tokenizer.encode(label.lower()) for label in labels]

            labels_encoded = pad_sequence([torch.tensor(le, dtype=torch.long) for le in labels_encoded], batch_first=True, padding_value=whisper_processor.tokenizer.pad_token_id).to(device)

        

            transcripts = []

            for input_features, label_encoded in zip(input_features_list, labels_encoded):

                outputs = ddp_model(input_features, labels=label_encoded.unsqueeze(0))

                loss = loss_fn(outputs.logits.permute(0, 2, 1), label_encoded.unsqueeze(0))

                loss.backward()

                optimizer.step()

                output_ids = outputs.logits.argmax(-1)

                transcript = whisper_processor.batch_decode(output_ids, skip_special_tokens=True)

                transcripts.extend(transcript)

            whisper_wer = wer_metric.compute(predictions=transcripts, references=[label.lower() for label in labels])

            total_loss += loss.detach()

            total_wer += torch.tensor(whisper_wer, device=device)

            num_batches += 1

        if world_size > 1:

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

            dist.all_reduce(total_wer, op=dist.ReduceOp.SUM)

            num_batches *= world_size

        mean_loss = total_loss / num_batches

        mean_wer = total_wer / num_batches

        if rank == 0:

            print(f"Epoch {epoch+1}, Mean WER: {mean_wer.item()}, Mean Loss: {mean_loss.item()}")

            mean_losses.append(mean_loss.item())

            mean_wers.append(mean_wer.item())

    elapsed_time = time.time() - start_time

    if rank == 0:

        print(f"Training completed in {elapsed_time:.2f} seconds.")

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)

        plt.plot(range(1, epochs + 1), mean_losses, marker='o')

        plt.title("Mean Loss vs Epochs")

        plt.xlabel("Epoch")

        plt.ylabel("Mean Loss")

        plt.subplot(1, 2, 2)

        plt.plot(range(1, epochs + 1), mean_wers, marker='o')

        plt.title("Mean WER vs Epochs")

        plt.xlabel("Epoch")

        plt.ylabel("Mean WER")

        plt.tight_layout()

        plt.show()

    if world_size > 1:

        dist.destroy_process_group()

        print(f"Destroyed process group for GPU {rank}.")


def main():

    world_size = torch.cuda.device_count()

    print(f"Number of GPUs available: {world_size}")

    if world_size > 1:

        os.environ['MASTER_ADDR'] = 'localhost'

        os.environ['MASTER_PORT'] = '12355'

    data_path = 'preprocessed_librispeech.pt'

    train_dataset = load_data(data_path)

    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    wer_metric = load("wer")

    epochs = 5  # Set epochs to 10

    if world_size > 1:

        torch.multiprocessing.spawn(train, args=(world_size, train_dataset, whisper_model, whisper_processor, wer_metric, epochs), nprocs=world_size, join=True)

    else:

        train(0, world_size, train_dataset, whisper_model, whisper_processor, wer_metric, epochs)


if __name__ == "__main__":

    main()