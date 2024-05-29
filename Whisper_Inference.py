import os

import pandas as pd

import torch

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler, Dataset

import torchaudio

from transformers import WhisperForConditionalGeneration, WhisperProcessor

import jiwer

from evaluate import load

import time

 

def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'

    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def cleanup():

    dist.destroy_process_group()


def load_data(rank, world_size):

    dataset = torchaudio.datasets.LIBRISPEECH("", url="dev-clean", download=True)

    data = []

    for i, (waveform, sample_rate, label, speaker_id, chapter_id, utterance_id) in enumerate(dataset):

        data.append({

            'waveform': waveform,

            'sample_rate': sample_rate,

            'label': label,

            'speaker_id': speaker_id,

            'chapter_id': chapter_id,

            'utterance_id': utterance_id

        })

    df = pd.DataFrame(data)

    length = []

    for index, row in df.iterrows():

        waveform = row['waveform'].squeeze(0)

        length.append(waveform.shape[0])

    df['length'] = length

    max_length = 96000

    sub_df = df[df.length<=96000].copy()

    sub_df['waveform'] = sub_df['waveform'].apply(lambda x: torch.nn.functional.pad(x, (0,max_length - x.size(-1)), mode='constant', value = 0))

    class SubDataset(Dataset):

        def __init__(self, df):

            self.waveforms = [x.clone().detach() for x in df['waveform'].tolist()]

            self.labels = df['label'].tolist()

 

        def __len__(self):

            return len(self.waveforms)

 

        def __getitem__(self, idx):

            return self.waveforms[idx], self.labels[idx]

    sub_dataset = SubDataset(sub_df)

    sampler = DistributedSampler(sub_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader = DataLoader(sub_dataset, batch_size=4, sampler=sampler)  

    return loader

 

def train(rank, world_size, epochs):

    print(f"Running basic DDP example on rank {rank}.")

    setup(rank, world_size)

    train_loader = load_data(rank, world_size)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    model = model.to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    wer = load("wer")

    total_wer = 0.0

    total_batches = 0

   

    start_time = time.time()

   

    for batch in train_loader:

        waveforms, labels = batch

        transcripts = []

        for waveform in waveforms:

            input_features = processor(waveform.squeeze(0).squeeze(0), sampling_rate=16000, return_tensors="pt").input_features

            input_features = input_features.to(rank)

            output_ids = ddp_model.module.generate(input_features, max_length=512, num_beams=4, early_stopping=True,

                                                   return_dict_in_generate=True, output_scores=True, output_hidden_states=True, language='en')

            transcript = processor.batch_decode(output_ids.sequences, skip_special_tokens=True)

            transcripts.extend(transcript)

        wer_batch = wer.compute(predictions=transcripts, references=[label.lower() for label in labels])

        total_wer += wer_batch

        total_batches += 1

    avg_wer = total_wer / total_batches if total_batches > 0 else 0.0

   

    print(f"Total WER on GPU {rank}: {avg_wer}")

    elapsed_time = time.time() - start_time

   

    print(f"Completed on GPU {rank} in {elapsed_time:.2f} seconds.")

    cleanup()


    
def main():

    world_size = torch.cuda.device_count()

    print(f"World size is {world_size}")

    epochs = 1  # Reduced for quicker demonstration

    torch.multiprocessing.spawn(train,

                                args=(world_size, epochs),

                                nprocs=world_size,

                                join=True)


if __name__ == "__main__":

    main()