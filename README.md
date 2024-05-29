# Fine-tuning OpenAI's Whisper Model on Noisy Audio Files

This project involves fine-tuning OpenAI's Whisper model on LibriSpeech noisy audio files using multiple GPUs on Northeastern University's Discovery Cluster. The training employs Torch DDP (Distributed Data Parallel) and Torch FSDP (Fully Sharded Data Parallel) for accelerated inference.

## Installation

Install the required packages using the following command:

```bash
pip install -r requirements.txt
