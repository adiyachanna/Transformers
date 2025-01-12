# Transformer Model Training

This repository contains an implementation of a Transformer model for sequence-to-sequence translation tasks using PyTorch. The implementation follows the architecture described in "Attention Is All You Need" paper and includes a complete training pipeline with logging and checkpointing.

## Features

- Full Transformer architecture implementation with encoder-decoder structure
- Training pipeline with validation loop and model checkpointing
- Progress tracking using tqdm
- TensorBoard integration for loss visualization
- Configurable model parameters through config file
- Support for custom tokenizers
- Automatic mixed precision training on CUDA devices

## Requirements

```bash
pytorch>=1.7.0
tqdm
tensorboard
transformers  # for tokenizers
```

## Training Process

The training loop implements the following key features:

1. **Batch Processing**: Uses DataLoader with tqdm progress bar
2. **Forward Pass**:
   - Encodes input sequences
   - Generates decoder outputs
   - Projects to vocabulary size
3. **Loss Calculation**: Uses cross-entropy loss
4. **Optimization**: Performs backpropagation and gradient updates
5. **Validation**: Runs validation after each epoch
6. **Model Checkpointing**: Saves model state after each epoch

## Usage

1. Prepare your configuration:

```python
config = get_config()  # Define your configuration parameters
```

2. Start training:

```python
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
```

## Configuration

The model accepts various configuration parameters including:

- Sequence length (`seq_len`)
- Model dimensions
- Number of encoder/decoder layers
- Learning rate
- Device settings (CPU/CUDA)

## Training Output

The training progress shows:
- Epoch progress
- Current loss
- Examples/second processing rate
- Validation results with source/target/predicted text samples

Example output:
```
Processing epoch 00: 100%|██████████| 1969/1969 [13:15<00:00, 2.48it/s, loss=6.007]
```

## Model Checkpointing

After each epoch, the model state is saved including:
- Epoch number
- Model state dictionary
- Optimizer state
- Global step count

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'global_step': global_step
}, model_filename)
```

## Validation

The training includes a validation phase that:
- Evaluates model performance on a separate validation set
- Provides example translations
- Logs metrics to TensorBoard

## TensorBoard Integration

Training progress can be monitored using TensorBoard:
- Loss curves
- Validation metrics
- Learning rate tracking

## Directory Structure

```
.
├── config/
│   └── config.py        # Configuration parameters
├── models/
│   └── transformer.py   # Model implementation
├── train.py            # Training script
├── utils/
│   ├── dataloader.py   # Data loading utilities
│   └── tokenizer.py    # Tokenization utilities
└── README.md
```

## References

1. Vaswani et al., "Attention Is All You Need" (2017)
   - Original Transformer architecture paper
   - https://arxiv.org/abs/1706.03762

2. "Coding a Transformer from scratch on PyTorch" by Umar Jamil
   - Implementation insights and techniques
   - https://www.youtube.com/watch?v=ISNdQcPhsts

3. Hugging Face Tokenizers
   - Text tokenization libraries
   - https://huggingface.co/docs/tokenizers

4. PyTorch Documentation
   - Framework documentation
   - https://pytorch.org/

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines if applicable]****
