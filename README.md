# Transformer Model Training 

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Training Process](#training-process)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Training Output](#training-output)
- [Model Checkpointing](#model-checkpointing)
- [Validation](#validation)
- [TensorBoard Integration](#tensorboard-integration)
- [Directory Structure](#directory-structure)
- [Code Examples](#code-examples)
- [References](#references)
- [License](#license)
- [Contributing](#contributing)

## Features

✨ Key features of this implementation:
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

The training loop implements:

1. **Batch Processing**: Uses DataLoader with tqdm progress bar
2. **Forward Pass**:
   - Encodes input sequences
   - Generates decoder outputs
   - Projects to vocabulary size
3. **Loss Calculation**: Uses cross-entropy loss
4. **Optimization**: Performs backpropagation and gradient updates
5. **Validation**: Runs validation after each epoch
6. **Model Checkpointing**: Saves model state after each epoch

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-training.git

# Install dependencies
pip install -r requirements.txt

# Verify CUDA installation (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

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

Sample configuration file:

```python
def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 10,
        'lr': 10**-4,
        'seq_len': 350,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'dropout': 0.1
    }
```

## Model Architecture

The Transformer model consists of:
- Encoder stack with self-attention and feed-forward layers
- Decoder stack with masked self-attention and encoder-decoder attention
- Position-wise feed-forward networks
- Multi-head attention mechanism
- Layer normalization and residual connections

## Training Output

Example training progress:

```
Processing epoch 00: 100%|██████████| 1969/1969 [13:15<00:00, 2.48it/s, loss=6.007]

--------------------------------------------------------------------------------
SOURCE: 'What are you to do with such people?'
TARGET: 'Что прикажете с этим народом делать?'
PREDICTED: 'Что ты ?'
```

## Model Checkpointing

Checkpoint saving code:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'global_step': global_step
}, model_filename)
```

Loading checkpoints:

```python
checkpoint = torch.load(model_filename)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

## Validation

Validation process implementation:

```python
def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, 
                  device, print_msg, global_step, writer):
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Generate translation
            model_out = model.generate(encoder_input, encoder_mask, max_len)
            
            # Convert ids to text
            predicted = tokenizer_tgt.decode(model_out[0].tolist())
            source = tokenizer_src.decode(encoder_input[0].tolist())
            target = batch['target_text'][0]
            
            # Log results
            print_msg(f"\nSOURCE: {source}")
            print_msg(f"TARGET: {target}")
            print_msg(f"PREDICTED: {predicted}")
```

## TensorBoard Integration

Start TensorBoard:

```bash
tensorboard --logdir=runs/
```

Logging code:

```python
writer.add_scalar('train loss', loss.item(), global_step)
writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)
writer.flush()
```

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

## Code Examples

Complete training loop:

```python
def train_model(config):
    batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
    for batch in batch_iterator:
        model.train()
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        
        # Forward pass
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, 
                                    decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        
        # Calculate loss
        label = batch['label'].to(device)
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                      label.view(-1))
        
        # Update progress
        batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
        
        # Log metrics
        writer.add_scalar('train loss', loss.item(), global_step)
        writer.flush()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        
        # Run validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                      config['seq_len'], device, 
                      lambda msg: batch_iterator.write(msg), 
                      global_step, writer)
        
        # Save checkpoint
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
```

## References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer architecture paper
   - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. **Coding a Transformer from scratch on PyTorch** by Umar Jamil
   - Implementation insights and techniques
   - [https://www.youtube.com/watch?v=ISNdQcPhsts](https://www.youtube.com/watch?v=ISNdQcPhsts)

3. **Hugging Face Tokenizers**
   - Text tokenization libraries
   - [https://huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers)

4. **PyTorch Documentation**
   - Framework documentation
   - [https://pytorch.org/](https://pytorch.org/)

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.
