# Extending Craft

This document provides guidance on how to extend Craft with custom models, datasets, and trainers.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
  - [Base Model Types](#base-model-types)
  - [Creating a Custom Model](#creating-a-custom-model)
  - [Model Factory](#model-factory)
- [Datasets](#datasets)
  - [Base Dataset Types](#base-dataset-types)
  - [Creating a Custom Dataset](#creating-a-custom-dataset)
  - [Dataset Factory](#dataset-factory)
- [Trainers](#trainers)
  - [Base Trainer Types](#base-trainer-types)
  - [Creating a Custom Trainer](#creating-a-custom-trainer)
  - [Trainer Factory](#trainer-factory)
- [Configuration](#configuration)
  - [Model Configuration](#model-configuration)
  - [Training Configuration](#training-configuration)
  - [Data Configuration](#data-configuration)
- [Example: Custom Image Model](#example-custom-image-model)

## Overview

Craft is designed to be extensible, allowing you to add custom models, datasets, and trainers while maintaining a consistent interface. This document explains how to leverage the base classes and factories to add new components.

## Models

### Base Model Types

Craft provides several base model types:

- `BaseModel`: The core base class for all models
- `GenerativeModel`: For models that generate outputs (inherits from `BaseModel`)
- `LanguageModel`: For language models (inherits from `GenerativeModel`)
- `VisionModel`: For vision models (inherits from `BaseModel`)
- `MultiModalModel`: For multi-modal models (inherits from `GenerativeModel`)

These classes define the interface and common functionality for each model type.

### Creating a Custom Model

To create a custom model, inherit from the appropriate base class and implement the required methods:

```python
from src.models.base import LanguageModel
import torch
import torch.nn as nn

class CustomLanguageModel(LanguageModel):
    """A custom language model implementation."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Log model size
        self._log_model_size()
    
    def forward(self, x, targets=None):
        """Forward pass of the model."""
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        logits = self.output_layer(output)
        
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, **kwargs):
        """Generate text from the model."""
        self.eval()
        
        # Clone input to avoid modifying the original
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions
                outputs = self(generated)
                next_token_logits = outputs[:, -1, :]
                
                # Sample next token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
```

### Model Factory

To make your custom model available through the model factory, you need to add it to the `create_model_from_config` function in `src/models/base.py`:

```python
def create_model_from_config(config: Dict) -> BaseModel:
    """Create a model from a configuration dictionary."""
    model_type = config.get("model_type", "language")
    
    if model_type == "language":
        # Add your custom model here
        if config.get("architecture") == "custom_lstm":
            from .custom_models import CustomLanguageModel
            return CustomLanguageModel(
                vocab_size=config.get("vocab_size", 100),
                embedding_dim=config.get("embedding_dim", 128),
                hidden_dim=config.get("hidden_dim", 256)
            )
        # Existing options...
        elif config.get("architecture") == "transformer":
            from .transformer import create_transformer_model
            return create_transformer_model(**config)
        else:
            raise ValueError(f"Unknown language model architecture: {config.get('architecture')}")
    # Other model types...
```

## Datasets

### Base Dataset Types

Craft provides several base dataset types:

- `BaseDataset`: The core base class for all datasets
- `TextDataset`: For text datasets (inherits from `BaseDataset`)
- `ImageDataset`: For image datasets (inherits from `BaseDataset`)
- `MultiModalDataset`: For multi-modal datasets (inherits from `BaseDataset`)

### Creating a Custom Dataset

To create a custom dataset, inherit from the appropriate base class and implement the required methods:

```python
from src.data.base import TextDataset
import torch

class CustomTextDataset(TextDataset):
    """A custom text dataset implementation."""
    
    def __init__(self, texts, labels, max_length=128):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Example tokenization (simplified)
        self.vocab = sorted(list(set("".join(texts))))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to token ids
        token_ids = [self.char_to_idx.get(c, 0) for c in text[:self.max_length]]
        
        # Pad to max length
        if len(token_ids) < self.max_length:
            token_ids.extend([0] * (self.max_length - len(token_ids)))
        
        return torch.tensor(token_ids), torch.tensor(label)
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return "".join([self.idx_to_char.get(i, "") for i in indices])
```

### Dataset Factory

To make your custom dataset available through the dataset factory, add it to the `create_dataset_from_config` function in `src/data/base.py`:

```python
def create_dataset_from_config(config: Dict) -> BaseDataset:
    """Create a dataset from a configuration dictionary."""
    data_type = config.get("data_type", "text")
    
    if data_type == "text":
        # Add your custom dataset here
        if config.get("format") == "custom_text":
            from .custom_datasets import CustomTextDataset
            
            # Load data from file
            with open(config["data_path"], "r") as f:
                lines = f.readlines()
            
            texts = [line.strip() for line in lines]
            labels = [0] * len(texts)  # Placeholder labels
            
            return CustomTextDataset(
                texts=texts,
                labels=labels,
                max_length=config.get("max_length", 128)
            )
        # Existing options...
        elif config.get("format") == "character":
            # ...
    # Other data types...
```

## Trainers

### Base Trainer Types

Craft provides several base trainer types:

- `Trainer`: The core base class for all trainers
- `LanguageModelTrainer`: For training language models (inherits from `Trainer`)

### Creating a Custom Trainer

To create a custom trainer, inherit from the appropriate base class and implement the required methods:

```python
from src.training.base import Trainer
import torch

class CustomTrainer(Trainer):
    """A custom trainer implementation."""
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in self.train_dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.calculate_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_samples += inputs.size(0)
        
        return {"loss": total_loss / len(self.train_dataloader)}
    
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.calculate_loss(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                total_samples += inputs.size(0)
        
        return {"loss": total_loss / len(self.val_dataloader)}
    
    def calculate_loss(self, outputs, targets):
        """Calculate loss based on outputs and targets."""
        # You can define a custom loss function here
        return torch.nn.functional.cross_entropy(outputs, targets)
```

### Trainer Factory

To make your custom trainer available through the trainer factory, add it to the `create_trainer_from_config` function in `src/training/base.py`:

```python
def create_trainer_from_config(
    model: BaseModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Dict[str, Any] = None,
) -> Trainer:
    """Create a trainer from a configuration dictionary."""
    if config is None:
        config = {}
    
    model_type = getattr(model, 'model_type', 'language')
    
    # Add your custom trainer here
    if config.get("trainer_type") == "custom":
        from .custom_trainers import CustomTrainer
        return CustomTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
        )
    
    # Existing options...
    elif model_type == 'language':
        return LanguageModelTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
        )
    # Other model types...
```

## Configuration

### Model Configuration

To configure your custom model, create a YAML file in the `configs/models/` directory:

```yaml
# configs/models/custom_lstm.yaml
model_type: language
architecture: custom_lstm
vocab_size: 100
embedding_dim: 128
hidden_dim: 256
```

### Training Configuration

To configure your custom training, create a YAML file in the `configs/training/` directory:

```yaml
# configs/training/custom_training.yaml
trainer_type: custom
epochs: 10
batch_size: 32
learning_rate: 0.001
```

### Data Configuration

To configure your custom dataset, create a YAML file in the `configs/data/` directory:

```yaml
# configs/data/custom_text.yaml
data_type: text
format: custom_text
data_path: data/raw/custom_text.txt
max_length: 128
```

## Example: Custom Image Model

Here's a complete example of adding a custom image model:

1. Define the model:

```python
# src/models/custom_cnn.py
from src.models.base import VisionModel
import torch
import torch.nn as nn

class CustomCNN(VisionModel):
    """A custom CNN model for image classification."""
    
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
        
        # Log model size
        self._log_model_size()
    
    def forward(self, x, targets=None):
        """Forward pass of the model."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        if targets is not None:
            loss = nn.functional.cross_entropy(logits, targets)
            return logits, loss
        
        return logits

# Add to the model factory
def create_cnn_model(num_classes=10, in_channels=3):
    """Create a CNN model."""
    return CustomCNN(num_classes=num_classes, in_channels=in_channels)
```

2. Define the dataset:

```python
# src/data/custom_image_dataset.py
from src.data.base import ImageDataset
import torch
from PIL import Image
import os

class CustomImageDataset(ImageDataset):
    """A custom dataset for image classification."""
    
    def __init__(self, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        
        # Example class mapping
        self.class_names = ["cat", "dog"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # Get label from filename (example: cat_001.jpg -> cat)
        label_name = self.image_files[idx].split("_")[0]
        label = self.class_to_idx.get(label_name, 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)
```

3. Define the trainer:

```python
# src/training/vision_trainer.py
from src.training.base import Trainer
import torch
import torch.nn.functional as F

class VisionTrainer(Trainer):
    """Trainer for vision models."""
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for i, batch in enumerate(self.train_dataloader):
            # Get batch
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, loss = outputs
            else:
                logits = outputs
                loss = F.cross_entropy(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Log progress
            if (i + 1) % self.log_interval == 0:
                logging.info(
                    f"Batch {i + 1}/{len(self.train_dataloader)} - "
                    f"loss: {loss.item():.4f} - "
                    f"accuracy: {100 * total_correct / total_samples:.2f}%"
                )
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = 100 * total_correct / total_samples
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Get batch
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, loss = outputs
                else:
                    logits = outputs
                    loss = F.cross_entropy(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100 * total_correct / total_samples
        
        return {"loss": avg_loss, "accuracy": accuracy}
```

4. Update the factories:

```python
# In src/models/base.py
def create_model_from_config(config: Dict) -> BaseModel:
    # ...
    elif model_type == "vision":
        if config.get("architecture") == "cnn":
            from .custom_cnn import create_cnn_model
            return create_cnn_model(
                num_classes=config.get("num_classes", 10),
                in_channels=config.get("in_channels", 3)
            )
    # ...

# In src/data/base.py
def create_dataset_from_config(config: Dict) -> BaseDataset:
    # ...
    elif data_type == "image":
        if config.get("format") == "custom_image":
            from .custom_image_dataset import CustomImageDataset
            from torchvision import transforms
            
            # Create transforms
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            return CustomImageDataset(
                image_dir=config["data_path"],
                transform=transform
            )
    # ...

# In src/training/base.py
def create_trainer_from_config(model, train_dataloader, val_dataloader=None, config=None):
    # ...
    elif model_type == "vision":
        from .vision_trainer import VisionTrainer
        return VisionTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config
        )
    # ...
```

5. Create configuration files:

```yaml
# configs/models/custom_cnn.yaml
model_type: vision
architecture: cnn
num_classes: 10
in_channels: 3

# configs/data/custom_image.yaml
data_type: image
format: custom_image
data_path: data/raw/images
batch_size: 32
num_workers: 4

# configs/training/vision_training.yaml
epochs: 10
batch_size: 32
learning_rate: 0.001
optimizer:
  name: adam
  beta1: 0.9
  beta2: 0.999
scheduler:
  name: cosine
  warmup_ratio: 0.1
```

6. Use it from the CLI:

```bash
# Train a vision model
craft train vision -c configs/experiments/custom_vision.yaml
```

This example demonstrates how to extend Craft with a custom vision model, dataset, and trainer. 