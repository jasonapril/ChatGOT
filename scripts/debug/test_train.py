#!/usr/bin/env python
"""
Simple test script to test model training
"""
import os
import pickle
import json
import sys
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 64
SEQ_LENGTH = 1024
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 0.001
NUM_EPOCHS = 1

class SimpleModel(nn.Module):
    """A simple model for character-level prediction."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # Embed the input
        x = self.embedding(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Project to vocabulary size
        logits = self.fc_out(lstm_out)
        return logits

def main():
    """Main function to test training."""
    logger.info("Starting simple training test")
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the processed data
    try:
        data_path = os.path.join("processed_data", "got_char_data.pkl")
        vocab_path = os.path.join("processed_data", "vocab.json")
        
        logger.info(f"Loading processed data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        train_data = data['train_sequences']
        val_data = data['val_sequences']
        logger.info(f"Loaded {len(train_data)} training sequences and {len(val_data)} validation sequences")
        
        # Convert char2idx and idx2char
        char2idx = vocab['char_to_idx'] if 'char_to_idx' in vocab else data['char_to_idx']
        idx2char = vocab['idx_to_char'] if 'idx_to_char' in vocab else data['idx_to_char']
        vocab_size = vocab.get('vocab_size', len(char2idx))
        logger.info(f"Vocabulary size: {vocab_size}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Create simple data loader
    def create_batch(data_list, batch_size):
        """Create batches from the data list."""
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            # Each item is a tuple (input_tensor, target_tensor)
            inputs = torch.stack([seq[0] for seq in batch])
            targets = torch.stack([seq[1] for seq in batch])
            yield inputs, targets
    
    # Initialize model
    logger.info("Initializing model")
    model = SimpleModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info(f"Starting training for {NUM_EPOCHS} epochs")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batches = 0
        
        for inputs, targets in create_batch(train_data, BATCH_SIZE):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batches += 1
            
            if batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batches
        logger.info(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in create_batch(val_data, BATCH_SIZE):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Save the model
    logger.info("Saving model")
    save_path = os.path.join("models", "test_model.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
        'config': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 