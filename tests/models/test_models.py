"""
Unit tests for model base classes.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from pydantic import ValidationError, ConfigDict

# Import components directly from their specific modules
# Import Model (the nn.Module base), not BaseModel (the Pydantic one by mistake)
from craft.models.base import (
    Model, # Import the actual base nn.Module class
    GenerativeModel, LanguageModel, 
    BaseModelConfig, GenerativeModelConfig, LanguageModelConfig
)
from craft.models.factory import create_model_from_config
from craft.models.transformer import TransformerModel


# --- Mock Classes --- #

# Inherit from Model (nn.Module base), NOT pydantic.BaseModel
class MockBaseModel(Model):
    """Mock implementation of BaseModel for testing."""
    def __init__(self):
        super().__init__(config=BaseModelConfig())
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

# Inherit from GenerativeModel (which inherits Model)
class MockGenerativeModel(GenerativeModel):
    """Mock implementation of GenerativeModel for testing generate method."""
    def __init__(self, vocab_size=10, d_model=8, max_seq_length=20):
        # Use a config that matches expected attributes
        config = GenerativeModelConfig(
            model_type="generative", 
            max_seq_length=max_seq_length
            # Add vocab_size here if needed by forward mock logic, 
            # though base GenerativeModelConfig doesn't require it.
            # Let's add it to the dict for flexibility, Pydantic allows extra fields.
            # vocab_size=vocab_size 
        )
        # We need d_model for the dummy linear layer
        config.d_model = d_model # Add dynamically since it's not in GenerativeModelConfig
        config.vocab_size = vocab_size # Add dynamically for dummy output layer
        
        super().__init__(config=config)
        # Simple linear layer to mimic output projection
        self.linear = nn.Linear(d_model, vocab_size) 
    
    # This forward method will be mocked during tests
    def forward(self, x: torch.Tensor):
        # Dummy forward: creates output features based on input indices
        # The actual logits will be controlled by the mock object in tests.
        # Needs an embedding-like step for dimensionality change.
        # Let's simulate a simple projection based on input shape.
        batch_size, seq_len = x.shape
        # Dummy projection to d_model
        dummy_features = torch.randn(batch_size, seq_len, self.config.d_model, device=x.device)
        # Project to vocab size
        logits = self.linear(dummy_features)
        return logits
    
    # Remove the simple generate override, we want to test the base class method.
    # def generate(self, prompt, max_length=10):
    #     return torch.zeros(1, max_length)

# Inherit from LanguageModel (which inherits GenerativeModel)
class MockLanguageModel(LanguageModel):
    """Mock implementation of LanguageModel for testing."""
    def __init__(self):
        config = LanguageModelConfig(vocab_size=100)
        super().__init__(config=config)
        self.embedding = nn.Embedding(self.config.vocab_size, 10)
        self.linear = nn.Linear(10, self.config.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)
    
    # Override generate for simple test
    def generate(self, input_ids, max_new_tokens=10):
        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, input_ids.shape[1] + max_new_tokens)


class TestBaseModel(unittest.TestCase):
    """Tests for the Model base class (nn.Module)."""
    
    def setUp(self):
        self.model = MockBaseModel()
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        # Check type and model_type from the config
        self.assertIsInstance(self.model.config, BaseModelConfig)
        self.assertEqual(self.model.model_type, "base")
        self.assertIsInstance(self.model, nn.Module) # Check it's an nn.Module
    
    def test_forward(self):
        """Test the forward method."""
        x = torch.randn(5, 10)
        output = self.model(x)
        self.assertEqual(output.shape, (5, 10))
    
    def test_get_config(self):
        """Test the get_config method."""
        config = self.model.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config["model_type"], "base")
    
    def test_save_load(self):
        """Test the save and load methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model.pt")
            self.model.save(path)
            self.assertTrue(os.path.exists(path))
            
            # Determine target device (consistent with load method)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create a new model and load the saved state
            new_model = MockBaseModel()
            # Load method moves new_model to device
            new_model.load(path, device=device) 
            
            # Ensure the original model is also on the same device for comparison
            self.model.to(device)
            
            # Check that the parameters are the same
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                # Ensure parameters being compared are on the same device explicitly
                self.assertTrue(torch.all(torch.eq(p1.to(device), p2.to(device))))


class TestGenerativeModel(unittest.TestCase):
    """Tests for the GenerativeModel class."""
    
    def setUp(self):
        self.model = MockGenerativeModel()
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model.config, GenerativeModelConfig)
        self.assertEqual(self.model.model_type, "generative")
        # Check that it IS an instance of the base Model class
        self.assertIsInstance(self.model, Model) 
    

class TestLanguageModel(unittest.TestCase):
    """Tests for the LanguageModel class."""
    
    def setUp(self):
        self.model = MockLanguageModel()
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model.config, LanguageModelConfig)
        self.assertEqual(self.model.model_type, "language")
        # Check inheritance
        self.assertIsInstance(self.model, GenerativeModel)
        self.assertIsInstance(self.model, Model)
    
    def test_forward(self):
        """Test the forward method."""
        x = torch.randint(0, 100, (5, 10))
        output = self.model(x)
        self.assertEqual(output.shape, (5, 10, 100))
    
    def test_generate_method(self):
        """Test the generate method."""
        input_ids = torch.randint(0, 100, (2, 5))
        output = self.model.generate(input_ids, max_new_tokens=15)
        self.assertEqual(output.shape, (2, 20))  # input (5) + new tokens (15)
    
    def test_calculate_perplexity(self):
        """Test the calculate_perplexity method."""
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        perplexity = self.model.calculate_perplexity(logits, targets)
        self.assertGreater(perplexity.item(), 0)


class TestModelCreation(unittest.TestCase):
    """Tests for the model creation factory function using the registry."""
    
    def test_create_language_transformer_model(self):
        """Test creating a TransformerModel via the factory."""
        # Define a minimal valid config
        config_dict = {
            "model_type": "language",
            "architecture": "transformer", # Matches the registered name
            "vocab_size": 50,          # Required for LanguageModelConfig
            "d_model": 128,            # Optional, provide for testing
            "n_head": 4,               # Optional, provide for testing
            "n_layers": 2,             # Optional, provide for testing
            # Other fields will use defaults from LanguageModelConfig
        }
        
        # Call the factory function
        model = create_model_from_config(config_dict)
        
        # --- Assertions --- 
        # Check the type
        self.assertIsNotNone(model)
        self.assertIsInstance(model, TransformerModel)
        
        # Check if the config object was created and assigned correctly
        self.assertTrue(hasattr(model, 'config'))
        self.assertEqual(model.config.model_type, "language")
        self.assertEqual(model.config.architecture, "transformer")
        
        # Check specific parameters passed or defaulted
        self.assertEqual(model.config.vocab_size, 50)
        self.assertEqual(model.config.d_model, 128)
        self.assertEqual(model.config.n_head, 4)
        self.assertEqual(model.config.n_layers, 2)
        # Check the defaulted d_hid ON THE MODEL ITSELF, as config might not store it
        self.assertEqual(model.d_hid, 128 * 4)
        self.assertEqual(model.model_type, "language")

        # Check parameters used by the model's __init__
        self.assertEqual(model.d_model, 128)
        self.assertEqual(model.transformer_decoder.num_layers, 2)
        self.assertEqual(model.token_embedding.num_embeddings, 50)
        self.assertEqual(model.output_layer.out_features, 50)
    
    def test_unregistered_model_type_or_architecture(self):
        """Test error for unregistered model type/architecture combo."""
        config = {
            "model_type": "language",
            "architecture": "non_existent_transformer",
            "vocab_size": 100
        }
        
        # Expect ValueError from factory if the combo isn't in the registry
        with self.assertRaises(ValueError):
            create_model_from_config(config)
            
        config_unknown_type = {
            "model_type": "audio", # Assuming 'audio' is not registered
            "architecture": "transformer",
            "vocab_size": 100
        }
        with self.assertRaises(ValueError):
             create_model_from_config(config_unknown_type)

    def test_invalid_config_validation(self):
        """Test Pydantic validation for missing required fields."""
        # Config missing the required 'vocab_size' for LanguageModelConfig
        invalid_config = {
            "model_type": "language",
            "architecture": "transformer",
            "d_model": 64 
        }
        
        # Expect Pydantic ValidationError
        with self.assertRaises(ValidationError):
            create_model_from_config(invalid_config)


# --- New Test Class for Base Generate Method --- #
class TestGenerateMethod(unittest.TestCase):
    """Tests the generate method implemented in GenerativeModel base class."""
    
    def setUp(self):
        self.vocab_size = 10
        self.d_model = 8
        self.max_seq_length = 20 # Model's max sequence length
        self.model = MockGenerativeModel(
            vocab_size=self.vocab_size, 
            d_model=self.d_model, 
            max_seq_length=self.max_seq_length
        )
        self.model.eval() # Set to eval mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def test_max_new_tokens(self):
        """Test if generation stops after max_new_tokens."""
        start_ids = torch.tensor([[1, 2]], dtype=torch.long, device=self.device)
        max_new = 5
        
        # Mock forward to always return the same logits
        # Logits shape: [batch_size, seq_len, vocab_size]
        # We only care about the last token's logits: [batch_size, vocab_size]
        mock_logits = torch.randn(1, self.vocab_size, device=self.device)
        
        with patch.object(self.model, 'forward', return_value=mock_logits.unsqueeze(1)) as mock_fwd:
            generated_tokens = self.model.generate(start_ids, max_new_tokens=max_new)
            
            # Check output length
            self.assertEqual(generated_tokens.shape[1], start_ids.shape[1] + max_new)
            # Check forward was called correct number of times
            self.assertEqual(mock_fwd.call_count, max_new)

    def test_eos_token_stopping(self):
        """Test if generation stops when eos_token_id is produced."""
        start_ids = torch.tensor([[1, 2]], dtype=torch.long, device=self.device)
        eos_id = 9
        max_new = 10
        
        # Mock forward to produce eos_id on the 3rd step
        # Logits: [batch_size, seq_len, vocab_size]
        # Mock return values for step 1, 2, 3...
        logits_step1 = torch.zeros(1, self.vocab_size, device=self.device)
        logits_step1[:, 3] = 10 # Predict token 3
        
        logits_step2 = torch.zeros(1, self.vocab_size, device=self.device)
        logits_step2[:, 4] = 10 # Predict token 4
        
        logits_step3 = torch.zeros(1, self.vocab_size, device=self.device)
        logits_step3[:, eos_id] = 10 # Predict EOS token

        # Make subsequent calls return EOS again (or anything, won't be used)
        mock_fwd = MagicMock(side_effect=[
            logits_step1.unsqueeze(1), 
            logits_step2.unsqueeze(1), 
            logits_step3.unsqueeze(1),
            logits_step3.unsqueeze(1), # Subsequent calls don't matter
        ] * (max_new // 4 + 1)) # Repeat to cover max_new calls

        with patch.object(self.model, 'forward', mock_fwd):
            generated_tokens = self.model.generate(
                start_ids, 
                max_new_tokens=max_new, 
                eos_token_id=eos_id,
                temperature=0 # Use greedy for predictability
            )
            
            # Expected output: [1, 2, 3, 4, 9]
            expected_len = start_ids.shape[1] + 3 # Start + 3 new tokens (inc EOS)
            self.assertEqual(generated_tokens.shape[1], expected_len)
            self.assertEqual(generated_tokens[0, -1].item(), eos_id)
            # Check forward was called only 3 times before stopping
            self.assertEqual(mock_fwd.call_count, 3)

    def test_greedy_decoding(self):
        """Test temperature=0 selects the highest logit."""
        start_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)
        max_new = 3
        
        # Logits where index 5 is highest, then 6, then 7
        logits1 = torch.arange(0, self.vocab_size, dtype=torch.float, device=self.device).unsqueeze(0)
        logits2 = logits1.clone(); logits2[0, 6] = 100
        logits3 = logits1.clone(); logits3[0, 7] = 100

        mock_fwd = MagicMock(side_effect=[
            logits1.unsqueeze(1),
            logits2.unsqueeze(1),
            logits3.unsqueeze(1),
        ])

        with patch.object(self.model, 'forward', mock_fwd):
             generated_tokens = self.model.generate(
                start_ids, 
                max_new_tokens=max_new, 
                temperature=0 # Greedy
            )
        
        # Expected: start_id + argmax of logits1[-1], logits2[-1], logits3[-1]
        expected_tokens = torch.tensor([[1, self.vocab_size-1, 6, 7]], dtype=torch.long, device=self.device)
        self.assertTrue(torch.equal(generated_tokens, expected_tokens))
        self.assertEqual(mock_fwd.call_count, max_new)

    def test_sequence_truncation(self):
        """Test input sequences longer than max_seq_length are truncated."""
        # Input length > model.max_seq_length (20)
        start_len = 25
        start_ids = torch.randint(1, self.vocab_size, (1, start_len), dtype=torch.long, device=self.device)
        max_new = 2
        
        # Mock forward to check the input it receives
        mock_fwd = MagicMock(return_value=torch.randn(1, 1, self.vocab_size, device=self.device))
        
        with patch.object(self.model, 'forward', mock_fwd):
            self.model.generate(start_ids, max_new_tokens=max_new)
        
        # Check that the first call to forward received a truncated sequence
        first_call_args, _ = mock_fwd.call_args_list[0]
        input_tensor_arg = first_call_args[0]
        self.assertEqual(input_tensor_arg.shape[1], self.max_seq_length) # Should be truncated to 20
        # Check it's the *last* part of the original input
        self.assertTrue(torch.equal(input_tensor_arg[0], start_ids[0, -self.max_seq_length:]))

    def test_temperature_sampling(self):
        """Test that temperature > 0 introduces randomness."""
        start_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)
        # Generate only one new token per iteration to isolate sampling
        max_new = 1 
        num_samples = 50 # Increase number of sampling trials
        temp = 0.8
        greedy_token_idx = 5
        
        # Logits where index 5 is highest, but 6 and 7 are plausible
        logits = torch.zeros(1, self.vocab_size, device=self.device)
        logits[0, greedy_token_idx] = 10.0
        logits[0, 6] = 9.0 
        logits[0, 7] = 8.0

        mock_fwd = MagicMock(return_value=logits.unsqueeze(1))

        generated_tokens_list = []
        with patch.object(self.model, 'forward', mock_fwd):
            for _ in range(num_samples): # Generate multiple times to check for variance
                generated = self.model.generate(
                    start_ids, 
                    max_new_tokens=max_new, 
                    temperature=temp
                )
                # Append the *single* newly generated token
                generated_tokens_list.append(generated[0, -1].item()) 
        
        # Check that not all generated tokens are the greedy choice
        # With T=0.8 and 50 samples, it's highly unlikely we get only the top token.
        self.assertFalse(all(token == greedy_token_idx for token in generated_tokens_list),
                         f"Temperature={temp} should introduce randomness, but only token {greedy_token_idx} was selected over {num_samples} samples.")
        # Check forward was called max_new times for each generation
        self.assertEqual(mock_fwd.call_count, num_samples * max_new)
        
    def test_top_p_sampling(self):
        """Test that top_p sampling restricts choices based on cumulative probability."""
        start_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)
        max_new = 1
        p = 0.9
        num_samples = 50 # Increase number of samples
        
        # Logits designed so top_p=0.9 selects indices {2, 3}
        # Logits -> Softmax -> Probs -> Cumsum
        # idx:   0    1     2     3     4    5    6    7    8     9
        # log: -10  -10   10.0   8.0  -10  -10  -10  -10  -10   7.0
        # prob:~0    ~0    0.88  0.12  ~0   ~0   ~0   ~0   ~0   0.004  (Approx after softmax)
        # sort: 0.88(2) 0.12(3) 0.004(9) ...
        # cumu: 0.88    1.0     1.004 ...
        # Nucleus for p=0.9 should be {2, 3}
        logits = torch.full((1, self.vocab_size), -10.0, dtype=torch.float, device=self.device)
        logits[0, 2] = 10.0
        logits[0, 3] = 8.0
        logits[0, 9] = 7.0 # Let's add index 9
        # Recalculate approx probabilities: exp(10)~22026, exp(8)~2981, exp(7)~1097. Sum ~ 26104
        # p(2) ~ 0.84, p(3) ~ 0.11, p(9) ~ 0.04. Others ~0. Sum ~ 0.99
        # Sorted: 0.84(2), 0.11(3), 0.04(9)
        # Cumsum: 0.84,      0.95,      0.99
        # Nucleus for p=0.9 should be {2, 3}
        nucleus_indices = {2, 3}
        
        mock_fwd = MagicMock(return_value=logits.unsqueeze(1))
        
        generated_tokens = []
        with patch.object(self.model, 'forward', mock_fwd):
            for _ in range(num_samples): # Sample multiple times (increased)
                generated = self.model.generate(
                    start_ids, 
                    max_new_tokens=max_new, 
                    top_p=p,
                    temperature=1.0 # Avoid temp=0 interfering
                )
                generated_tokens.append(generated[0, -1].item())
                
        # Check that all generated tokens are within the nucleus set
        self.assertTrue(all(token in nucleus_indices for token in generated_tokens),
                        f"Generated tokens {generated_tokens} contain items outside the top_p nucleus {nucleus_indices}")
        # Check that some variation occurred (probabilistic)
        self.assertTrue(len(set(generated_tokens)) > 1, 
                        f"Expected variation in top-p sampling (nucleus={nucleus_indices}), but got only {set(generated_tokens)} over {num_samples} trials.")
        self.assertEqual(mock_fwd.call_count, num_samples * max_new)


# --- Test Specific Model Implementations --- #

class TestTransformerModel(unittest.TestCase):
    """Tests for the TransformerModel implementation."""
    
    def setUp(self):
        self.config = LanguageModelConfig(
            model_type="language", # Base type
            model_architecture="transformer", # Specific type
            vocab_size=50,
            max_seq_length=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.1,
            bias=False
        )
        self.model = TransformerModel(config=self.config)
        self.model.eval() # Set to eval mode for consistency
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def test_forward_pass_shape(self):
        """Test the shape of the output from the forward pass."""
        batch_size = 4
        seq_len = self.config.max_seq_length // 2 # Use a seq_len < max_seq_length
        # Input tensor with random token indices (must be within vocab_size)
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=self.device)
        
        # Perform forward pass
        with torch.no_grad(): # No need to track gradients for shape testing
            logits = self.model(input_ids)
            
        # Check output shape
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(logits.shape, expected_shape)


# --- New Pytest-style Tests for TransformerModel (from craft_test_cases.py) ---

def test_transformer_creation():
    """Test creating a transformer model with default parameters."""
    # Create config for a small model
    config = LanguageModelConfig(
        vocab_size=100, 
        max_seq_length=128,
        d_model=256, 
        n_head=4, 
        d_hid=512, # Explicitly set d_hid for clarity in test
        n_layers=2,
        dropout=0.1
    )
    # Instantiate the model directly
    model = TransformerModel(config=config)
    
    # Test that the model is a nn.Module
    assert isinstance(model, nn.Module)
    
    # Test that the model has the correct attributes from config
    assert model.d_model == 256
    assert model.n_head == 4
    assert model.n_layers == 2
    assert model.vocab_size == 100
    
    # Test model output with a sample input
    batch_size = 2
    seq_len = 128
    
    # Generate sample input
    x = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, 100)

def test_transformer_parameter_count():
    """Test that the transformer model has a reasonable number of parameters."""
    # Create a small model config
    config = LanguageModelConfig(
        vocab_size=100, 
        max_seq_length=128,
        d_model=256, 
        n_head=4, 
        d_hid=512, 
        n_layers=2,
        dropout=0.1
    )
    # Instantiate the model
    model = TransformerModel(config=config)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # The actual count will vary, but should be in a reasonable range
    # This test is just to catch major changes that affect parameter count
    assert param_count > 100000  # At least 100K parameters
    assert param_count < 5000000   # Less than 5M parameters


if __name__ == "__main__":
    unittest.main() 