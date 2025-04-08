import pytest
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import hydra
from pathlib import Path
import tempfile
import os
import pickle # Added for dummy data
import logging # Added logging

# Necessary imports from the project
from craft.data.tokenizers.char import CharTokenizer
from craft.data.datasets.pickled_dataset import PickledDataset
from craft.models.transformer import TransformerModel # Corrected model class name
from craft.training.training_loop import TrainingLoop
from craft.training.progress import ProgressTracker
from craft.config.schemas import TrainingConfig, LanguageModelConfig # Import the Pydantic models

# TODO: Define or import necessary fixtures/helper functions

def test_core_pipeline(tmp_path: Path):
    """
    Tests the core pipeline: Config -> Data -> Model -> Train (minimal steps).
    """
    # --- 1. Configuration ---
    
    # Create dummy tokenizer vocab file
    vocab_content = " .abcdefghijklmnopqrstuvwxyz\n" # Note the leading space and trailing newline
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text(vocab_content)
    vocab_size = len(vocab_content)

    # Create dummy tokenized data and pickle it
    # Simulate some token IDs based on the vocab (ensure IDs match CharTokenizer)
    # IDs: space=0, .=1, a=2, b=3, ..., z=27, \n=28
    # h=9, e=6, l=13, o=16, w=24, r=19, d=5, t=21, i=10, s=20
    dummy_token_ids = [
        9, 6, 13, 13, 16, 0, 24, 16, 19, 13, 5, 1, 28,  # "hello world.\n"
        21, 9, 10, 20, 0, 10, 20, 0, 2, 0, 21, 6, 20, 21, 1, 28, # "this is a test.\n"
        # Current block_size=16, batch_size=2. Need >= 32 tokens per step.
        # max_steps=3. Need >= 32 * 3 = 96 tokens.
        # Current length is (13 + 16) = 29 tokens per sequence.
        # Repeating 5 times gives 145 tokens. Should be sufficient.
    ] * 5 
    
    dummy_pickled_data_path = tmp_path / "dummy_train.pkl"
    with open(dummy_pickled_data_path, 'wb') as f:
        pickle.dump(torch.tensor(dummy_token_ids, dtype=torch.long), f)

    # --- OmegaConf Configuration Dict --- 
    cfg_dict = {
        'defaults': ['_self_'], 
        
        'data': {
            'tokenizer': {
                '_target_': 'craft.data.tokenizers.char.CharTokenizer',
                'vocab_path': str(vocab_file),
            },
            'dataset': {
                '_target_': 'craft.data.datasets.pickled_dataset.PickledDataset',
                'file_path': str(dummy_pickled_data_path),
                'block_size': 16,
            },
            'dataloader': {
                '_target_': 'torch.utils.data.DataLoader',
                'dataset': '${data.dataset}', 
                'batch_size': 2,
                'shuffle': False, 
            }
        },
        'model': {
            # Target for the model class itself
            '_target_': 'craft.models.transformer.TransformerModel', 
            # Nested config object that will be passed to the model's __init__
            'config': {
                # Target for the Pydantic config schema
                '_target_': 'craft.config.schemas.LanguageModelConfig',
                # Actual model parameters, now fields within the config object
                'vocab_size': vocab_size, 
                'block_size': 16, 
                'max_seq_length': 16, # Add max_seq_length if required by schema
                'n_layer': 1,
                'n_head': 1,
                'd_model': 4, # Rename n_embd to d_model if schema uses that
                # 'n_embd': 4, # Keep only one embedding dim name
                'dropout': 0.0,
                'bias': False,
                # Add other fields required by LanguageModelConfig if any
                # e.g., d_hid, activation, norm_first, layer_norm_eps
                'd_hid': 16, # Example: d_model * 4 often default
                'activation': 'relu', # Example
                'norm_first': False, # Example
                'layer_norm_eps': 1e-5 # Example
            }
            # 'vocab_size': vocab_size, # Moved into config sub-dict
            # 'block_size': 16, # Moved
            # 'n_layer': 1, # Moved
            # 'n_head': 1, # Moved
            # 'n_embd': 4, # Moved & potentially renamed
            # 'dropout': 0.0, # Moved
            # 'bias': False, # Moved
        },
        'optimizer': {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-4,
            'weight_decay': 0.01,
        },
        'training': {
            # Args directly for TrainingLoop constructor (non-config ones)
            '_target_': 'craft.training.training_loop.TrainingLoop',
            'device': 'cpu', 
            'use_amp': False,
            'gradient_accumulation_steps': 1,
            'log_interval': 1,
            'max_steps': 3, 
            # Pydantic TrainingConfig structure (passed as 'config' arg)
            'config': { 
                 # These need to match the fields in the TrainingConfig Pydantic model
                 '_target_': 'craft.config.schemas.TrainingConfig', # Target for the config object itself
                 'max_steps': '${..max_steps}', # Reference parent node
                 'log_interval': '${..log_interval}', # Reference parent node
                 'batch_size': '${data.dataloader.batch_size}',
                 'use_amp': '${..use_amp}', # Reference parent node
                 'gradient_accumulation_steps': '${..gradient_accumulation_steps}', 
                 'learning_rate': '${optimizer.lr}', 
                 'weight_decay': '${optimizer.weight_decay}', 
                 'max_grad_norm': None 
            },
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    OmegaConf.resolve(cfg) # Resolve interpolations

    print("--- Resolved Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------")

    # --- 2. Instantiate Components ---
    try:
        print("Instantiating Dataset...")
        dataset = instantiate(cfg.data.dataset) 
        print(f"Dataset instantiated: {type(dataset)}")
        assert dataset is not None
        assert dataset.block_size == cfg.data.dataset.block_size
        assert len(dataset) > 0 
        sample_x, sample_y = dataset[0] 
        assert sample_x.shape == (cfg.data.dataset.block_size,)
        assert sample_y.shape == (cfg.data.dataset.block_size,)
        print(f"Dataset length: {len(dataset)}, Sample X shape: {sample_x.shape}")

        print("Instantiating DataLoader...")
        dataloader = instantiate(cfg.data.dataloader, dataset=dataset)
        print(f"DataLoader instantiated: {type(dataloader)}")
        assert dataloader is not None
        assert dataloader.batch_size == cfg.data.dataloader.batch_size
        first_batch = next(iter(dataloader))
        assert isinstance(first_batch, (list, tuple)) and len(first_batch) == 2 
        assert first_batch[0].shape == (cfg.data.dataloader.batch_size, cfg.data.dataset.block_size)
        print(f"DataLoader batch size: {dataloader.batch_size}, First batch X shape: {first_batch[0].shape}")

        print("Instantiating Model...")
        model = instantiate(cfg.model)
        model.to(cfg.training.device) 
        print(f"Model instantiated: {type(model)}")
        assert model is not None
        # Test forward pass
        inputs = first_batch[0].to(cfg.training.device)
        targets = first_batch[1].to(cfg.training.device) 
        with torch.no_grad():
             try:
                 logits, loss = model(inputs, targets=targets)
                 assert loss is not None
             except TypeError:
                 logits = model(inputs)
                 loss = None # Loss check might need manual calculation if not returned
        assert logits.shape == (cfg.data.dataloader.batch_size, cfg.data.dataset.block_size, cfg.model.config.vocab_size)
        print(f"Model forward pass OK. Logits shape: {logits.shape}")

        print("Instantiating Optimizer...")
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        print(f"Optimizer instantiated: {type(optimizer)}")
        assert optimizer is not None

        print("Instantiating TrainingLoop...")
        # Instantiate the Pydantic TrainingConfig object first
        training_config_obj = instantiate(cfg.training.config)
        assert isinstance(training_config_obj, TrainingConfig)

        # Instantiate TrainingLoop directly, bypassing Hydra instantiate for this step
        # Simplified instantiate call that failed:
        # training_loop = instantiate(
        #     {'_target_': 'craft.training.training_loop.TrainingLoop'},
        #     model=model,
        #     optimizer=optimizer,
        #     train_dataloader=dataloader,
        #     device=torch.device(cfg.training.device), # Ensure device object is passed
        #     config=training_config_obj # Pass the instantiated Pydantic object
        # )
        
        try:
             print("Instantiating TrainingLoop directly...")
             training_loop = TrainingLoop(
                 model=model,
                 optimizer=optimizer,
                 train_dataloader=dataloader,
                 device=torch.device(cfg.training.device),
                 config=training_config_obj,
                 # Explicitly pass other args from config that have defaults in __init__
                 experiment_config=cfg, # Pass the full OmegaConf config
                 use_amp=cfg.training.use_amp,
                 gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                 log_interval=cfg.training.log_interval,
                 max_steps=cfg.training.max_steps,
                 max_grad_norm=cfg.training.config.max_grad_norm
                 # Relying on __init__ defaults for: scheduler, callbacks, checkpoint_manager, etc.
             )
        except Exception as direct_init_error:
             import traceback
             trace = traceback.format_exc()
             pytest.fail(f"Direct TrainingLoop instantiation failed: {direct_init_error}\n{trace}")

        
        print(f"TrainingLoop instantiated: {type(training_loop)}")
        assert training_loop is not None
        assert training_loop.max_steps == cfg.training.max_steps
        assert training_loop.model is model
        assert training_loop.optimizer is optimizer
        assert training_loop.train_dataloader is dataloader
        assert training_loop.device == torch.device(cfg.training.device)
        assert training_loop.config == training_config_obj 

    except hydra.errors.InstantiationException as e:
        # Safely access config if available, otherwise just print the exception
        config_str = OmegaConf.to_yaml(getattr(e, 'config', None)) if hasattr(e, 'config') else "N/A"
        pytest.fail(f"Hydra Instantiation failed: {e}\nConfig Node:\n{config_str}")
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        pytest.fail(f"Setup failed during instantiation: {e}\n{trace}")

    # --- 3. Minimal Training Run ---
    print("Starting minimal training run...")
    try:
        class MockTrainer: 
            def __init__(self):
                self.state = {'epoch': 0, 'global_step': 0} 
                self.should_stop = False
                self.logger = logging.getLogger("MockTrainerIntegrationTest") 
        
        mock_trainer = MockTrainer()
        
        class MockProgressTracker(ProgressTracker):
             def __init__(self, total_steps):
                 self.total = total_steps
                 self.n = 0 
                 self.description="Integration Test Mock Progress"
                 self.last_print_n = 0
                 # Add the missing attribute expected by TrainingLoop
                 self.progress_bar = None # Can be None or a mock tqdm object if needed
                 
             def start(self): self.n = 0
             # Accept arbitrary kwargs to handle calls like update(step=..., loss=...)
             def update(self, n=1, **kwargs): self.n += n 
             def stop(self): pass
             def close(self): pass 
             def get_value(self, key): return 0.0 
             def step(self, advanced: int = 1, **kwargs): self.update(advanced)
             def set_postfix(self, ordered_dict: dict | None = None, refresh: bool = True, **kwargs): pass
             def set_description(self, desc: str, refresh: bool = True): self.description = desc
             def moveto(self, n: int): self.n = n 
             def __enter__(self): return self
             def __exit__(self, exc_type, exc_value, tb): self.close()

        mock_progress = MockProgressTracker(total_steps=cfg.training.max_steps)
        mock_progress.progress_bar = None # Manually add attribute
        initial_param_sum = sum(p.sum().item() for p in model.parameters() if p.requires_grad)

        # Initialize TrainingLoop's internal step counter if necessary
        training_loop.global_step = 0 
        
        training_output = training_loop.train_epoch(
            trainer=mock_trainer, 
            current_epoch=0, 
            global_step=0, 
            progress=mock_progress
        )

        print(f"Training run finished. Output: {training_output}")

        # --- 4. Assertions ---
        final_global_step = training_output.get('final_global_step', -1) # Check returned value
        assert final_global_step == cfg.training.max_steps, \
               f"Expected final step {cfg.training.max_steps}, got {final_global_step}"

        assert mock_progress.n == cfg.training.max_steps, \
                f"Progress tracker steps {mock_progress.n} != max_steps {cfg.training.max_steps}"


        assert 'loss' in training_output, f"'loss' not found in train_epoch output: {training_output}"

        assert isinstance(training_output['loss'], float)
        assert training_output['loss'] > 0.0 # Sanity check loss

        final_param_sum = sum(p.sum().item() for p in model.parameters() if p.requires_grad)
        assert initial_param_sum != final_param_sum, \
               f"Model parameters did not change. Initial sum: {initial_param_sum}, Final sum: {final_param_sum}"

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        pytest.fail(f"Training run failed: {e}\n{trace}")

    print("--- Core pipeline test passed. ---")

# Add more integration tests here for other major features (e.g., evaluation, checkpointing) 