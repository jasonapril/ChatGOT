# Data Pipeline

This document describes the data pipeline for the ChatGoT project.

## Directory Structure

The project uses the following data directory structure:

```
data/
├── got/                    # Game of Thrones data
│   ├── got_s01e01.txt      # Individual episode transcripts 
│   ├── got_s01e02.txt      # ...more episodes
│   └── game_of_thrones.txt # Combined file for training
├── processed/              # Processed data ready for training
│   ├── got_char.bin        # Character-level tokenized data
│   └── got_word.bin        # Word-level tokenized data (if used)
└── other_sources/          # Other potential data sources
    └── ...
```

## Data Processing Pipeline

The data processing pipeline consists of the following steps:

1. **Raw Data Collection**: Individual episode transcript files are stored in `data/got/`.

2. **Data Combination**: The script `scripts/process_got_data.py` combines all episode files into a single large text file (`game_of_thrones.txt`) which is used for training.

3. **Tokenization and Preparation**: The training scripts (`train_with_samples.py`) perform character-level tokenization on-the-fly. If needed, pre-tokenized versions can be saved in the `data/processed/` directory.

## Running the Pipeline

To process the Game of Thrones data:

```bash
# Combine episode transcripts into a single file
python scripts/process_got_data.py

# Optional: Create pre-tokenized data
# python scripts/preprocess_data.py --input_file data/got/game_of_thrones.txt --output_file data/processed/got_char.bin
```

## Adding New Data Sources

To add new data sources:

1. Create a new directory under `data/` for your source
2. Add raw text files
3. Modify or create a new processing script based on `process_got_data.py`
4. Update the training configuration to use the new data

## Data Statistics

- Game of Thrones Dataset:
  - Episodes: 73 (Seasons 1-8)
  - Total size: ~2.5MB of text
  - Character count: Varies depending on preprocessing

## Naming Conventions

- **Raw data files**: Descriptive names indicating source and organization
  - Example: `got_s01e01.txt` (Season 1, Episode 1)
- **Processed data files**: Indicate source and tokenization method
  - Example: `got_char.bin` (Character-level tokenization) 