# Data (`data/`)

## Purpose

This directory contains all datasets used for training, validation, and testing models within the Craft framework.

## Structure

-   `data/`
    -   `raw/`: Original, immutable data files as downloaded or received. (Consider adding `data/raw/*` to `.gitignore` if files are large).
    -   `processed/`: Data transformed into a format suitable for model consumption (e.g., tokenized text, pre-computed features).

## Guidelines

- Place data processing scripts in `scripts/data/` or implement data handling logic within `src/data/`.
- Document the source and preprocessing steps for each dataset.
- Ensure processed data formats are compatible with the data loaders defined in `src/data/`.