import pickle
import os
import json

def check_vocab_size():
    # Check character-level data
    char_data_path = 'data/processed/got/char/game_of_thrones_train.pkl'
    if not os.path.exists(char_data_path):
        print(f"Character-level file not found: {char_data_path}")
    else:
        print("\nCharacter-Level Data:")
        with open(char_data_path, 'rb') as f:
            data = pickle.load(f)
            print(f"  Vocabulary size: {data.get('vocab_size')}")
            print(f"  Number of tokens: {len(data.get('token_ids', []))}")
            print(f"  Number of unique characters: {len(data.get('chars', []))}")
    
    # Check subword-level data
    # Note: Subword data files themselves might not exist yet, 
    # the script currently only checks for a hypothetical train file.
    subword_data_path = 'data/processed/got/subword/game_of_thrones_train.pkl' 
    if not os.path.exists(subword_data_path):
        print(f"\nSubword-level data file not found: {subword_data_path}")
    else:
        print("\nSubword-Level Data:")
        with open(subword_data_path, 'rb') as f:
            data = pickle.load(f)
            print(f"  Vocabulary size: {data.get('vocab_size')}")
            print(f"  Tokenizer name: {data.get('tokenizer_name')}")
            print(f"  Number of tokens: {len(data.get('token_ids', []))}")
    
    # Check the tokenizer file
    tokenizer_path = 'data/processed/got/subword/tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print(f"\nTokenizer file not found: {tokenizer_path}")
    else:
        print("\nTokenizer Data:")
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            print(f"  Model type: {tokenizer_data.get('model', {}).get('type')}")
            vocab_data = tokenizer_data.get('model', {}).get('vocab', {})
            print(f"  Vocabulary size: {len(vocab_data)}")
            print(f"  Special tokens: {tokenizer_data.get('added_tokens', [])}")

if __name__ == "__main__":
    check_vocab_size() 