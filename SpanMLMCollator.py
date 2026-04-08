%%writefile SpanMLMCollator.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
import json
import multiprocessing

class ConfigJson():
    def __init__(self, **kwargs):
        # Assign attributes from keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        return cls(**data)



class SpanMLMCollator:

    # Define Init
    # Most of the things are just stored in the config lwk
    # Imma just pull from here
    def __init__(self, config = None, tokenizer = None, mlm_probability = None, mlm_use_span_masking = None, mlm_span_length = None):
        
        # Defaults
        self.mlm_probability = 0.15
        self.mlm_use_span_masking = False
        self.mlm_span_length = 3
        self.tokenizer = None

        # Override with Config (if provided)
        if (config is not None):
            if (isinstance(config, str)):
                try:
                    config = ConfigJson.from_json(config)
                except Exception as e:
                    raise ValueError(f"Blud was not a json. Either some sort of dictionary ahhh config or .json. Error: {e}")

            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            self.mlm_probability = config.mlm_probability
            self.mlm_use_span_masking = config.mlm_use_span_masking
            self.mlm_span_length = config.mlm_span_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        if mlm_probability is not None:
            self.mlm_probability = mlm_probability
        if mlm_use_span_masking is not None:
            self.mlm_use_span_masking = mlm_use_span_masking
        if mlm_span_length is not None:
            self.mlm_span_length = mlm_span_length
            
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None. Either pass in the config or tokenizer")

        # AI HELP:
        # Create vocab for legal words to replace
        valid_ids = [i for i in range(len(self.tokenizer)) if i not in self.tokenizer.all_special_ids]
        self.valid_vocab = torch.tensor(valid_ids)

    # __call__() function returning a DataLoader with Span_masking
    def __call__(self, data):

        # Convert List of Dictionaries into 1 large tensor
        # inside, convert extract input_ids from dictionary
        batch_input_ids = [d["input_ids"] for d in data]
        input_ids = torch.tensor(batch_input_ids)
        labels = input_ids.clone()

        # Set mlm_prob
        mlm_prob = self.mlm_probability
        if (self.mlm_use_span_masking):
            mlm_prob /= self.mlm_span_length

        # Create the Masking Tensor
        masked_tensor = torch.full(input_ids.shape, mlm_prob, dtype=torch.float32)


        # HUH???


        # =====================================================================
        # THE GUARD STEP: Protect special tokens from being masked
        # =====================================================================
        # 1. Ask the tokenizer which tokens in each row are special (CLS, SEP, PAD)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(seq, already_has_special_tokens=True) 
            for seq in input_ids.tolist()
        ]
        
        # 2. Convert that nested list into a PyTorch Boolean tensor
        special_tokens_map = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # 3. Overwrite the probability in masked_tensor to 0.0 wherever special_tokens_map is True
        masked_tensor.masked_fill_(special_tokens_map, value=0.0)
        # =====================================================================


        # Apply Bernoulli to get 1s and 0s for the masks
        masked_tensor = torch.bernoulli(masked_tensor).bool()

        temp_clone = masked_tensor.clone()
        # Apply rolling for span masking
        if (self.mlm_use_span_masking):
            for i in range (1, self.mlm_span_length):
                rolled_tensor = torch.roll(temp_clone, shifts = i)
                rolled_tensor[:,:i] = False
                masked_tensor = masked_tensor | rolled_tensor

        # Mask all untampered tokens with -100
        labels[~masked_tensor] = -100
        
        # Create new Tensor w/ random values from 0-1
        type_tensor = torch.rand(input_ids.shape)

        # Create 80% mask_token_tensor
        mask_token_tensor = (type_tensor <= .8) & masked_tensor

        # Create 10% corrupted_token_tensor
        corrupted_token_tensor = (type_tensor > .8) & (type_tensor <= .9) & masked_tensor

        # Apply Mask tokens to input_ids
        input_ids[mask_token_tensor] = self.tokenizer.mask_token_id

        # LOOK HERE ##################################################
        
        # Apply corrupted tokens to input_ids
        num_to_replace = corrupted_token_tensor.sum().item()

        # Generate random indices for the valid_vocab_tensor
        indices = torch.randint(0,len(self.valid_vocab), (num_to_replace,))

        # Take the words form valid_vocab
        random_words = self.valid_vocab[indices]

        # Apply the words to the input_ids
        input_ids[corrupted_token_tensor] = random_words

        # ############################################################

        # Return Dictionary
        return {"input_ids": input_ids, "labels": labels}