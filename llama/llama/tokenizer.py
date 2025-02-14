# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from logging import getLogger
from typing import List
import os
from transformers import AutoTokenizer


logger = getLogger()


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pass
        # reload tokenizer
        #self.sp_model = SentencePieceProcessor(model_file=model_path)
        #logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        #self.n_words: int = self.sp_model.vocab_size()
        #self.bos_id: int = self.sp_model.bos_id()
        #self.eos_id: int = self.sp_model.eos_id()
        #self.pad_id: int = self.sp_model.pad_id()
        #logger.info(
        #    f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        #)
        #assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.n_words = tokenizer.vocab_size
        self.bos_id =  tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        encoded_input =  self.tokenizer.encode(s, add_special_tokens=False,return_tensors=None)
        # Add BOS token if specified
        if bos:
                encoded_input = [ self.bos_id] + encoded_input 
        # Add EOS token if specified
        if eos:
                encoded_input = encoded_input + [self.eos_id]
                
        return encoded_input

    def decode(self, t: List[int]) -> str:

        return self.tokenizer.decode(t)
