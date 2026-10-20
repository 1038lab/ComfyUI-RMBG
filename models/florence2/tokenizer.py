# Based on code from ComfyUI-Florence2 by Jukka Seppänen (kijai), MIT License
import os
from tokenizers import Tokenizer, AddedToken


class Florence2Tokenizer:
    def __init__(self, model_path):
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self._add_special_tokens()

        # Standard token IDs
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2

    def _add_special_tokens(self):
        """Add Florence2-specific special tokens."""
        special_tokens = ['<od>', '</od>', '<ocr>', '</ocr>']
        special_tokens += [f'<loc_{x}>' for x in range(1000)]
        special_tokens += [
            '<cap>', '</cap>', '<ncap>', '</ncap>', '<dcap>', '</dcap>',
            '<grounding>', '</grounding>', '<seg>', '</seg>', '<sep>',
            '<region_cap>', '</region_cap>', '<region_to_desciption>',
            '</region_to_desciption>', '<proposal>', '</proposal>',
            '<poly>', '</poly>', '<and>'
        ]
        added = [AddedToken(t, special=True) for t in special_tokens]
        self.tokenizer.add_special_tokens(added)

    def encode(self, text):
        """Encode text to token ids. Returns dict with 'input_ids' tensor."""
        import torch
        encoding = self.tokenizer.encode(text)
        return {"input_ids": torch.tensor([encoding.ids], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token ids to text."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids_batch, skip_special_tokens=False):
        """Decode a batch of token id sequences."""
        results = []
        if hasattr(token_ids_batch, 'tolist'):
            token_ids_batch = token_ids_batch.tolist()
        for ids in token_ids_batch:
            results.append(self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))
        return results

    @property
    def all_special_tokens(self):
        """Return set of all special token strings. Needed by post-processor."""
        added = self.tokenizer.get_added_tokens_decoder()
        tokens = set()
        for token_obj in added.values():
            tokens.add(str(token_obj))
        # Also add the built-in ones
        tokens.update({'<s>', '</s>', '<pad>', '<unk>', '<mask>'})
        return tokens

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
