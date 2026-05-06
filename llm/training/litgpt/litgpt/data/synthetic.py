from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from litgpt.data.base import DataModule
from litgpt.tokenizer import Tokenizer


class SyntheticTokenDataset(Dataset):
    def __init__(self, num_samples: int, block_size: int, vocab_size: int, seed: int) -> None:
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + index)
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.block_size,),
            generator=generator,
            dtype=torch.int64,
        )


@dataclass
class Synthetic(DataModule):
    train_samples: int = 10000
    val_samples: int = 1000
    seed: int = 42
    num_workers: int = 4

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SyntheticTokenDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SyntheticTokenDataset] = field(default=None, init=False, repr=False)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1

    def setup(self, stage: str = "") -> None:
        vocab_size = self._vocab_size()
        self.train_dataset = SyntheticTokenDataset(
            num_samples=self.train_samples,
            block_size=self.max_seq_length,
            vocab_size=vocab_size,
            seed=self.seed,
        )
        self.val_dataset = SyntheticTokenDataset(
            num_samples=self.val_samples,
            block_size=self.max_seq_length,
            vocab_size=vocab_size,
            seed=self.seed + self.train_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None. Please provide a valid tokenizer_dir for synthetic data generation.")
        return self.tokenizer.vocab_size
