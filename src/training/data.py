import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray[np.integer] | np.memmap,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}.")

    num_tokens = len(dataset)
    num_start_positions = num_tokens - context_length
    if num_start_positions <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens to form "
            "input/target pairs."
        )

    start_indices = np.random.randint(0, num_start_positions, size=batch_size)
    x = np.stack(
        [dataset[start : start + context_length] for start in start_indices],
        axis=0,
    )
    y = np.stack(
        [dataset[start + 1 : start + context_length + 1] for start in start_indices],
        axis=0,
    )

    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return x_tensor, y_tensor


def load_token_dataset(
    path: str,
    *,
    mmap: bool = True,
) -> npt.NDArray[np.integer] | np.memmap:
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)
