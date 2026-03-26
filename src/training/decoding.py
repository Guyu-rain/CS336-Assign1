from collections.abc import Sequence

import torch

from src.tokenizer.core import BPE


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative_probs <= top_p
        keep_mask[..., 0] = True

        filtered_probs = sorted_probs * keep_mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        sampled_index = torch.multinomial(filtered_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_index)
        return int(next_token.item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt_tokens: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    context_length: int | None = None,
) -> list[int]:
    tokens = list(prompt_tokens)
    model.eval()
    model.to(device)

    for _ in range(max_new_tokens):
        model_input = tokens
        if context_length is not None:
            model_input = model_input[-context_length:]

        x = torch.tensor([model_input], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_token = sample_from_logits(
            logits,
            temperature=temperature,
            top_p=top_p,
        )
        tokens.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokens


def generate_text(
    model: torch.nn.Module,
    tokenizer: BPE,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    context_length: int | None = None,
    end_of_text_token: str = "<|endoftext|>",
) -> dict[str, object]:
    prompt_tokens = tokenizer.encode(prompt)
    eos_token_id = tokenizer.special_token_to_id.get(end_of_text_token)
    output_tokens = decode(
        model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        context_length=context_length,
    )
    generated_tokens = output_tokens[len(prompt_tokens) :]
    generated_text = tokenizer.decode(output_tokens)
    return {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "output_tokens": output_tokens,
        "text": generated_text,
    }
