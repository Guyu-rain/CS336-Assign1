from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class Adam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 拿到超参数
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]

            # 拿到每个参数
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients.")
                # 拿到参数状态
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    m = state["m"]
                    v = state["v"]

                    state["step"] += 1
                    t = state["step"]

                    # Update biased first moment estimate
                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                    # Update biased second raw moment estimate
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # Bias correction
                    bias_correction1 = 1.0 - beta1 ** t
                    bias_correction2 = 1.0 - beta2 ** t

                    m_hat = m / bias_correction1
                    v_hat = v / bias_correction2

                    # Parameter update
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

            return loss

class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer.

    AdamW decouples weight decay from the gradient-based Adam update.

    Update rule:
        g_t = grad

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t * g_t)

        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)

        theta <- theta - lr * weight_decay * theta
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional callable that reevaluates the model
                     and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step_t = state["step"]

                # 1) Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # 2) Update first and second moments using pure gradient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3) Bias correction
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t

                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                # 4) Parameter update
                denom = exp_avg_sq_hat.sqrt().add_(eps)
                p.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss

def learning_rate_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    anneal_steps: int,
) -> float:
    if step < 0:
        raise ValueError(f"`step` must be >= 0, got {step}.")
    if max_lr < 0.0:
        raise ValueError(f"`max_lr` must be >= 0, got {max_lr}.")
    if min_lr < 0.0:
        raise ValueError(f"`min_lr` must be >= 0, got {min_lr}.")
    if max_lr < min_lr:
        raise ValueError(
            f"`max_lr` must be >= `min_lr`, got max_lr={max_lr}, min_lr={min_lr}."
        )
    if warmup_steps < 0:
        raise ValueError(f"`warmup_steps` must be >= 0, got {warmup_steps}.")
    if anneal_steps < 0:
        raise ValueError(f"`anneal_steps` must be >= 0, got {anneal_steps}.")
    if anneal_steps < warmup_steps:
        raise ValueError(
            f"`anneal_steps` must be >= `warmup_steps`, got "
            f"anneal_steps={anneal_steps}, warmup_steps={warmup_steps}."
        )

    if warmup_steps == 0:
        if step <= anneal_steps:
            # Special case: if anneal_steps == 0, schedule immediately lands at min_lr
            if anneal_steps == 0:
                return min_lr

            progress = step / anneal_steps  # in [0, 1]
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + cosine_decay * (max_lr - min_lr)
        else:
            return min_lr

    if step < warmup_steps:
        return (step / warmup_steps) * max_lr

    if step <= anneal_steps:
        progress = (step - warmup_steps) / (anneal_steps - warmup_steps)  # [0, 1]
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + cosine_decay * (max_lr - min_lr)

    return min_lr

def gradient_clipping(
    params: list[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    """
    将所有参数看成一个整体计算 l2-norm，如果超过阈值，做 gradient_clipping
    """
    params = list(params)

    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return

    total_norm_sq = sum(torch.sum(g.detach() * g.detach()) for g in grads)
    total_norm = torch.sqrt(total_norm_sq)

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
