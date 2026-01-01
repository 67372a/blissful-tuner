import torch
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union
)

# Inspired by Grokking at the Edge of Numerical Stability (https://arxiv.org/abs/2501.04697)
def mse_loss(predictions, targets):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = differences ** 2
    return loss

def smooth_l2_loss(predictions, targets, beta=1.0):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = ((differences**2) / differences.abs().add(beta)).pow(2)
    return loss

def pseudo_huber_loss(predictions, targets, delta=1.0):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    # Compute the loss
    loss = delta**2 * (torch.sqrt(1 + (differences / delta)**2) - 1)
    return loss

def scaled_quadratic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0
) -> torch.Tensor:
    r = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = (r / delta)**2
    return loss

def smooth_l1_loss(predictions, targets, beta=1.0):
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    condition = diff < beta
    
    # Where diff < beta, use quadratic form
    quadratic = 0.5 * diff.pow(2) / beta
    
    # Where diff >= beta, use linear form
    linear = diff - 0.5 * beta
    
    # Combine the two parts based on the condition
    loss = torch.where(condition, quadratic, linear)
    
    return loss


def huber_loss(predictions, targets, delta=1.0):
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    abs_error = torch.abs(diff)
    
    # For small errors (≤ delta): use squared error (L2)
    quadratic = 0.5 * diff.pow(2)
    
    # For large errors (> delta): use modified absolute error (L1)
    linear = delta * (abs_error - 0.5 * delta)
    
    # Combine both parts
    loss = torch.where(abs_error <= delta, quadratic, linear)

    return loss

def l1_loss(predictions, targets):
    loss = torch.abs(predictions - targets)
    return loss


def conditional_loss(
    model_pred: torch.Tensor, 
    target: torch.Tensor, 
    loss_type: str = "l2", 
    delta_beta: float = None,
):
    loss_type_lower = loss_type.lower()

    # Make loss float64
    model_pred = model_pred.to(torch.float64)
    target = target.to(torch.float64)

    if loss_type_lower == "l2":
        loss = mse_loss(model_pred, target)
    elif loss_type == "l1":
        loss = l1_loss(model_pred, target)
    elif loss_type == "pseudo_huber":
        loss = pseudo_huber_loss(model_pred, target, delta=delta_beta)
    elif loss_type == "huber":
        loss = huber_loss(model_pred, target, delta=delta_beta)
    elif loss_type == "smooth_l1":
        loss = smooth_l1_loss(model_pred, target, beta=delta_beta)
    elif loss_type == "scaled_quadratic":
        loss = scaled_quadratic_loss(model_pred, target, delta=delta_beta)
    elif loss_type == "smooth_l2":
        loss = smooth_l2_loss(model_pred, target, delta=delta_beta)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss
