import torch
from config import TRAIN_CLASS_FREQ, WEIGHT_MAX_CAP_LOG_INVERSE_FREQUENCY


def get_inverse_log_weights(device, k=1.02):

    freq = torch.tensor(TRAIN_CLASS_FREQ, dtype=torch.float32, device=device)

    weights = 1.0 / torch.log(k + freq)

    #
    weights = torch.clamp(
        weights,
        max=WEIGHT_MAX_CAP_LOG_INVERSE_FREQUENCY
    )

    weights = weights / weights.sum() * len(freq)

    return weights
