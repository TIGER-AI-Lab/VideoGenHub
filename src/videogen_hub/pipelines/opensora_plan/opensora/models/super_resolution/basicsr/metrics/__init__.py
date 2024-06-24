from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY

from videogen_hub.pipelines.opensora_plan.opensora.models.super_resolution.basicsr.metrics.psnr_ssim import \
    calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr', 'calculate_ssim']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
