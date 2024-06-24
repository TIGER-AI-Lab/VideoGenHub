from videogen_hub.pipelines.opensora_plan.opensora.models.ae.imagebase import imagebase_ae, imagebase_ae_stride, \
    imagebase_ae_channel
from videogen_hub.pipelines.opensora_plan.opensora.models.ae.videobase import (
    VQVAEConfiguration,
    VQVAEModel,
    VQVAETrainer,
    CausalVQVAEModel,
    CausalVQVAEConfiguration,
    CausalVQVAETrainer
)
from videogen_hub.pipelines.opensora_plan.opensora.models.ae.videobase import videobase_ae, videobase_ae_stride, \
    videobase_ae_channel

ae_stride_config = {}
ae_stride_config.update(imagebase_ae_stride)
ae_stride_config.update(videobase_ae_stride)

ae_channel_config = {}
ae_channel_config.update(imagebase_ae_channel)
ae_channel_config.update(videobase_ae_channel)

def getae(args):
    """deprecation"""
    ae = imagebase_ae.get(args.ae, None) or videobase_ae.get(args.ae, None)
    assert ae is not None
    return ae(args.ae)

def getae_wrapper(ae):
    """deprecation"""
    ae = imagebase_ae.get(ae, None) or videobase_ae.get(ae, None)
    assert ae is not None
    return ae