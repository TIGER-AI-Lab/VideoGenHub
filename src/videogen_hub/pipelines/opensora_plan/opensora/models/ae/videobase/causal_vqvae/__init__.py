from einops import rearrange
from torch import nn

from videogen_hub.pipelines.opensora_plan.opensora.models.ae.videobase.causal_vqvae.configuration_causalvqvae import \
    CausalVQVAEConfiguration
from videogen_hub.pipelines.opensora_plan.opensora.models.ae.videobase.causal_vqvae.modeling_causalvqvae import \
    CausalVQVAEModel
from videogen_hub.pipelines.opensora_plan.opensora.models.ae.videobase.causal_vqvae.trainer_causalvqvae import \
    CausalVQVAETrainer


class CausalVQVAEModelWrapper(nn.Module):
    def __init__(self, ckpt):
        super(CausalVQVAEModelWrapper, self).__init__()
        self.vqvae = CausalVQVAEModel.load_from_checkpoint(ckpt)
    def encode(self, x):  # b c t h w
        x = self.vqvae.pre_vq_conv(self.vqvae.encoder(x))
        return x
    def decode(self, x):
        vq_output = self.vqvae.codebook(x)
        x = self.vqvae.decoder(self.vqvae.post_vq_conv(vq_output['embeddings']))
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x