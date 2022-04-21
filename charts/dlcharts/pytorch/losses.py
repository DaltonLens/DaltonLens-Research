import torch

def rgb_variance_loss (rgb: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
    assert rgb.dim() == 4 and fg_mask.dim() == 3
    # Do everything as B,C,N to keep stats simpler to write
    B = rgb.size()[0]
    rgb = rgb.view(B,3,-1) # B,C,N
    fg_mask = fg_mask.view(B,1,-1) # B,1,N
    n_fg_per_channel = fg_mask.sum(dim=-1) # B,C

    bg_mask = torch.logical_not(fg_mask)
    bg_mask_rgb = bg_mask.expand (B,3,-1) # B,3,N
    masked_rgb = rgb.clone()
    masked_rgb[bg_mask_rgb] = 0.0 # B,C,N
    sums = masked_rgb.sum(dim=-1) # B,C
    means = torch.divide(sums, n_fg_per_channel) # B,C
    means = means.unsqueeze(-1) # B,C,1 for broadcasting
    diff_sqr = (masked_rgb - means).square() # B,C,N
    diff_sqr[bg_mask_rgb] = 0.0
    variances = torch.divide (diff_sqr.sum(dim=-1), n_fg_per_channel) # B,C
    variances = variances.mean(dim=-1) # B
    # If we don't have enough foreground, skip the reg_loss
    variances[n_fg_per_channel.view(-1) < 64] = 0.0
    assert not torch.isnan(variances).any()
    reg_loss = torch.mean (variances)
    return reg_loss
