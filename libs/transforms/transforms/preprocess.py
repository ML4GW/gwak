import torch


# precompute_embeddings
def frequency_cos_similarity(batch, mode):
    H = torch.fft.rfft(batch[:, 0, :], dim=-1)
    L = torch.fft.rfft(batch[:, 1, :], dim=-1)
    numerator = torch.sum(H * torch.conj(L), dim=-1)
    norm_H = torch.linalg.norm(H, dim=-1)
    norm_L = torch.linalg.norm(L, dim=-1)
    rho_complex = numerator / (norm_H * norm_L + 1e-8)

    if mode == "real":
        return torch.real(rho_complex).unsqueeze(-1)
    if mode == "real_imag":
        score_1 = torch.real(rho_complex).unsqueeze(-1)
        score_2 = torch.imag(rho_complex).unsqueeze(-1)
        score = torch.cat([score_1, score_2], dim=1)
        return score
    if mode == "abs":
        return torch.abs(rho_complex).unsqueeze(-1)





# fm_models
# combine_model
# plots

