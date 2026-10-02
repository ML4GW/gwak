def cohmode_to_dim(coh_mode):

    coh_mode_to_dim_dict = {
        "real": 1,
        "abs": 1,
        "random": 1,
        "half": 1,
        "random2": 2,
        "real_imag": 2,
    }

    return coh_mode_to_dim_dict[coh_mode]