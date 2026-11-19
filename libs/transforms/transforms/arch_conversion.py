def cohmode_to_dim(coh_mode):

    coh_mode_to_dim_dict = {
        "muted": 0,
        "real": 1,
        "abs": 1,
        "random": 1,
        "half": 1,
        "random2": 2,
        "real_imag": 2,
        "corr15": 15,
        "corr21": 21,
        "selected-9": 9,
    }

    return coh_mode_to_dim_dict[coh_mode]
