from ml_collections import ConfigDict


def default() -> ConfigDict:
    return ConfigDict({
        
        "seed": 0,

        "restart": {
            "params": None,
            "states": None,
        },

        "system": {
            "ions": [[0., 0., 0.], [0., 0., 1.]],
            "elems": [1., 1.],
            "charge": 0,
            "spin": None,
        },

        "ansatz": {},

        "sample": {
            "size": 2048,
            "chains": None,
            "burn_in": 0,
            "sampler": {
                "name": "mcmc",
            }
        },

        "optimize": {
            "iterations": 100_000,
            "lr": {},
            "energy_clipping": 5.,
            "optimizer": {
                "name": "kfac",
            }
        },

        "log": {
            "stat_path": "tbdata/",
            "stat_every": 100,
            "ckpt_path": "checkpoint.pkl",
            "ckpt_every": 100,
            "ckpt_keep": 5,
        },

    })