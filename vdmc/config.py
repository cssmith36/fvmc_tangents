from ml_collections import ConfigDict


def default() -> ConfigDict:
    return ConfigDict({
        
        "seed": 0,

        "verbosity": "WARNING",

        "multi_device": False,

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

        #TODO make a default ansatz parameters
        "ansatz": {},

        "sample": {
            "size": 2048,
            "chains": None,
            "burn_in": 100,
            "sampler": {
                "name": "mcmc",
            }
        },

        "optimize": {
            "iterations": 100_000,
            "lr": {},
            "loss": {
                "energy_clipping": 5.,
                "grad_stablizing": True,
            },
            "optimizer": {
                "name": "kfac",
            },
        },

        "log": {
            "stat_path": "tbdata/",
            "stat_every": 100,
            "ckpt_path": "checkpoint.pkl",
            "ckpt_every": 100,
            "ckpt_keep": 1,
            "hpar_path": "hparams.yaml",
        },

    })