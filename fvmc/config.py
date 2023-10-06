from ml_collections import ConfigDict


def default() -> ConfigDict:
    return ConfigDict({

        "seed": 0,

        "verbosity": "WARNING",

        "multi_device": False,

        "fully_quantum": False,

        "restart": {
            "params": None,
            "states": None,
        },

        "system": {
            "nuclei": [[0., 0., 0.], [0., 0., 1.]],
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
            "sampler": "mcmc",
            "mcmc": {},
            "mala": {},
            "hmc": {},
            "black": {},
        },

        "loss": {
            "energy_clipping": 5.,
            "center_shifting": True,
            "clip_from_median": True,
        },

        "optimize": {
            "iterations": 100_000,
            "lr": {},
            "optimizer": "kfac",
            "grad_clipping": None, # will not work for kfac
            "kfac": {},
            "sr": {},
        },

        "log": {
            "stat_path": "data.txt",
            "stat_every": 100,
            "stat_keep": 2, # only back up once, avoid clutter
            "ckpt_path": "checkpoint.pkl",
            "ckpt_every": 100,
            "ckpt_keep": 2, # keep current and last check point
            "dump_path": "trajdump.npy",
            "dump_every": 0, # change to positive number to dump
            "dump_keep": 2, # only back up once, avoid clutter
            "hpar_path": "hparams.yaml",
            "use_tensorboard": True,
            "tracker_path": "tbdata/",
        },

    }, type_safe=False)
