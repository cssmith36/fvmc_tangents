from ml_collections import ConfigDict


def default() -> ConfigDict:
    return ConfigDict({

        "seed": 0,

        "verbosity": "WARNING",

        "multi_device": False,

        "quantum_nuclei": False,

        "restart": {
            "params": None,
            "chains": None,
            "states": None,
        },

        "system": system_placeholder(),

        "penalty_method": penalty_method(),

        #TODO make a default ansatz parameters
        "ansatz": {"freeze_params":False,
                   "split_params":None},

        "sample": {
            "size": 2048,
            "chains": None,
            "burn_in": 100,
            "sampler": "mcmc",
            "mcmc": {},
            "mala": {},
            "hmc": {},
            "black": {},
            "adaptive": None,
        },

        "loss": {
            "ke_kwargs": {},
            "pe_kwargs": {},
            "mini_batch": None,
            "energy_clipping": 5.,
            "center_shifting": True,
            "clip_from_median": True,
        },

        "optimize": {
            "iterations": 100,
            "lr": {},
            "optimizer": "kfac",
            "grad_clipping": None, # will not work for kfac
            "kfac": {},
            "sr": {},
        },

        "log": {
            "stat_path": "data.txt",
            "stat_every": 1,
            "stat_keep": 2, # only back up once, avoid clutter
            "ckpt_path": "checkpoint.pkl",
            "ckpt_every": 100,
            "ckpt_keep": 2, # keep current and last check point
            "dump_path": "trajdump.npy",
            "dump_every": 0, # change to positive number to dump
            "dump_keep": 2, # only back up once, avoid clutter
            "hpar_path": "hparams.yaml",
            "use_tensorboard": False,
            "tracker_path": "tbdata/",
        },

    }, type_safe=False)


def default_tangents() -> ConfigDict:
    return ConfigDict({

        "verbosity": "WARNING",

        "system": system_placeholder(),

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
            "adaptive": None,
        },

        "eval_tangents": eval_tangents(),

        "restart": {
            "params": "checkpoint.pkl",
            "chains": None,
            "states": None,
        },

    }, type_safe=False)


def eval_tangents() -> ConfigDict:
    return ConfigDict({
            "ke_kwargs": {"partition_size": None},
            "pe_kwargs": {},
            "reweighting": False,
            "iterations": 100,
            "dev_2": False,
            "eval_obs": False,
            "evecs": None,
            "save_folder": "tangents/",
            "dense_mode": False,
        }, type_safe=False)

def penalty_method() -> ConfigDict:
    return ConfigDict({
        "gs_ansatz": {}
    })


def system_placeholder() -> ConfigDict:
    return ConfigDict({
        "nuclei": [[0., 0., 0.], [0., 0., 1.]],
        "elems": [1., 1.],
        "charge": 0,
        "spin": None,
        "cell": None,
    }, type_safe=False)

