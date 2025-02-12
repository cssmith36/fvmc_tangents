import fvmc

cfg = fvmc.config.default()

cfg.verbosity = "INFO"
cfg.multi_device = True
cfg.sample.burn_in = 1
cfg.optimize.iterations = 10
cfg.log.stat_every = 1

if __name__ == "__main__":
    fvmc.train.main(cfg)
