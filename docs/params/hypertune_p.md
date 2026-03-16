# hypertune_params

`hypertune_params` determines the behavior of hypertune (hyperparameter tuning) mode and the range of optimizable parameters
Hypertune optimizable parameters are specified in 'tune_params', these will override the corresponding values in 'exp_params' but follows the exact same definition and unit.
Set 'state' to true to enable hypertuning that parameter. 'min', 'max', and 'step' defines the search space and sampling rate. For example, conv_angle with min:24, max:26, step:0.1 would have a search range between 24 and 26 with 0.1 minimal spacing
It's a better practice to limit your hypertune optimization with no more than 4 parameters simultaneously, and some hierachical strategy (i.e., optimizing dz first, then scale with rotation, then all 4 scan_affine, lastly object tilts) could be much faster / stable
For typical dataset with sufficient dose, both conv_angle and defocus can be automatically handled by probe optimization. However for low dose dataset (like 1e2 e-/DP), some hypertune of probe parameters could be necessary


```yaml
hypertune_params : {
    'if_hypertune'   : false, # type: boolean. Default is false. Set to true to run PtyRAD in hypertune (hyperparameter optimization) mode. This is the main switch for hypertune mode so none of the settings in 'hypertune_params' would take effect if 'if_hypertune' is false.
    'collate_results': true, # type: boolean. Default is true. Set to true to collect results/figs specified under 'recon_params' from each hypertune trial to 'output_dir/<HYPERTUNE_FOLDER>' and name them by error metrics followed by trial parameters. This provides a convenient way to quickly examine the hypertune results inside the hypertune folder under the main output directory 'output_dir'. This is an independent control than 'SAVE_ITERS' and will save by the end of the 'NITER' or when the trial is pruned.
    'append_params'  : true, # type: boolean. Default is true. Set to true to append the hyperparameter name/values to the file names of collated result. If set to false, only the error, trial ID, and iter would be preserved. Set this to false to reduce the length of the output file name. The exact parameter values are stored in the .sqlite3 database file.
    'n_trials'       : 5, # type: int. Number of hypertune trials. Each trial is a separate PtyRAD reconstruction with a set of optimizable parameter values (i.e., a configuration). Note that when the hypertune mode is run in parallel (i.e., multiple jobs on multiple GPU nodes accesing the same database/study), each job will run for 'n_trials' times. So submiting 5 jobs with 'n_trials': 200 will get you a total of 1000 trials in the database/study
    'timeout'        : null, # type: float. Stop "study" after the given number of second(s). Null represents no limit in terms of elapsed time. The study continues to create trials until the number of trials reaches n_trials, timeout period elapses, stop() is called or, a termination signal such as SIGTERM or Ctrl+C is received.
    'sampler_params' : {'name': 'TPESampler', 'configs': {'multivariate': true, 'group': true, 'constant_liar': true}}, # type: dict. Sampler is the optimization algorithm used for hyperparameter tuning. See https://optuna.readthedocs.io/en/stable/reference/samplers/index.html for more details.
    'pruner_params'  : {'name': 'HyperbandPruner', 'configs': {'min_resource': 5, 'reduction_factor': 2}}, # type: null, dict. Pruning is early termination of unpromising trials to save computation budget. Set to False to disable pruning (i.e., no early termination). The recommended prunner is HyperbandPruner, see Optuna document for more details
    'storage_path'   : 'sqlite:///hypertune.sqlite3', # type: str. Path to the SQLite database file (i.e., sotrage) as 'sqlite:///<DATABASE_FILENAME>.sqlite3'. This database file keeps the record of hypertune runs and will be automatically created with new hypertube run. If you're doing distributed (e.g. multiple GPU nodes) hypertune by submitting this params file multiple times, then all the GPU worker would be accessing this database to be aware of each other's progress
    'study_name'     : 'tBL_WSe2', # type: str. Name of the hypertune record (i.e., study). Hypertune of different dataset or different search space (i.e., different optimizable parameters) are encouraged to use different study name or even separate database file
    'error_metric'   : 'loss', # type: str. Either use 'loss' or 'contrast'. The current suggested approach is to use 'loss' for rough optimization, while switch to 'contrast' with loaded reconstructed object/probe/pos to refine remaining hyperparameters. 'contrast' is simply calculated by std(img)/mean(std) to reflect reconstruction quality of the object because 'loss' doesn't correlate that well. Note that `contrast` doesn't necessarily change monotonically with iterations, especially at early iterations so you may want to disable pruning and set NITER carefullly.
    'tune_params'    : { # See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial for the syntax to configure the search space ranges and types
        'optimizer'  : {'state': false, 'suggest': 'cat', 'kwargs': {'choices': ["Adam", "AdamW", "RMSprop", 'SGD']}, 'optim_configs': {}},
        'batch_size' : {'state': false, 'suggest': 'int', 'kwargs': {'low': 16, 'high': 512, 'log': true}},
        'plr'        : {'state': false, 'suggest': 'cat', 'kwargs': {'choices': [1.0e-2, 1.0e-4, 1.0e-4]}}, # can only set as (1) step != 1 and log=False or (2) step=None and log=True
        'oalr'       : {'state': false, 'suggest': 'float', 'kwargs': {'low': 1.0e-4, 'high': 1.0e-2, 'step': null, 'log': true}},
        'oplr'       : {'state': false, 'suggest': 'float', 'kwargs': {'low': 1.0e-4, 'high': 1.0e-2, 'step': null, 'log': true}},
        'slr'        : {'state': false, 'suggest': 'float', 'kwargs': {'low': 1.0e-4, 'high': 1.0e-2, 'step': null, 'log': true}},
        'tlr'        : {'state': false, 'suggest': 'float', 'kwargs': {'low': 1.0e-4, 'high': 1.0e-2, 'step': null, 'log': true}},
        'dzlr'       : {'state': false, 'suggest': 'float', 'kwargs': {'low': 1.0e-4, 'high': 1.0e-2, 'step': null, 'log': true}},
        'dx'         : {'state': false, 'suggest': 'float', 'kwargs': {'low': 0.1400, 'high': 0.1600, 'step': null, 'log': false}},
        'pmode_max'  : {'state': false, 'suggest': 'int', 'kwargs': {'low': 1, 'high': 8, 'step': 1, 'log': false}},
        'conv_angle' : {'state': false, 'suggest': 'float', 'kwargs': {'low': 24, 'high': 26, 'step': 1, 'log': false}},
        'defocus'    : {'state': false, 'suggest': 'float', 'kwargs': {'low': -50, 'high': 50, 'step': 0.1, 'log': false}},
        'c3'         : {'state': false, 'suggest': 'float', 'kwargs': {'low': 4000, 'high': 10000, 'step': null, 'log': false}},
        'c5'         : {'state': false, 'suggest': 'float', 'kwargs': {'low': 50000, 'high': 100000, 'step': 5000, 'log': false}},
        'Nlayer'     : {'state': false, 'suggest': 'int', 'kwargs': {'low': 1, 'high': 8, 'step': 1, 'log': false}},
        'dz'         : {'state': false, 'suggest': 'float', 'kwargs': {'low': 4, 'high': 8, 'step': 0.5, 'log': false}},
        'scale'      : {'state': true, 'suggest': 'float', 'kwargs': {'low': 0.8, 'high': 1.2, 'step': 0.02, 'log': false}}, # This modifies the effective scan step size. [scale, asymmetry, rotation, shear] corresponds to 'scan_affine' under 'exp_params'
        'asymmetry'  : {'state': false, 'suggest': 'float', 'kwargs': {'low': -0.2, 'high': 0.2, 'step': 0.05, 'log': false}},
        'rotation'   : {'state': true, 'suggest': 'float', 'kwargs': {'low': -4, 'high': 4, 'step': 0.5, 'log': false}}, # This is essentially scan rotation
        'shear'      : {'state': false, 'suggest': 'float', 'kwargs': {'low': -4, 'high': 4, 'step': 0.5, 'log': false}},
        'tilt_y'     : {'state': false, 'suggest': 'float', 'kwargs': {'low': -5, 'high': 5, 'step': 0.5, 'log': false}}, # This refers to the 'init_tilts' under 'source_params['tilt_params']['init_tilts']'
        'tilt_x'     : {'state': false, 'suggest': 'float', 'kwargs': {'low': -5, 'high': 5, 'step': 0.5, 'log': false}}
    }
}
```
