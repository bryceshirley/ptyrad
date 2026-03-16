# init_params

`init_params` determines the experimental parameters, model complexity, preprocessing steps, and where to load the input data.

```yaml
init_params : {
    # Experimental params
    'probe_illum_type'       : 'electron',
    'probe_kv'               : 80,
    'probe_conv_angle'       : 24.9,
    'probe_defocus'          : 0,
    'probe_c3'               : 0,
    'probe_c5'               : 0,
    'meas_Npix'              : 128,
    'pos_N_scans'            : 16384,
    'pos_N_scan_slow'        : 128,
    'pos_N_scan_fast'        : 128,
    'pos_scan_step_size'     : 0.4290,
    'meas_calibration'       : {'mode': 'fitRBF', 'value': null},
    # Model complexity
    'probe_pmode_max'        : 6,
    'probe_pmode_init_pows'  : [0.02],
    'obj_omode_max'          : 1,
    'obj_omode_init_occu'    : {'occu_type': 'uniform', 'init_occu': null},
    'obj_Nlayer'             : 6,
    'obj_slice_thickness'    : 2,
    # Preprocessing
    'meas_permute'           : null,
    'meas_reshape'           : null,
    'meas_flipT'             : [1,0,0],
    'meas_crop'              : null,
    'meas_pad'               : {'mode': null, 'padding_type': 'power', 'target_Npix': 256,'value': 0},
    'meas_resample'          : {'mode': null, 'scale_factors': [2,2]},
    'meas_add_source_size'   : null,
    'meas_add_detector_blur' : null,
    'meas_remove_neg_values' : {'mode': 'clip_neg', 'value': null},
    'meas_add_poisson_noise' : null,
    'meas_export'            : null,
    'probe_permute'          : null,
    'pos_scan_flipT'         : null,
    'pos_scan_affine'        : null,
    'pos_scan_rand_std'      : 0.15,
    # Input source and params
    'meas_source'            : 'file',
    'meas_params'            : {'path': 'data/tBL_WSe2/Panel_g-h_Themis/scan_x128_y128.raw'},
    'probe_source'           : 'simu',
    'probe_params'           : null,
    'pos_source'             : 'simu',
    'pos_params'             : null,
    'obj_source'             : 'simu',
    'obj_params'             : null,
    'tilt_source'            : 'simu',
    'tilt_params'            : {'tilt_type':'all', 'init_tilts':[[0,0]]},
}
```
