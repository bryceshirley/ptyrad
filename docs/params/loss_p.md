# loss_params

`loss_params` determines the individual loss terms for the CombinedLoss used for PtyRAD reconstruction
Generally, the reconstruction loss is the CombinedLoss = weight1 * loss1 + weight2 * loss2 + weight3 * loss3 ...
Use 'state' to switch on/off each loss term, and use 'weight' to determine their relative importance.
Each loss term would generate their corresponding gradient to the variable, and the final update is determined by the weighted sum of all gradients coming from all participating loss terms.
Data-error related loss terms ('loss_single', 'loss_poissn', and 'loss_pacbed') compare simulated DP with experimental DP, and their 'dp_pow' would raise the diffraction pattern to a power before the calculation
For ptychography purpose, you MUST have at least 1 out of the 3 data-error loss terms. Although you can set all of them to true, typical dataset works fine with 'loss_single' alone
Soft constraint (regularization)-like loss terms ('loss_sparse', and 'loss_simlar') are optional addition to the required data-error loss terms
Common regularization terms for image reconstruction tasks are total variation (TV) for smoothness and L_n norm (L1 and L2) for sparsity (i.e., promote near-zero or zero in the reconstructed tensor)

```yaml
loss_params : {
    'loss_single': {'state': true, 'weight': 1.0, 'dp_pow': 0.5}, # NRMSE error between single simulated and experimental diffraction pattern. 'dp_pow' is commonly set at 0.5 so NRMSE(DP_sim^0.5 - DP_exp^0.5) is equivalent to the Gaussian noise model for typical dataset (dose-sufficient) under the maximum-likelihood formalism
    'loss_poissn': {'state': false, 'weight': 1.0, 'dp_pow': 1.0, 'eps': 1.0e-6}, # negative log likelihood between simulated and experimental diffraction pattern. 'dp_pow' is commonly set at 1 so - (DP_sim * (DP_exp) - DP_exp) is equivalent to the Poisson noise model for low dose dataset under maximum likelihood formalism. See Odstrˇcil2018 https://doi.org/10.1364/OE.26.003108 for more details
    'loss_pacbed': {'state': false, 'weight': 0.5, 'dp_pow': 0.2}, # NRMSE error between simulated and experimental position-averaved CBED (PACBED). Similar to 'loss_single', except that it's comparing PACBED with PACBED and mostly focusing on the diffuse background when 'dp_pow' is set at 0.2
    'loss_sparse': {'state': true, 'weight': 0.1, 'ln_order': 1}, # L_n norm regularization calculated for object phase. 'ln_order' means the L_n norm (|a_i^n|^(1/n)) used to regularize object phase ('objp'). Usually 'ln_order' is set at 1 for L1 norm (|a|), this promotes 0 in the objp and enhance the sparsity (i.e. discrete atoms). 'ln_order' = 2 would be equivalent to L2 norm that promotes near-0 values
    'loss_simlar': {'state': false, 'weight': 0.1, 'obj_type': 'both', 'scale_factor': [1,1,1], 'blur_std': 1} # std across omode dimension for obj. This promotes similarity between object modes. 'obj_type' can be either 'amplitude', 'phase', or 'both'. 'scale_factor' as (zoom_z, zoom_y, zoom_x) is used to scale the object before calculating the std, setting 'scale_factor' to [1,0.5,0.5]  is equivalent to downsampling the obj 2x along y and x directions before calculating the std, which should encourage the obj modes to keep lateral atom shifts. Similarly, 'blur_std' applies a 2D (lateral) Gaussian blur kernel with specified std to blur the obj before calculating std along omode dimension
}
```
