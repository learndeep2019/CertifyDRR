# Certifying Distributional Robustness using Lipschitz Regularisation

## Environment
* MATLAB2018

## Classification Task

* Model Configuration (config.json)

  `data`: Set the dataset and its size. Currently support "mnist", "fashion-mnist" and "cifar10".

  `algorithm`: Set the parameters for the algorithm: loss, defense_ord and sampling etc.

  `model`: Set the parameters for the kernel model. Currently support "inverse" and "gauss" kernel.

  `attack`: Set the parameters for the attacker. Currently support "pgd", "fgs" and "random" attackers.

  For specific parameter setup, please checkout the [parser_parameter.m](lip_kernel_method/parser_parameter.m)

* Training
    ```
    cd lip_kernel_method
    run main.m
    ```
    
* Attack
    ```
    run attack_script.m
    ```
    
    
## Certificate and Bounds 
* The empirical gap between adversarial risk and lipschitz regularised empirical risk (Figure 2 in the paper):
    ```
    run testbounds/at_certificate.m
    ```
* Comparison between our new bound and RKHS norm bound (Figure 3 in the paper):
    ```
    run testbounds/gap_simulation_binary.m
    ```

