# Generalised Lipschitz Regularisation Equals Distributional Robustness

## Environment
* MATLAB2018

## Classification Task

* Model Configuration ([config.json](Lipschitz_kernel_method/config.json))

  `data`: Set the dataset and its size. Currently support "mnist", "fashion-mnist" and "cifar10".

  `algorithm`: Set the parameters for the algorithm: loss, defense_ord and sampling etc.

  `model`: Set the parameters for the kernel model. Currently support "inverse" and "gauss" kernel.

  `attack`: Set the parameters for the attacker. Currently support "pgd", "fgs" and "random" attackers.

  For specific parameter setup, please checkout the [parser_parameter.m](Lipschitz_kernel_method/parser_parameter.m)

* Training
    ```
    cd Lipschitz_kernel_method
    run main.m
    ```
    
* Attack
    ```
    run attack_script.m
    ```
    
    
## Comparison of Bounds
* The empirical gap between adversarial risk and lipschitz regularised empirical risk (Figure 3 in the paper):
    ```
    run testbounds/at_certificate.m
    ```
* Comparison between our new bound and RKHS norm bound (Figure 4 in the paper):
    ```
    run testbounds/gap_simulation_binary.m
    ```


