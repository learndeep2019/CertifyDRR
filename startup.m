disp('Initializing');
root = pwd;

% Add library folder which contains L-BFGS package
addpath(genpath([root '/lib']));

% Add code folder
addpath(genpath([root '/lip_kernel_method']));