function [algo, model, data, attack, log] = parser_parameter(config)
    %% Loading data config
    if strcmp(config.data.name, 'mnist') || strcmp(config.data.name, 'fashion-mnist') 
      data = data_preprocess_mnist(config);
    elseif strcmp(config.data.name, 'cifar10') 
      data = data_preprocess_cifar10(config);
    else
      error('No such dataset\n');
    end
      
    %% Loading model
    model.kernel         = config.model.kernel;
    switch model.kernel
      case 'inverse'
        model.layers      = config.model.layers;     % layers for stack inverse kernel
        model.kernel_func = @(x, y)kernel_inverse(x, y, model.layers);
        data.X            = normc(data.oriX);
        data.Xte          = normc(data.oriXte);
        model.normalize   = true;
      case 'gauss'
        model.bandwidth   = config.model.bandwidth;
        model.kernel_func = @(x, y)kernel_gauss(x, y, model.bandwidth^2);
        data.X            = data.oriX;
        data.Xte          = data.oriXte;
        model.normalize   = false;
        data              = rmfield(data, {'oriX', 'oriXte'});
      otherwise
        error('Not implemented kernel\n');
    end
    
    
    % Nystrom approximation of Gram matrix and precompute auxiliary matrix
    % 1. using all trainset   2. uniformly sampling  3. kmean
    nystrom_sampling    = config.model.nystrom.sampling;    
    switch nystrom_sampling
      case 'all'
        model.num_bases = data.train_size;
        model.B         = data.X;
      case 'uniform'
        rng(1)
        model.num_bases = config.model.nystrom.landmarks;
        sampleIdx       = randperm(data.train_size, model.num_bases);
        model.B         = data.X(:, sampleIdx);
      case 'kmean'
        model.num_bases = config.model.nystrom.landmarks;
        if model.normalize
          model.B       = normc(mykmeans([data.X, data.X], model.num_bases));
        else
          model.B     = mykmeans([data.X, data.X], model.num_bases);
        end
    end
    KBB                 = model.kernel_func(model.B, model.B);  % Adding + 1e-7*eye(model.num_bases) if neccessary
    model.halfinvKBB    = sqrtm(invChol_mex(KBB));
    model.KBX           = model.kernel_func(model.B, data.X);   % Precompute phi(xtrain)
    model.KBXte         = model.kernel_func(model.B, data.Xte);   % Precompute k(B, Xte)
    model.PhiX          = model.halfinvKBB*model.KBX;
    
    
    %% Loading algorithm setup
    algo.task     = config.task;
    algo.loss     = config.algorithm.loss;    % loss function for training
    algo.margin   = config.algorithm.margin;  % margin in loss
    
    algo.defense_ord    = config.algorithm.defense_ord;    % Defensed norm: '2' or 'inf'
    algo.max_lip        = config.algorithm.max_lip;        % Set as 0 means no constraints on Lip
    algo.rkhs_norm      = config.algorithm.rkhs_norm;      % Set as 0 means no constraints on RHKS norm
    algo.iter           = config.algorithm.iter;
    algo.poly           = config.algorithm.poly;           % True to use poly algorithm, otherwise exponential alg.
    
    algo.greedysampling = config.algorithm.sampling.greedysampling;    % True to use greedy sampling, otherwise random sampling
    algo.lbfgs          = config.algorithm.sampling.lbfgs;             % For greedy sampling: True to use lbfgs find max lip points, otherwise using fmincon.
    algo.initialization = config.algorithm.sampling.initialization;    % 1. random init  2. init by train points with largest grad
    algo.sampling_iter  = config.algorithm.sampling.sampling_iter;     % try #iter initializations and report largest lip constant
    

    %% Loading attack setup
    attack.task     = config.task;                     
    attack.workers  = config.workers;                 % parallel running on different perturbation radius. Just set as 0
    attack.dataname = config.attack.dataname;         % attack on 'train' or 'test'
    attack.method   = config.attack.method;           % attack method: 'pgd' or 'fgs' or 'random'
    attack.loss     = config.attack.loss;             % attack loss: 'cw' or 'crossent'
    attack.margin   = config.attack.margin;           % margin in the attack loss
    attack.ord      = str2num(config.attack.ord);     % l2 or linf attacks
    eps_para        = config.attack.eps;              % radius array,  e.g., [0, 0.5, 0.1] means [0, 0.1, 0.2, 0.3, 0.4, 0.5] 
    attack.eps      = linspace(eps_para(1), eps_para(2), eps_para(3));
    attack.rand_init= config.attack.rand_init;        % random init
    attack.iters    = config.attack.iters;            % PGD iterations
    attack.clip     = config.attack.clip;             % clip to [0, 1]
    attack.islog    = config.attack.islog;
    attack.save_adv = config.attack.save_adv;

    
    
    if nargout > 4
      %% Loading logging setup
      log.islog        = config.logging.islog;     % True to log in file, otherwise output in terminal
      log.root         = path_generator(model, algo, data);
      log.filename     = config.logging.filename;
      log.save_model   = config.logging.save_model;
    end
    
    
end

