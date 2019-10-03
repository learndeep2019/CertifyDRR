% This code is used to validate
% our regularizer is tighter than RHKS norm regularizer empirically 
% in binary classification
clear
rng(1)
trainingimg     = '../data/mnist/train-images.idx3-ubyte';
traininglabel   = '../data/mnist/train-labels.idx1-ubyte';
load_size       = 1000;
cropborder      = 0;
[imgs, labels]  = readMNIST(trainingimg, traininglabel, load_size, 0, cropborder);
data.oriX       = reshape(imgs, [], size(imgs, 3));
data.Y          = labels;
data.num_fea    = size(data.oriX, 1);

% Init model
model.kernel          = 'gauss';
switch model.kernel
  case 'inverse'
    model.layers      = 5;     % layers for stack inverse kernel
    model.kernel_func = @(x, y)kernel_inverse(x, y, model.layers);
    data.X            = normc(data.oriX);
    logfile           = ['inverse_' num2str(model.layers) '.txt'];
    model.normalize   = true;
    fprintf('kernel = %s, layer = %d\n', model.kernel, model.layers);
    
  case 'rbf'
    model.bandwidth   = median(pdist(normc(data.oriX)'));
    model.kernel_func = @(x, y)kernel_rbf(x, y, model.bandwidth^2);
    data.X            = normc(data.oriX);
    logfile           = ['rbf_' num2str(model.bandwidth) '.txt'];
    model.normalize   = true;
    fprintf('kernel = %s, bandwidth = %d\n', model.kernel, model.bandwidth);
    
  case 'gauss'
    model.bandwidth   = 3;
    model.kernel_func = @(x, y)kernel_gauss(x, y, model.bandwidth^2);
    data.X            = data.oriX;
    logfile           = ['gauss_' num2str(model.bandwidth) '.txt'];
    model.normalize   = false;
    fprintf('kernel = %s, bandwidth = %d\n', model.kernel, model.bandwidth);
  otherwise
    error('Not implemented kernel\n');
end
model.B           = data.X;
model.halfinvKBB  = sqrtm(invChol_mex(model.kernel_func(model.B, model.B)));
model.KBX         = model.kernel_func(model.B, data.X);   % Precompute phi(xtrain)
model.num_bases   = size(model.B, 2);
model.classes     = 10;

% setup for bfgs to find max lipschitz
opt.initialization = 2;
opt.lbfgs          = true;
opt.sampling_iter  = 10;
opt.defense_ord    = 'inf';
opt.num_simulation = 100;


%% Simulation
simulation(model, opt, data);



%%
function simulation(model, opt, data)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Simulation starts
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_rep   = opt.num_simulation;
  classes   = model.classes;
  num_bases = model.num_bases;
  for i = 1:num_rep
    rng(i)
    % Randomly choose alpha
    alpha        = 4*rand(num_bases, classes) - 2;
    %     alpha    = randn(model.num_bases, model.classes);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. Largest Gradient
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    model.W = alpha;
    [xvec, lipvec] = maxLip_multiclass(model, data, opt);
    [maxlip, idx]  = max(lipvec);
    optx           = xvec(:, idx);
    %     maxlip         = 0;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2. max eigenvalue bound (multiclass)
    % L2: \max_v \lambda_max( sum_j^d G_j v v^T G^T )
    % Linf: \max_c \sup_{\nbr{u}_\infty \le 1, \nbr{\phi}_1\le 1 } ( u^T G_c^T \phi )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    W = model.B;
%   W = [W, optx];  % appending points with largest grad 
    
    Kinv        = invChol_mex(model.kernel_func(W, W));
    K_W_halfinv = sqrtm(Kinv);
    G = cell(1, classes);
    for j = 1:classes
      model.alpha  = alpha(:, j);
      [~, grad_fW] = kernel_classifier(W, model);
      G{j}         = K_W_halfinv*grad_fW';
    end
    
    if strcmp(opt.defense_ord, '2')
      [eigenbound, ~, ~] = constr_solver_l2(G);
    else
      [eigenbounds, ~, ~] = constr_solver_linf(G);
      eigenbound = max(eigenbounds);
    end

    fprintf('maxlip = %f, eigenbound = %f\n', maxlip, eigenbound);
  end

end


% solving max_v \lambda_max( sum_j^d G_j v v^T G^T )
function [f, leading_v, leading_u] = constr_solver_l2(G)
  classes = length(G);
  [samples_size, coor] = size(G{1});
  
  leading_v = ones(samples_size, 1)*0.1;
  for i = 1:10
    H = zeros(coor, classes);
    for c = 1:classes
      H(:, c) = (leading_v'*G{c})';
    end
    [leading_u, ~, ~] = svds(H,1);
    
    H = zeros(samples_size, classes);
    for c = 1:classes
      H(:, c) = G{c}*leading_u;
    end
    [leading_v, singularvalue, ~] = svds(H,1);
  end
  f = singularvalue;
end

% Solving \sup_{\nbr{u}_\infty \le 1, \nbr{\phi}_1\le 1 } ( u^T G_c^T \phi )
% for each c.
% Return the optimal solutions for all ten constraints.
function [f_vec, u_star_mat, phi_star_mat] = constr_solver_linf(G)
  classes = length(G);
  [base_size, coor] = size(G{1});
  
  f_vec             = zeros(classes, 1);
  u_star_mat        = zeros(coor, classes);
  phi_star_mat      = zeros(base_size, classes);
  
  for c = 1:classes
    Gtil  = G{c};
    phi   = ones(size(Gtil,1), 1)*0.01;
    for i = 1:10
      optu  = sign(Gtil'*phi);
      phi   = Gtil*optu;
      phi   = phi/norm(phi);
    end
    f_vec(c) = optu'*Gtil'*phi;
    u_star_mat(:, c)   = optu;
    phi_star_mat(:, c) = phi;
  end
end