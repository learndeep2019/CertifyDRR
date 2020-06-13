% This code is used to validate
% our regularizer is tighter than RHKS norm regularizer empirically 
% in binary classification
clear
rng(1)
trainingimg    = '../data/mnist/train-images.idx3-ubyte';
traininglabel  = '../data/mnist/train-labels.idx1-ubyte';
loading_size   = 6000;
count_per_class= 400;
cropborder     = 0;
[imgs, labels] = readMNIST(trainingimg, traininglabel, loading_size, 0, cropborder);
Xtrain  = reshape(imgs, [], size(imgs, 3));
pos_idx = find(labels == 1);
neg_idx = find(labels == 0);
pos_idx = pos_idx(1:count_per_class);
neg_idx = neg_idx(1:count_per_class);
Ytrain  = [ones(count_per_class, 1); -ones(count_per_class, 1)];
Xtrain  = Xtrain(:, [pos_idx; neg_idx]);
p = randperm(count_per_class*2);
data.oriY = Ytrain(p);
data.oriX = Xtrain(:, p);
data.num_fea = size(data.oriX, 1);
data.count_per_class = count_per_class;

% Init model
model.kernel          = 'gauss';
switch model.kernel
  case 'inverse'
    model.layers      = 5;     % layers for stack inverse kernel
    model.kernel_func = @(x, y)kernel_inverse(x, y, model.layers);
    fprintf('kernel = %s, layer = %d\n', model.kernel, model.layers);
    data.X            = normc(data.oriX);
    logfile           = ['inverse_' num2str(model.layers) '.txt'];
  case 'rbf'
    model.bandwidth   = median(pdist(normc(data.oriX)'));
    model.kernel_func = @(x, y)kernel_rbf(x, y, model.bandwidth^2);
    fprintf('kernel = %s, bandwidth = %d\n', model.kernel, model.bandwidth);
    data.X            = normc(data.oriX);
    logfile           = ['rbf_' num2str(model.bandwidth) '.txt'];
  case 'gauss'
    model.bandwidth   = 3;
    model.kernel_func = @(x, y)kernel_gauss(x, y, model.bandwidth^2);
    fprintf('kernel = %s, bandwidth = %d\n', model.kernel, model.bandwidth);
    data.X            = data.oriX;
    logfile           = ['gauss_' num2str(model.bandwidth) '.txt'];
  otherwise
    error('Not implemented kernel\n');
end
model.B           = data.X;
model.halfinvKBB  = eye(size(model.B, 2));


% setup for bfgs to find max lipschitz
opt.initialization = 2;
opt.lbfgs          = true;
opt.sampling_iter  = 10;
opt.defense_ord    = '2';
opt.num_simulation = 100;


%% Simulation and Logging
% If relog=true, run simulation and log results
% Else, read existing logfile
relog = true;
if relog
  fileID = fopen(logfile,'w');
  simulation(model, opt, data, fileID);
  fclose(fileID);
end
plotgap(logfile, model);



%%
function simulation(model, opt, data, fileID)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Simulation starts
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_rep = opt.num_simulation;
  count_per_class = data.count_per_class;
  for i = 1:num_rep
    rng(i)
    % Randomly choose alpha
    model.alpha        = 4*rand(2*count_per_class, 1) - 2;
    %     model.alpha    = randn(2*count_per_class, 1);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. Largest Gradient
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [xvec, lipvec] = maxLip_binary(model, data, opt);
    [maxlip, idx]  = max(lipvec);
    optx           = xvec(:, idx);
    %     maxlip         = 0;



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2. RKHS norm bound ||f||_H^2
    %    This only bounds L2 norm of gradient
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(opt.defense_ord, '2')
      gradmatrix = model.kernel_func(data.X, data.X);
      switch model.kernel
        case 'rbf'
          t = min(model.bandwidth, 1);
          square_rkhs_f = model.alpha'*gradmatrix*model.alpha/t^2;
        case 'inverse'
          square_rkhs_f = model.alpha'*gradmatrix*model.alpha;
        case 'gauss'
          t = min(model.bandwidth, 1);
          square_rkhs_f = model.alpha'*gradmatrix*model.alpha/t^2;
      end
      %         square_rkhs_f = 0;
    else
      square_rkhs_f = 0;
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3. Our first regularizer: sum_i ||\gtil_i||_2^2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nystrom_samples_num = 2000;  % nystrom approximation basis num
    nystrom_sampling    = 3;
    switch nystrom_sampling
      case 1    % generate samples from L2 unit ball
        W = randomvector(nystrom_samples_num, data.num_fea)';

      case 2    % generate samples from L2 sphere
        W = rand(nystrom_samples_num, data.num_fea)'*2-1;
        %         W = [W, normc(optx)];   % appending points with largest grad

      case 3    % using training data
        W = model.B;
%         W = [W, optx];  % appending points with largest grad 
    end 

    [~, gradfW] = kernel_classifier(W, model);
    if strcmp(model.kernel, 'inverse') % using projected gradient
      gradfW = gradfW - bsxfun(@times, sum(gradfW.*W), W);
    end
    Kinv        = invChol_mex(model.kernel_func(W, W));
    Khalfinv    = sqrtm(Kinv);
    
    if strcmp(opt.defense_ord, '2')
      ourbound  = trace(gradfW*Kinv*gradfW');
    else
      Gtil      = Khalfinv*gradfW';
      ourbound  = sum(sqrt(sum(Gtil.^2)));
    end
    %     ourbound = 0;

    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 4. max eigenvalue bound (binary)
    % L2: \lambda_max( \sum_j \gtil_j \gtil_j^T )
    % L1: \sup_{\nbr{u}_\infty \le 1, \nbr{\phi}_1\le 1 } ( u^T G^T \phi )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(opt.defense_ord, '2')
      [~, eigenbound] = eigs(Khalfinv*(gradfW'*gradfW)*Khalfinv, 1);
    else
      eigenbound = boundsolver(Gtil);
    end

    fprintf('maxlip = %f, rkhsbound = %f, firstbound = %f, eigenbound = %f\n', maxlip, square_rkhs_f, ourbound, eigenbound);
  end

  our_bound  = sqrt(eigenbound) - sqrt(maxlip);
  rkhs_bound = sqrt(square_rkhs_f) - sqrt(maxlip);

  % dump numerical bounds into files
  fprintf(fileID, '%.4f, %.4f\n', rkhs_bound, our_bound);
end


function plotgap(logfile, model)
  fid = fopen(logfile,'rt');
  C = textscan(fid, '%f %f', 'Delimiter',',');
  fclose(fid);
  rkhsgap = C{1};
  ourgap = C{2};
  
  x_lim    = max(ourgap);
  y_lim    = max(rkhsgap);
  axislim  = ceil(max(x_lim, y_lim));

  % xticks(0:1:axislim);
  % yticks(0:1:axislim);
  
  scatter(ourgap, rkhsgap, 30, 'filled')
  hold on
  % x = linspace(0,axislim);
  % line(x,x, 'Color','red', 'LineStyle','--', 'LineWidth', 2)
  fplot(@(x) x, [0 axislim+1], 'Color','red', 'LineStyle','--', 'LineWidth', 2)
  box on

  % xlFontSize = get(xl,'FontSize');
  xAX = get(gca,'XAxis');
  set(xAX,'FontSize', 25);
  yAX = get(gca,'YAxis');
  set(yAX,'FontSize', 25);

  xlhand = get(gca,'xlabel');
  set(xlhand, 'string', '\boldmath$\lambda_{\max} (G^T G)^{1/2} - \textbf{lip}_X(f)$', 'fontsize',25, 'Interpreter','latex');
  ylhand = get(gca,'ylabel');
  set(ylhand, 'string', '\boldmath$||f||_\mathcal{H} \sup_{z>0}\frac{g(z)}{z} - \textbf{lip}_X(f)$', 'fontsize',25, 'Interpreter','latex');

  daspect([1,1,1]);
  xlim([0 axislim*1]);
  ylim([0 axislim*1]);
  xl = xlim;
  fplot(@(x) x, [0 xl(2)], 'Color','red', 'LineStyle','--', 'LineWidth', 2)
  if strcmp(model.kernel, 'inverse')
    saveas(gcf, [num2str(model.kernel) '_' num2str(model.layers) '.pdf']);
  else
    saveas(gcf, [num2str(model.kernel) '_' num2str(model.bandwidth) '.pdf']);
  end
  close
end



%%
% h = ttest(res_bound(2,:) - res_bound(1, :), 0, 'Tail', 'right')

% plotgap(res_bound(1, :), res_bound(2,:), model)
% scatter(res_bound(1,:), res_bound(2, :));
% axis equal
% xlim([0,130]); ylim([0,130]); hold on;
% plot([0, 130], [0, 130]);


% Solving \sup_{\nbr{u}_\infty \le 1, \nbr{\phi}_1\le 1 } ( u^T G^T \phi )
function bound = boundsolver(Gtil)
  
  phi  = ones(size(Gtil,1), 1)*0.01;
  for i = 1:10
    optu = sign(Gtil'*phi);
    phi = Gtil*optu;
    phi = phi/norm(phi);
  end
  bound = optu'*Gtil'*phi;
  
%   obj     = @(x)constr_obj(x, Gtil);
%   u  = ones(size(Gtil,2), 1)*0.01;
%   lb = -1*ones(size(u));
%   ub = ones(size(u));
%     % check gradient
% %   [f, g] = obj(u);
% %   [grad, err] = gradest(obj, u);
% %   [g(:) grad(:)]
% %   fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));
%   options = optimoptions('fmincon','SpecifyObjectiveGradient',true, 'Display','none');
%   [opt_phi,bound] = fmincon(obj,u,[],[],[],[],lb,ub,[], options);
%   bound = -bound;
end

% function [f, g] = constr_obj(u, Gtil)
%   optphi = Gtil*u;
%   f = -norm(optphi);
%   optphi = -optphi/f;
%   g = -Gtil'*optphi;
% end
