% 
% Tbeoretically, adversarial risk <= Lipschitz regularized empirical risk 
%
clear
rng(1)
trainingimg    = '../data/mnist/train-images.idx3-ubyte';
traininglabel  = '../data/mnist/train-labels.idx1-ubyte';
loading_size   = 2000;
count_per_class= 50;
[imgs, labels] = readMNIST(trainingimg, traininglabel, loading_size, 0, 0);
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


% init model
model.kernel = 'gauss';  model.bandwidth = median(pdist(data.oriX'));
model.kernel_func = @(x, y)kernel_gauss(x, y, model.bandwidth^2);
fprintf('kernel = %s, bandwidth = %d\n', model.kernel, model.bandwidth);
data.X            = data.oriX;
model.normalize   = false;
model.B           = data.X;
model.halfinvKBB  = eye(size(model.B, 2));



% setup for bfgs to find max lipschitz
opt.initialization = 2;
opt.lbfgs          = true;
opt.sampling_iter  = 10;
opt.defense_ord    = '2';
opt.num_simulation = 100;
opt.margin         = 0;
opt.loss_sup       = @(z)loss_logistic(z, data.oriY, opt.margin);
logfile            = ['certificate_gauss_' num2str(model.bandwidth) '_ord_' opt.defense_ord '.txt'];
  
% Logging
% If relog=true, run simulation and log results
% Else, read existing logfile
relog = true;
if relog
  fileID = fopen(logfile,'w');
  simulation(model, opt, data, fileID);
  fclose(fileID);
end
plotcert(logfile, model, opt.defense_ord);


function simulation(model, opt, data, fileID)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Simulation starts
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  num_rep = opt.num_simulation;
  count_per_class = data.count_per_class;
  for i = 1:num_rep
    rng(i)
    % Randomly choose alpha
    model.alpha   = 4*rand(count_per_class*2, 1) - 2;
%     model.alpha   = randn(count_per_class*2, 1);
    
    if strcmp(opt.defense_ord, '2')
      rho = 3;
    else
      rho = 0.3;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % 1. Adversarial Risk
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PGD attack setup
    attack.task      = 'binary';
    attack.method    = 'pgd';
    attack.rand_init = false;
    attack.iters     = 20;
    attack.clip      = false;
    attack.ord       = str2num(opt.defense_ord);
    attack.eps       = rho;
    attack.logging   = false;
    attack.save_adv  = false;
    attack.lossfunc  = opt.loss_sup;
    
    % Attack training samples
    [~, adv_risk] = attack_model(model, data.oriX, attack.eps, attack);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2. Lipschitz regularized empirical risk 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Empirical risk
    PhiX          = model.kernel_func(data.X, data.X);
    emp_risk      = opt.loss_sup(PhiX'*model.alpha);
    % Regularization term
    [~, lipvec]   = maxLip_binary(model, data, opt);
    [maxlip, ~]   = max(lipvec);
    % our loss does not average, so here we need to multiply #training point
    if strcmp(opt.defense_ord, '2')
      reg         = 2*count_per_class*rho*sqrt(maxlip);
    else
      reg         = 2*count_per_class*rho*maxlip;
    end
    RHS = emp_risk + reg;
    
    
    fprintf('emp_risk = %f, adv_risk = %f, lip_reg_emp_risk = %f\n',emp_risk, adv_risk, RHS);
    avg_adv_risk = adv_risk/(2*count_per_class);
    avg_lip_reg_emp_risk = RHS/(2*count_per_class);
    
    % dump numerical bounds into files
    fprintf(fileID, '%.4f, %.4f\n', avg_adv_risk, avg_lip_reg_emp_risk);
  end
end

function plotcert(logfile, model, defense_ord)
  fid = fopen(logfile,'rt');
  C = textscan(fid, '%f %f', 'Delimiter',',');
  fclose(fid);
  avg_adv_risk = C{1};
  avg_lip_reg_emp_risk = C{2};

  x_lim    = max(avg_adv_risk);
  y_lim    = max(avg_lip_reg_emp_risk);
  axislim  = ceil(max(x_lim, y_lim));

  % xticks(0:1:axislim);
  % yticks(0:1:axislim);

  scatter(avg_adv_risk, avg_lip_reg_emp_risk, 30, 'filled')
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
  set(xlhand, 'string', 'Adversarial risk in Eq (3)', 'fontsize', 30, 'Interpreter','latex');
  ylhand = get(gca,'ylabel');
  set(ylhand, 'string', 'RHS of Eq (1)', 'fontsize', 30, 'Interpreter','latex');

  daspect([1,1,1]);
  % xlim([0 axislim*1.1]);
  % ylim([0 axislim*1.1]);
  xl = xlim;
  fplot(@(x) x, [0 xl(2)], 'Color','red', 'LineStyle','--', 'LineWidth', 2)
  saveas(gcf, ['certificate_' model.kernel '_' num2str(model.bandwidth) '_ord_' defense_ord '.pdf']);
  close
end
