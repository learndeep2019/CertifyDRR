function model = Solver_multiclass_linf(algo, model, data, log, attack)
  iter = algo.iter;
  for i = 1:iter
    if i==1
      % random initialization
      num_samples = 2;
%       samples = randomvector(num_samples, data.num_fea)';
      samples = data.X(:,1:num_samples);
      nonlconstr = @(W)linf_constr(W, samples, model, algo);
      
      % Check the derivative of nonlinear constraints
%       nonlconstr = @(W)linf_constr_test(W, samples, model, algo);
%       W = 0.1*ones(model.num_bases, 1);
%       [f, g] = nonlconstr(W);
%       [grad, err] = gradest(nonlconstr, W);
%       [g(:) grad(:)]
%       fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));
    end

    fprintf(log.fileID, '---%d: samples = %d\n', i, num_samples);
    tic;
    fun = @(x)lossobj(x, model, algo);
    W0 = rand(model.num_bases*model.classes, 1)*0.5;
    
%     [f, g] = fun(W0);
%     [grad, err] = gradest(fun, W0);
%     [g(:) grad(:)]
%     fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));
    
    options = optimoptions(@fmincon,'Algorithm','interior-point', ...
        'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true, ...
        'MaxFunctionEvaluations', 300, 'optimalityTolerance',1e-2, ...
        'HessianApproximation', 'lbfgs', 'SubproblemAlgorithm', 'cg', ...
        'StepTolerance', 1e-8, 'Display','final');
    [W,fval,eflag,output] = fmincon(fun,W0,[],[],[],[],[],[],nonlconstr,options); % [alpha,fval,eflag,output,lambda]
    
    fprintf(log.fileID, '#iter: %d, #funccall: %d, flag: %d, loss: %.3f, cputime: %.2f \n', ...
                        output.iterations, output.funcCount, eflag, fval, toc);
    model.W = reshape(W, model.num_bases, model.classes);
    
    % Sampling new points and add to constraints
    [new_samples, lbfgs_linfnorm] = maxLip_multiclass(model, data, algo);
    % return only different samples
    [new_samples, ~] = checksingular(samples, new_samples, log, algo);

    fprintf(log.lipfileID, '%d, %f\n', size(samples, 2), max(lbfgs_linfnorm));
    
    samples = [samples, new_samples];
    if algo.greedysampling
      fprintf(log.fileID, 'Smallest singular value: %f\n', min(svd(samples)));
    end
    num_samples = size(samples, 2);
    
    nonlconstr = @(W)linf_constr(W, samples, model, algo);

    
    % Print largest linf norm
    fprintf(log.fileID, 'Max_lbfgs_linfnorm: %f\n', max(lbfgs_linfnorm));
    model.final_spec = max(lbfgs_linfnorm);
    
   
    % Compute training error and test error
    lastsecondlayer   = model.halfinvKBB*model.KBX;
    logit             = model.W'*lastsecondlayer;
    [~, largestlabel] = max(logit);
    pred_label        = largestlabel-1;
    train_acc         = mean(pred_label' == data.Y)*100;
    sorted_logit      = sort(logit, 'descend');
    train_margin      = mean(sorted_logit(1,:) - sorted_logit(2,:));
    
    
    lastsecondlayer   = model.halfinvKBB*model.KBXte;
    logit             = model.W'*lastsecondlayer;
    [~, largestlabel] = max(logit);
    pred_label        = largestlabel-1;
    test_acc          = mean(pred_label' == data.Yte)*100;
    sorted_logit      = sort(logit, 'descend');
    test_margin       = mean(sorted_logit(1,:) - sorted_logit(2,:));
    
    fprintf(log.fileID, 'Train_acc    Test_acc        Train_margin    Test_margin\n');
    fprintf(log.fileID, '%8.3f%%    %8.3f%%    %12.3f    %12.3f\n\n\n', ...
                        train_acc, test_acc, train_margin, test_margin);

                      
    % Attack train data
%     attack.lossfunc = algo.lossfunc;
%     attack_main(model, data.oriX, data.Y, attack, 'train');
    
    pause(0.2);
  end
end


% Loss objective
function [y,grady] = lossobj(W, model, algo)
  mW = reshape(W, model.num_bases, model.classes);
  [loss, g] = algo.lossfunc(mW'*model.PhiX);
  y = loss;
  grady = model.PhiX*g';
  grady = grady(:);
end

% Constraint \sup_{\nbr{u}_\infty \le 1, \nbr{\phi}_1\le 1 } ( u^T G^T \phi ) < L
% for all classes.
function [y, yeq, grady, gradyeq] = linf_constr(W, samples, model, algo)
  classes = model.classes;
  K_W_halfinv = sqrtm(invChol_mex(model.kernel_func(samples, samples) + 1e-7*eye(size(samples,2))));
  G = cell(1, classes);
  for i = 1:classes
    model.alpha  = W( (i-1)*model.num_bases+1: i*model.num_bases );
    [~, grad_fW] = kernel_classifier(samples, model);
    G{i}         = K_W_halfinv*grad_fW';
  end
  
  [f_vec, u_star_mat, phi_star_mat] = constr_solver(G);
  y = f_vec - algo.max_lip;
  
  value = zeros(size(W));
  
  for c = 1:classes
    switch model.kernel
      case 'inverse'
        ustar_B     = u_star_mat(:, c)'*model.B;
        inprod      = model.B'*samples;    i = model.layers;
        K_BW        = 1./(i+1-i*inprod).^2;
        K_BW_vstar  = K_BW*K_W_halfinv*phi_star_mat(:, c);
        grad_uGv    = (ustar_B.*K_BW_vstar')*model.halfinvKBB;
      case 'gauss'
        ustar_B       = u_star_mat(:, c)'*model.B/model.bandwidth^2;
        K_BW          = model.kernel_func(model.B, samples);
        K_BW_phistar  = K_BW*K_W_halfinv*phi_star_mat(:, c);
        ustar_W       = u_star_mat(:, c)'*samples/model.bandwidth^2;
        grad_uGv      = ((ustar_B.*K_BW_phistar') - (ustar_W.*(K_W_halfinv*phi_star_mat(:, c))')*K_BW')*model.halfinvKBB;
    end
    
    value((c-1)*model.num_bases+1: c*model.num_bases) = grad_uGv';
  end
  var_size  = size(W, 1);
  column    = repmat(linspace(1,classes,classes),model.num_bases,1);
  grady     = sparse(linspace(1,var_size,var_size), column(:), value);
  
  yeq       = [];
  gradyeq   = [];
end


% Solving \sup_{\nbr{u}_\infty \le 1, \nbr{\phi}_1\le 1 } ( u^T G_c^T \phi )
% for each c.
% Return the optimal solutions for all ten constraints.
function [f_vec, u_star_mat, phi_star_mat] = constr_solver(G)
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


function [new_samples, flag] = checksingular(samples, new_samples, log, algo)
  [num_fea, newlen] = size(new_samples);
  oldlen = size(samples, 2);
  conflictidx = [];
  flag = 0;
  for i = 1:newlen
    for j = oldlen:-1:1
      if ~(sum(abs(samples(:, j) - new_samples(:, i)) > 0.01) > num_fea/8)
        % new sample conflict with old sample
        conflictidx = [conflictidx, i];
        break;
      end
    end
  end
  
  if length(conflictidx) == newlen % all samples conflict
    new_samples = randomvector(2, num_fea)';
    fprintf(log.fileID, 'all new samples conflict!!!\n');
    flag = 1;
  else
    new_samples(:, conflictidx) = [];
  end
end
