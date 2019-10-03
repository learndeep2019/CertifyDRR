function model = Solver_multiclass_l2(algo, model, data, log, attack)
  iter = algo.iter;
  for i = 1:iter
    if i==1
      % random initialization
      num_samples = 2;
%       samples = randomvector(num_samples, data.num_fea)';
      samples = data.X(:,1:num_samples);
      nonlconstr = @(W)spectral_constr(W, samples, model, algo);
      
      % Check the derivative of nonlinear constraints
%       W = 0.1*ones(model.num_bases*model.classes, 1);
%       [f, g] = nonlconstr(W);
%       [grad, err] = gradest(nonlconstr, W);
%       [g(:) grad(:)]
%       fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));
    end

    fprintf(log.fileID, '---%d: samples = %d\n', i, num_samples);
    tic;
    fun = @(x)lossobj(x, model, algo);
    switch algo.loss
      case 'cs'
        W0 = rand(model.num_bases*model.classes, 1);
      case 'ww'
        W0 = zeros(model.num_bases*model.classes, 1);
      case 'crossent'
        W0 = zeros(model.num_bases*model.classes, 1);
    end
    
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
    [new_samples, lbfgs_spec] = maxLip_multiclass(model, data, algo);
    % return only different samples
    [new_samples, ~] = checksingular(samples, new_samples, log, algo);

    fprintf(log.lipfileID, '%d, %f\n', size(samples, 2), max(lbfgs_spec));
    
    samples = [samples, new_samples];
    if algo.greedysampling
      fprintf(log.fileID, 'Smallest singular value: %f\n', min(svd(samples)));
    end
    num_samples = size(samples, 2);
    
    nonlconstr = @(W)spectral_constr(W, samples, model, algo);

    
    % Print largest spectral norm
    fprintf(log.fileID, 'Max_lbfgs_spec: %f\n', max(lbfgs_spec));
    model.final_spec = max(lbfgs_spec);
    
   
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

% Constraint max_v \lambda_max( sum_j^d G_j v v^T G^T ) < L^2
function [y, yeq, grady, gradyeq] = spectral_constr(W, samples, model, algo)
  classes = model.classes;
  K_W_halfinv = sqrtm(invChol_mex(model.kernel_func(samples, samples) + 1e-7*eye(size(samples,2))));
  G = cell(1, classes);
  for i = 1:classes
    model.alpha  = W( (i-1)*model.num_bases+1: i*model.num_bases );
    [~, grad_fW] = kernel_classifier(samples, model);
    G{i}         = K_W_halfinv*grad_fW';
  end
  
  [f, v_star, u_star] = constr_solver(G);
  y = f - algo.max_lip^2;
  
  switch model.kernel
    case 'inverse'
      ustar_B     = u_star'*model.B;
      inprod      = model.B'*samples;    i = model.layers;
      K_BW        = 1./(i+1-i*inprod).^2;
      K_BW_vstar  = K_BW*K_W_halfinv*v_star;
      grad_uGv    = (ustar_B.*K_BW_vstar')*model.halfinvKBB;
    case 'gauss'
      ustar_B     = u_star'*model.B/model.bandwidth^2;
      K_BW        = model.kernel_func(model.B, samples);
      K_BW_vstar  = K_BW*K_W_halfinv*v_star;
      ustar_W     = u_star'*samples/model.bandwidth^2;
      grad_uGv    = ((ustar_B.*K_BW_vstar') - (ustar_W.*(K_W_halfinv*v_star)')*K_BW')*model.halfinvKBB;
  end
  

  grady = zeros(size(W));
  for i = 1:classes
    grady( (i-1)*model.num_bases+1: i*model.num_bases ) = 2*v_star'*G{i}*u_star*grad_uGv';
  end
  
  yeq = [];
  gradyeq = [];
end


% solving max_v \lambda_max( sum_j^d G_j v v^T G^T )
function [f, leading_v, leading_u] = constr_solver(G)
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
  f = singularvalue^2;
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
