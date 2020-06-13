function [xvec, lipvec] = maxLip_binary(model, data, algo)
%   Input arguments
%   ----------
%   model:
%       model.kernel
%       model.layer/model.bandwidth
%       model.kernel_func
%       model.B    
%       model.halfinvKBB (if using all training, then set as identity)
%       model.alpha
%       model.PhiX (algoional)
%   data:  training samples
%   algo:
%       algo.lbfgs
%       algo.sampling_iter
%       algo.initialization

  num_fea = data.num_fea;

  t = algo.sampling_iter;
  xvec = [];
  lipvec = [];
  
  model.invKBBalpha  = model.halfinvKBB*model.alpha;
  switch algo.initialization
    case 1
      % random initialization
      initpoints = 4.*rand(num_fea, t) - 2;
    case 2
      % init with train points of largest grad
      [~, gradfX] = kernel_classifier(data.X, model);
      if strcmp(algo.defense_ord, '2')
        gradnormvec = sum(gradfX.^2);
      else
        gradnormvec = sum(abs(gradfX));
      end
      [~, idx] = sort(gradnormvec, 'descend');
      initpoints = data.X(:, idx(1:t));
    case 3
      % margin trianing points initialization
      f_x = model.alpha'*model.PhiX;
      [~, idx] = sort((abs(f_x)));
      initpoints = data.X(:, idx(1:t));
  end

  % Finding max Lipschitz constant
  if algo.lbfgs
    param = [];
    param.maxIter = 50;     % max number of iterations
    param.maxFnCall = 100;  % max number of calling the function
    param.relCha = 1e-6;      % tolerance of constraint satisfaction
    param.tolPG = 1e-6;   % final objective function accuracy parameter
    param.m = 50;
    ub = ones(num_fea, 1);   lb = zeros(num_fea, 1);
    show_iter = @(t, f, x)perf_iter(t, f, x);
    obj = @(x)gradf(x, model, algo);
    
    % Check the derivative of \grad f(w)=\sum_i alpha_i \grad k(x_i, w)
%     sol = initpoints(:, 1);
%     [f, g] = obj(sol);
%     [grad, err] = gradest(obj, sol);
%     [g(:) grad(:)]
%     fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));
    

    for i=1:t
      [x, fval, iter, numCall, msg] = lbfgsb(initpoints(:,i), lb, ub, obj, [], [], param); % [y, fval, iter, numCall, msg]
%       fprintf("%f, %f, %f\n", obj(initpoints(:,i)), fval, iter);
%       fprintf("f(x) = %f, fval = %f\n", obj(x), fval);
      
      if strcmp(model.kernel, 'inverse') || strcmp(model.kernel, 'rbf') 
        x = x./norm(x);
      end
      
      if i==1 || isunique(xvec, x)
        xvec = [xvec, x];
        lipvec = [lipvec, -fval];
      end
    end
    
  else
    obj = @(x)gradf(x, model, algo);
    for i=1:t
      options = optimoptions(@fminunc,'Display','none','Algorithm','trust-region','SpecifyObjectiveGradient',true);
      [x, fval, exitflag] = fminunc(obj, initpoints(:,i), options);
%       fprintf("%f, %f, %f\n", obj(initpoints(:,i)), fval, exitflag);

      if i==1 || isunique(xvec, x)
        xvec = [xvec, x];
        lipvec = [lipvec, -fval];
      end
    end
  end
end



function flag = isunique(xvec, x)
  flag = 1;
  [num_fea, len] = size(xvec);
  for i = 1:len
    if ~(sum(abs(xvec(:, i) - x) > 0.01) > num_fea/8)
      flag = 0;
    end
  end
end

function [f, g] = gradf(y, model, algo) 
  
  switch model.kernel
    case 'inverse'
      normy   = norm(y);
      x       = y./normy;
      inprod  = model.B'*x;    i = model.layers;
      gradKXW = 1./(i+1-i*inprod).^2;
      gradfW  = model.B*bsxfun(@times, gradKXW, model.invKBBalpha);


      hess    = model.B*bsxfun(@times, model.invKBBalpha.*(2*i./(i+1-i*inprod).^3), model.B');
      if strcmp(algo.defense_ord, '2')
        % Full gradient
%         f     = -norm(gradfW)^2;
%         G     = -2*gradfW'*hess';
        
        % Projected gradient
        f     = -norm(gradfW)^2 + (gradfW'*x)^2;
        G     = -2*gradfW'*hess' + 2*gradfW'*x*(gradfW+hess'*x)';
      else
        f     = -sum(abs(gradfW));
        G     = -sign(gradfW')*hess';
      end
      g       = G/normy-G*y*y'/normy^3;
    
    case 'rbf'
      normy   = norm(y);
      x       = y./norm(y);
      KXW     = model.kernel_func(model.B, x);
      coff    = model.invKBBalpha.*KXW;
      gradfW  = model.B*coff/model.bandwidth^2;

      
      hess    = bsxfun(@times, (model.invKBBalpha.*KXW)', model.B)*model.B'/model.bandwidth^4;
      if strcmp(algo.defense_ord, '2')
        % Full gradient
%         f       = -norm(gradfW)^2;
%         G       = -2*gradfW'*hess;
        
        % Projected gradient
        f     = -norm(gradfW)^2 + (gradfW'*x)^2;
        G     = -2*gradfW'*hess + 2*gradfW'*x*(gradfW+hess*x)';
      else
        f     = -sum(abs(gradfW));
        G     = -sign(gradfW')*hess;
      end
      g       = G/normy-G*y*y'/normy^3;
      
    case 'gauss'
      KXW     = model.kernel_func(model.B, y);
      coff    = bsxfun(@times, model.invKBBalpha, KXW);
      gradfW  = (model.B*coff - sum(coff)*y)/model.bandwidth^2;

      s = bsxfun(@minus, model.B, y);
      hess = bsxfun(@times, (model.invKBBalpha.*KXW)', s)*s'/model.bandwidth^4 ...
                          - (model.invKBBalpha'*KXW)*eye(length(y))/model.bandwidth^2;
                        
      if strcmp(algo.defense_ord, '2')
        f     = -norm(gradfW)^2;
        g     = -2*gradfW'*hess;
      else
        f     = -sum(abs(gradfW));
        g     = -sign(gradfW')*hess;
      end
  end
end

function perf_iter(t, f, x)
  fprintf('%3d: f = %0.5g\n', t, f);
  pause(0.1);
end
