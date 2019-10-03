function [adv_x, loss] = attack_model(model, pureX, pureY, eps, attack)

  % Check gradient of attack loss
%   true_prob = zeros(10, 1); true_prob(3) = 1;
%   attack.lossfunc = @(z)loss_cw(z, true_prob);
%   obj = @(x)attackloss(x, model, attack);
%   x = rand(size(pureX, 1), 1);
%   [fval, g] = obj(x);
%   [grad, err] = gradest(obj, x);
%   [g(:) grad(:)]
%   fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));  
%   error('gradest');
  
  
  adv_obj = @(x)attackloss(x, model, attack);
  switch attack.method
    case 'fgs'
      adv_x = fgm(pureX, adv_obj, eps, attack);
      
    case 'pgd'
      nb_iter = attack.iters;
      step_size = 20*eps/nb_iter;
      
      if attack.rand_init
        if isinf(attack.ord)  % Linf norm ball random init
          adv_x = pureX + (rand(size(pureX))*2*eps - eps);
        else  % L2 norm ball random init
          [fea, num] = size(pureX);
          adv_x = pureX + randomvector(num, fea, eps)';
        end
      else
        adv_x = pureX;
      end
      
      for i=1:nb_iter
        if attack.clip
          adv_x(adv_x>1) = 1;
          adv_x(adv_x<0) = 0;
        end
        
        adv_x = fgm(adv_x, adv_obj, step_size, attack);
        eta = adv_x - pureX;
        eta = clip_eta(eta, attack.ord, eps);
        adv_x = pureX + eta;
      end
      
    case 'random'
      rng('shuffle')
      nb_iter     = attack.iters;
      [fea, num]  = size(pureX);
      
      adv_x      = zeros(fea, num);
      failed_idx = linspace(1, num, num);
      for i=1:nb_iter
        if isinf(attack.ord)
          adv_x(:,failed_idx) = pureX(:,failed_idx) + (rand(fea, length(failed_idx))*eps*2-eps);
        else
          adv_x(:,failed_idx) = pureX(:,failed_idx) + randomvector(length(failed_idx), fea, eps)';
        end
        
        % Predict on adversarial examples
        % If attack successfully, keep the perturbation
        if model.normalize
          KBadv = model.kernel_func(model.B, normc(adv_x));
        else
          KBadv = model.kernel_func(model.B, adv_x);
        end
        adv_logit  = model.W*(model.halfinvKBB*KBadv);
        [~, largestlogit] = max(adv_logit);
        adv_Pred   = largestlogit'-1;
        failed_idx = find(adv_Pred == pureY);
      end
      
    otherwise
      error('no such attacker!\n')
  end
  
  if attack.clip
    adv_x(adv_x>1) = 1;
    adv_x(adv_x<0) = 0;
  end
  [loss, ~] = adv_obj(adv_x);
end

function eta = clip_eta(eta, ord, eps)
  if isinf(ord)
    eta(eta>eps) = eps;
    eta(eta<-eps) = -eps;
  elseif ord == 2
    factor = min(1, eps./sqrt(sum(eta.^2)));
    eta = bsxfun(@times, eta, factor);
  else
    error('Only L-inf, L2 norms are currently implemented');
  end
end

function adv_x = fgm(X, adv_obj, eps, attack)
  [loss, grad] = adv_obj(X);
%   fprintf('loss: %f  nonzero_derivative: %f   l2norm_grad: %f\n', loss, mean(sum(grad==0)), mean(sum(grad.^2)));
  
  if isinf(attack.ord)
    adv_x = X + eps*sign(grad);
  elseif attack.ord == 2
    adv_x = X + eps*normc(grad);
  else
    error('Only L-inf and L2 norms are currently implemented');
  end
end


% Maximize loss(f(x/||x||), y)
function [loss, g] = attackloss(X, model, attack)
  if model.normalize
    KBX = model.kernel_func(model.B, normc(X));
  else
    KBX = model.kernel_func(model.B, X);
  end
  phi_x = model.halfinvKBB*KBX;
  invKBf = model.halfinvKBB*model.W';

  % gradient of l(\dots, y)
  fx = model.W*phi_x;
  [loss, A] = attack.lossfunc(fx);


  % gradient of l(x)
  switch model.kernel
    case 'inverse'
      inprod  = model.B'*normc(X);    i = model.layers;
      grad    = 1./(i+1-i*inprod).^2;
      coff    = (invKBf*A).*grad;
      l_x     = model.B*coff;
    case 'rbf'
      grad = KBX/model.bandwidth^2;
      coff = (invKBf*A).*grad;
      l_x  = model.B*coff;
    case 'gauss'
      grad = KBX/model.bandwidth^2;
      coff = (invKBf*A).*grad;
      l_x  = model.B*coff - bsxfun(@times, sum(coff), X);
  end
  
  % gradient of l(x)
  if model.normalize
    tmp = sqrt(sum(X.^2));
    g = bsxfun(@rdivide, l_x, tmp) - bsxfun(@times, sum(l_x.*X)./(tmp.^3), X);
  else
    g = l_x;
  end
end