% function gradient_plot(model, pureX, pureY, dataname)
% % Generate gradient and provide it to Cem's code.
%   pureX = pureX(:, 100:end);
%   pureY = pureY(100:end);
%   idx = zeros(10, 1);
%   c = 0;
%   margin = 100;
%   for i = 1:numel(pureY)
%     if pureY(i) == c
%       c = c + 1;
%       idx(c) = i;
%       if c == 10
%         break
%       end
%     end
%   end
%   random_x = pureX(:, idx);
%   
%   G = zeros(10, 28, 28);
%   G(1:10, :,:) = permute(reshape(random_x, [28, 28, 10]), [3 1 2]);
%   for i = 1:10
%     true_prob = zeros(10, 10);
%     true_prob(i, :) = 1;
% %     opt.loss_sup = @(z)loss_cw(z, true_prob, margin);
%     opt.loss_sup = @(z)loss_crossent(z, true_prob, margin);
%     [~, g] = objective(random_x, model, opt);
%     a = permute(reshape(g, [28, 28, 10]), [3 1 2]);
%     aa = ones(size(a));
%     aa(:,:,1) = -1;
%     G(10*i+1:10*(i+1),:,:) = aa;
%   end
%   
%   save G.mat G
% end


function gradient_plot(model, pureX, pureY, attack)
% Generate gradient and provide it to Cem's code.
  pureX = pureX(:, 100:end);
  pureY = pureY(100:end);
  idx = zeros(10, 1);
  c = 0;
  margin = 0;
  for i = 1:numel(pureY)
    if pureY(i) == c
      c = c + 1;
      idx(c) = i;
      if c == 10
        break
      end
    end
  end
  pureX = pureX(:, idx);
  nb_iter   = 10;
  eps       = 6;
  step_size = 2*eps/nb_iter;
  
  PXinterm = cell(nb_iter, 1);
  for i = 1:10
    PXinterm{i} = pureX;
  end
  PX       = cell(nb_iter, 1);
  Grad     = cell(nb_iter, 1);
  for j = 1:nb_iter
    G = zeros(10, 28, 28);
    P = zeros(10, 28, 28);
    G(1:10, :,:) = permute(reshape(pureX, [28, 28, 10]), [3 1 2]);
    P(1:10, :,:) = permute(reshape(pureX, [28, 28, 10]), [3 1 2]);
    for i = 1:10
      true_prob = zeros(10, 10);
      true_prob(i, :) = 1;
  %     opt.loss_sup = @(z)loss_cw(z, true_prob, margin);
      opt.loss_sup = @(z)loss_crossent(z, true_prob, margin);
      [loss, g] = objective(PXinterm{i}, model, opt); fprintf('%f \t', loss);
      random_x = fgm(PXinterm{i}, g, step_size, attack);
      eta = random_x - pureX;
      eta = clip_eta(eta, attack.ord, eps);
      PXinterm{i} = pureX + eta;
      a = permute(reshape(normc(g), [28, 28, 10]), [3 1 2]);
      G(10*i+1:10*(i+1),:,:) = a;
      P(10*i+1:10*(i+1),:,:) = permute(reshape(PXinterm{i}, [28, 28, 10]), [3 1 2]);
    end
    fprintf('\n');
    Grad{j} = G;
    PX{j}   = P;
  end
  save('G.mat', 'Grad', 'PX');
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

function adv_x = fgm(X, grad, eps, attack)
  if isinf(attack.ord)
    adv_x = X - eps*sign(grad);
  elseif attack.ord == 2
    adv_x = X - eps*normc(grad);
  else
    error('Only L-inf and L2 norms are currently implemented');
  end
end

function [loss, g] = objective(X, model, opt)
  if model.normalize
      KBX = model.kernel_func(model.B, normc(X));
  else
      KBX = model.kernel_func(model.B, X);
  end
    phi_x = model.halfinvKBB*KBX;
    invKBf = model.halfinvKBB*model.W';
    
    % gradient of l(\dots, y)
    fx = model.W*phi_x;
    [loss, A] = opt.loss_sup(fx);  
    
    
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
    if model.normalize
        tmp = sqrt(sum(X.^2));
        g = bsxfun(@rdivide, l_x, tmp) - bsxfun(@times, sum(l_x.*X)./(tmp.^3), X);
    else
        g = l_x;
    end    
%   tmp = sqrt(sum(X.^2));
%   g = bsxfun(@rdivide, l_x, tmp) - bsxfun(@times, sum(l_x.*X)./(tmp.^3), X);
%   g = l_x;
end

