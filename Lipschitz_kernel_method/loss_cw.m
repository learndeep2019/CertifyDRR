function [f, g] = loss_cw(z, y, margin)
  truelabelidx  = (y==1);
  correct_logit = z(truelabelidx);
  tmp = z;
  tmp(truelabelidx) = -inf;
  [worst_wrong_logit, idx] = max(tmp);
  
  adv_margin = correct_logit' - worst_wrong_logit;
  f  = -sum(max(adv_margin + margin, 0))/size(y, 2);
  xx = (adv_margin + margin > 0);
  relugrad = -double(xx);
  a  = zeros(size(z));
  a(sub2ind(size(z),idx,1:size(a,2))) = 1;
  adv_margin_grad = y - a;
  g = bsxfun(@times, relugrad, adv_margin_grad)/size(y, 2);
end

% y = [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.; 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]';
% fun = @(x)loss_cw(x, y);
% sol = [ 1.2669, -2.4148, -3.8249, -4.4015, -0.3374, -0.7502, -4.3116, -2.5288, -2.8040, -0.2770; 0.6018, -2.7256, -3.7101, -5.0313, -2.8608, -1.7691,  1.1948, -6.3999, -0.8025, -1.0450]';
% [f, g] = fun(sol);
% [grad, err] = gradest(fun, sol);
% [g(:) grad(:)]
% fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));