function [f, g] = loss_ww(z, y, kappa)
  truelabelidx = (y==1);
  logits_margins = kappa + bsxfun(@minus, z, z(truelabelidx)');
  f = sum( sum( max(0, logits_margins) ) - kappa );
  g = double(logits_margins > 0);
  g(truelabelidx) = -sum(g) + 1;
end

% y = [0, 1, 0, 0; 1, 0 ,0, 0]';
% fun = @(x)loss_crossent(x, y, 0);
% sol = [0, 1, -0.5, 0.5; 1, 0.2, 0.1, -0.6]';
% [f, g] = fun(sol);
% [grad, err] = gradest(fun, sol);
% [g(:) grad(:)]
% fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));