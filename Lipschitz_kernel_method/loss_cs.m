function [f, g] = loss_cs(z, y, kappa)
  truelabelidx = (y==1);
  [margindiff, maxlogitidx] = max(reshape(z(~truelabelidx), size(z)-[1, 0])); 
  [~, truelabelposition] = max(truelabelidx);
  maxlogitidx = maxlogitidx + (truelabelposition <= maxlogitidx);
  logits_margins = kappa + margindiff - z(truelabelidx)';
  f = sum( max(0, logits_margins) );
  active = double(logits_margins > 0);
  g      = zeros(size(z));
  g( sub2ind(size(z),maxlogitidx,1:size(y,2)) ) = active;
  g(truelabelidx) = -active;
end

% y = [0, 1, 0, 0; 1, 0 ,0, 0]';
% fun = @(x)loss_crossent(x, y, 0);
% sol = [0, 1, -0.5, 0.5; 1, 0.2, 0.1, -0.6]';
% [f, g] = fun(sol);
% [grad, err] = gradest(fun, sol);
% [g(:) grad(:)]
% fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));