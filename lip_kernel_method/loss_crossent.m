function [f, g] = loss_crossent(z, y, margin)
%   [~, largestlogit]  = max(z);
%   largestidx = sub2ind(size(z),largestlogit,1:size(y,2));
%   z(largestidx) = z(largestidx) - margin;

  truelabelidx = (y==1);
  z(truelabelidx) = z(truelabelidx) - margin;
  prob  = softmax(z);
  f = sum(-sum(y.*log(prob)));
  g = prob-y;
end

% y = [0, 1, 0, 0; 1, 0 ,0, 0]';
% fun = @(x)loss_crossent(x, y, 0);
% sol = [0, 1, -0.5, 0.5; 1, 0.2, 0.1, -0.6]';
% [f, g] = fun(sol);
% [grad, err] = gradest(fun, sol);
% [g(:) grad(:)]
% fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));