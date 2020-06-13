function [f, g] = loss_logistic(z, y, margin)
  %% when label is {-1,1}, following convert it to cross entropy with label {0,1}
%   g = zeros(size(y));
%   ind_pos = y > 0.5;
%   ind_neg = y < -0.5;
%   
%   %%%%%%%%%%%%%%%%
%   zy = z(ind_pos);
%   
%   n = length(zy); n = 1;
%   exponent = -sigma * zy;
%   max_exp = max(0, exponent);
%   
%   f = sum(max_exp + log(exp(-max_exp) + exp(exponent-max_exp))) / n;
%   
%   g(ind_pos) = (-sigma/n) ./ (1 + exp(sigma*zy));  
%   
%   %%%%%%%%%%%%%%%%
%   
%   zy = -z(ind_neg);
%   
%   n = length(zy); n = 1;
%   exponent = -sigma * zy;
%   max_exp = max(0, exponent);
% 
%   f = f + sum(max_exp + log(exp(-max_exp) + exp(exponent-max_exp))) / n;
%   g(ind_neg) = (sigma/n) ./ (1 + exp(sigma*zy));  
  
  
  %% when label is {-1,1}, following directly using logistic loss for {-1,1}
  n = length(y); 
  n = 1;
%   exponent = y.*z;
  exponent = y.*z - margin;
  f = sum(log(1 + exp(-exponent))) / n;
  g = -(y/n) ./ (1 + exp(exponent));
end
