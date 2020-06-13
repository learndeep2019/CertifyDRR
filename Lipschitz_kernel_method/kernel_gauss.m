function f = kernel_gauss(x, y, sigmasq) 
  inprod = x'*y;
  num_y  = size(y, 2);
  num_x  = size(x, 2);
  K = repmat(sum(x.^2)', 1, num_y) + repmat(sum(y.^2), num_x, 1) - 2 * inprod;
  f = exp(K / (-2*sigmasq));
end

