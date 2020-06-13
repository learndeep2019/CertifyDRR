function vec = randomvector(m, n, r)
    if nargin <= 2
      r = 1;
    end
    X = randn(m,n);
    s2 = sum(X.^2,2);
    vec = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);
end