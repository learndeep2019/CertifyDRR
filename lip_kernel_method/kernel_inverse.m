function [f, g] = kernel_inverse(x, y, layers)
    if layers < 1
        error('Invalid layers for inverse kernel!');
    end
    
    f = x'*y;
    f = (layers-(layers-1)*f)./(layers+1-layers*f);
%     g = ones(size(f));
%     g = g.*(2-f).^(-2);
        
end

