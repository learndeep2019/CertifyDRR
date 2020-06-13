function [f, g] = kernel_classifier(W, model)
      KBW = model.kernel_func(model.B, W);
      model.invKBBalpha = model.halfinvKBB*model.alpha;
      f   = KBW'*model.invKBBalpha;
      
      if nargout > 1
        switch model.kernel
          case 'inverse'
            % 1 layer  f: 1/(2-x)        grad: 1/(2-x)^2  
            % 2 layers f: 1/(2-1/(2-x))  grad: 1/(3-2x)^2
            % 3 layers f: ...            grad: 1/(4-3x)^2
            inprod  = model.B'*W;    i = model.layers;
            gradKXW = 1./(i+1-i*inprod).^2;
            g       = model.B*bsxfun(@times, gradKXW, model.invKBBalpha);
          case 'rbf'
            coff    = bsxfun(@times, model.invKBBalpha, KBW);
            g       = model.B*coff/model.bandwidth^2;
          case 'gauss'
            coff    = bsxfun(@times, model.invKBBalpha, KBW);
            g       = (model.B*coff - bsxfun(@times, sum(coff), W))/model.bandwidth^2;
        end
      end
end