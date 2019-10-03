function [xvec, lipvec] = maxLip_multiclass(model, data, algo)
    num_fea = size(model.B, 1);
    param = [];
    param.maxIter = 50;     % max number of iterations
    param.maxFnCall = 100;  % max number of calling the function
    param.relCha = 1e-5;      % tolerance of constraint satisfaction
    param.tolPG = 1e-5;   % final objective function accuracy parameter
    param.m = 100;
    ub = Inf(num_fea, 1);   lb = -ub;
    show_iter = @(t, f, x)perf_iter(t, f, x);
    

    t = 10;
    xvec = [];
    lipvec = [];
%     initpoints = 6.*rand(num_fea, t) - 3;
    
    % init with train points of largest linf norm
    if strcmp(algo.defense_ord, 'inf')
      halfinvW   = model.halfinvKBB*model.W;
      switch model.kernel
        case 'inverse'
          inprod  = model.B'*data.X;    i = model.layers;
          gradKXW = 1./(i+1-i*inprod).^2;
          linfvec = zeros(size(data.X, 2), 1);
          for i = 1:size(data.X, 2)
            gradf       = model.B*bsxfun(@times, halfinvW, gradKXW(:, i));
            linfvec(i)  = norm(gradf',Inf);
          end
        case 'gauss'
          coffsum    = model.KBX'*halfinvW;
          linfvec    = zeros(size(data.X, 2), 1);
          for i = 1:size(data.X, 2)
            gradf       = (model.B*bsxfun(@times, halfinvW, model.KBX(:, i))  - data.X(:,i)*coffsum(i,:))/model.bandwidth^2;
            linfvec(i)  = norm(gradf',Inf);
          end
      end
      
      [~, idx] = sort(linfvec, 'descend');
      initpoints = data.X(:, idx(1:t)); 
      
      obj = @(x)Linfnorm_obj(x, model);
      
    else  % init with train points of largest spectral norm
      halfinvW   = model.halfinvKBB*model.W;
      switch model.kernel
        case 'inverse'
          inprod  = model.B'*data.X;    i = model.layers;
          gradKXW = 1./(i+1-i*inprod).^2;
          specvec = zeros(size(data.X, 2), 1);
          for i = 1:size(data.X, 2)
            gradf       = model.B*bsxfun(@times, halfinvW, gradKXW(:, i));
            specvec(i)  = svds(gradf, 1);
          end
        case 'gauss'
          coffsum    = model.KBX'*halfinvW;
          specvec    = zeros(size(data.X, 2), 1);
          for i = 1:size(data.X, 2)
            gradf       = (model.B*bsxfun(@times, halfinvW, model.KBX(:, i))  - data.X(:,i)*coffsum(i,:))/model.bandwidth^2;
            specvec(i)  = svds(gradf, 1);
          end
      end
      
      [~, idx] = sort(specvec, 'descend');
      initpoints = data.X(:, idx(1:t)); 
      
      obj = @(x)spectralnorm_obj(x, model);
    end
    
    for i=1:t
      [x, fval] = lbfgsb(initpoints(:, i), lb, ub, obj, [], [], param);
      if model.normalize
        x = x./norm(x);
      end
  
      if i==1 || isunique(xvec, x)
        xvec = [xvec, x];
        lipvec = [lipvec, -fval];
      end
    end
end

function [f, g] = spectralnorm_obj(y, model)
  num_fea = size(model.B, 1);

  halfinvW = model.halfinvKBB*model.W;
  switch model.kernel
    case 'inverse'
      x       = y./norm(y);
      inprod  = model.B'*x;    i = model.layers;
      gradKXW = 1./(i+1-i*inprod).^2;
      gradf   = model.B*bsxfun(@times, halfinvW, gradKXW);
    case 'gauss'
      KXW   = model.kernel_func(model.B, y);
      coff  = bsxfun(@times, halfinvW, KXW);
      gradf = (model.B*coff - y*sum(coff))/model.bandwidth^2;
  end
  
  [u1,s,v1] = svds(gradf, 1);
  f = -s;
  gradW = u1*v1';
  
  if model.normalize
    G = model.B';
    normy = norm(y);
    temp = G/normy-G*(y*y')/normy^3;
  end
  g = zeros(1, num_fea);
  for i = 1:model.classes
    switch model.kernel
      case 'inverse'
        inprod = model.B' * x;    k = model.layers;
        a = bsxfun(@times, halfinvW(:,i).*(2*k./(k+1-k*inprod).^3), model.B');
        secondgradfi_x = a'*temp;
      case 'gauss'
        s = bsxfun(@minus, model.B, y);
        a = bsxfun(@times, (halfinvW(:,i).*KXW)', s)*s'/model.bandwidth^4 ...
                          - (halfinvW(:,i)'*KXW)*eye(length(y))/model.bandwidth^2;
        secondgradfi_x = a';
    end
    g = g + gradW(:, i)'*secondgradfi_x;
  end
  g = -g;
end

function [f, g] = Linfnorm_obj(y, model)
  halfinvW = model.halfinvKBB*model.W;
  switch model.kernel
    case 'inverse'
      x   = y./norm(y);
      inprod  = model.B'*x;    i = model.layers;
      gradKXW = 1./(i+1-i*inprod).^2;
      gradf = model.B*bsxfun(@times, halfinvW, gradKXW);
    case 'gauss'
      KXW = model.kernel_func(model.B, y);
      coff  = bsxfun(@times, halfinvW, KXW);
      gradf = (model.B*coff - y*sum(coff))/model.bandwidth^2;
  end
  
  [f, idx] = max(sum(abs(gradf)));
  f = -f;
  gradW = zeros(size(gradf));
  gradW(:,idx) = sign(gradf(:, idx));
  
  if model.normalize
    G = model.B';
    normy = norm(y);
    temp = G/normy-G*(y*y')/normy^3;
  end
  switch model.kernel
    case 'inverse'
      inprod = model.B' * x;    k = model.layers;
      a = bsxfun(@times, halfinvW(:,idx).*(2*k./(k+1-k*inprod).^3), model.B');
      secondgradfi_x = a'*temp;
    case 'gauss'
      s = bsxfun(@minus, model.B, y);
      a = bsxfun(@times, (halfinvW(:,idx).*KXW)', s)*s'/model.bandwidth^4 ...
        - (halfinvW(:,idx)'*KXW)*eye(length(y))/model.bandwidth^2;
      secondgradfi_x = a';
  end
  g = gradW(:, idx)'*secondgradfi_x;
  g = -g;
end

function perf_iter(t, f, x)
  fprintf('%3d: f = %0.5g\n', t, f);
  pause(0.1);
end

function flag = isunique(xvec, x)
  flag = 1;
  [num_fea, len] = size(xvec);
  for i = 1:len
    if ~(sum(abs(xvec(:, i) - x) > 0.01) > num_fea/8)
      flag = 0;
    end
  end
end