function model = Solver_multiclass_noconstr(algo, model, data, log, attack)
    tic;
    fun = @(x)lossobj(x, model, algo);
    W0  = rand(model.num_bases*model.classes, 1)*0.5;

%     [f, g] = fun(W0);
%     [grad, err] = gradest(fun, W0);
%     [g(:) grad(:)]
%     fprintf('diff = %g, err = %g\n', max(abs(g(:) - grad(:))), max(abs(err)));
    
    options = optimoptions(@fmincon,'Algorithm','interior-point', ...
        'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true, ...
        'MaxFunctionEvaluations', 600, 'optimalityTolerance',1e-2, ...
        'HessianApproximation', 'lbfgs', 'SubproblemAlgorithm', 'cg', ...
        'StepTolerance', 1e-8, 'Display','final');
    [W,fval,eflag,output] = fmincon(fun,W0,[],[],[],[],[],[],[],options);

    
    
    fprintf(log.fileID, '#iter: %d, #funccall: %d, flag: %d, loss: %.3f, cputime: %.2f \n', ...
                        output.iterations, output.funcCount, eflag, fval, toc);
    model.W = reshape(W, model.num_bases, model.classes);
    
    % Sampling new points and add to constraints
    [~, lbfgs_lip] = maxLip_multiclass(model, data, algo);
    
    % Print largest lipschitz constant
    fprintf(log.fileID, 'Max_lbfgs_Lip: %f\n', max(lbfgs_lip));
    model.final_spec = max(lbfgs_lip);
    
   
    % Compute training error and test error
    lastsecondlayer   = model.halfinvKBB*model.KBX;
    logit             = model.W'*lastsecondlayer;
    [~, largestlabel] = max(logit);
    pred_label        = largestlabel-1;
    train_acc         = mean(pred_label' == data.Y)*100;
    sorted_logit      = sort(logit, 'descend');
    train_margin      = mean(sorted_logit(1,:) - sorted_logit(2,:));
    
    
    lastsecondlayer   = model.halfinvKBB*model.KBXte;
    logit             = model.W'*lastsecondlayer;
    [~, largestlabel] = max(logit);
    pred_label        = largestlabel-1;
    test_acc          = mean(pred_label' == data.Yte)*100;
    sorted_logit      = sort(logit, 'descend');
    test_margin       = mean(sorted_logit(1,:) - sorted_logit(2,:));
    
    fprintf(log.fileID, 'Train_acc    Test_acc        Train_margin    Test_margin\n');
    fprintf(log.fileID, '%8.3f%%    %8.3f%%    %12.3f    %12.3f\n\n\n', ...
                        train_acc, test_acc, train_margin, test_margin);

                      
    % Attack train data
%     attack.lossfunc = algo.lossfunc;
%     attack_main(model, data.oriX, data.Y, attack, 'train');
    
    pause(0.2);
end


% Loss objective
function [y,grady] = lossobj(W, model, algo)
  mW = reshape(W, model.num_bases, model.classes);
  [loss, g] = algo.lossfunc(mW'*model.PhiX);
  y = loss;
  grady = model.PhiX*g';
  grady = grady(:);
end