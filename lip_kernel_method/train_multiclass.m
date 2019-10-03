function train_multiclass(algo, model, data, attack, log)
  if ~exist(log.root, 'dir')
    mkdir(log.root);
  end
  
  if log.islog
    log.fileID     = fopen([log.root '/' log.filename],'w');
  else
    log.fileID     = 1;   % output on screen
  end
  log.lipfileID    = fopen([log.root '/poly_' num2str(algo.poly) '_greedy_' num2str(algo.greedysampling) ...
                                          '_L' algo.defense_ord 'Lip_converge.log'],'w');

  % Init loss function                                      
  true_prob = zeros(10, length(data.Y));
  for i=1:length(data.Y)
    true_prob(data.Y(i)+1, i) = 1;
  end
  switch algo.loss
    case 'cs'
      algo.lossfunc = @(z)loss_cs(z, true_prob, algo.margin);
    case 'ww'
      algo.lossfunc = @(z)loss_ww(z, true_prob, algo.margin);
    case 'crossent'
      algo.lossfunc = @(z)loss_crossent(z, true_prob, algo.margin);
    otherwise
      error('Not implemented loss\n');
  end
  
  if isfield(model, 'layers') 
    kernel_para = model.layers;
  else
    kernel_para = model.bandwidth;
  end
  model.classes = length(unique(data.Y));
  
  % Training
  fprintf(log.fileID, '-----------------------Multiclass Classification Task-----------------------\n');
  fprintf(log.fileID, 'Data  |  trainsize  = %5d, testsize  = %5d, features = %d\n', data.train_size, data.test_size, data.num_fea);
  fprintf(log.fileID, 'Algo  |  lip_constr = %5d, rkhs_norm = %5d, margin   = %d, iters = %d\n', algo.max_lip, algo.rkhs_norm, algo.margin, algo.iter);
  fprintf(log.fileID, 'Model |  kernel = %s, parameter = %3.1f, defense_ord = L%s\n', model.kernel, kernel_para, algo.defense_ord);
  fprintf(log.fileID, '----------------------------------------------------------------------------\n');
  
  start = tic;
  if algo.max_lip
    if strcmp(algo.defense_ord, 'inf')
      model = Solver_multiclass_linf(algo, model, data, log, attack);
    else
      model = Solver_multiclass_l2(algo, model, data, log, attack);
    end
  else
    model = Solver_multiclass_noconstr(algo, model, data, log, attack);
  end

  fprintf(log.fileID, '-----------Multiclass Task finished in %.1f seconds!-----------\n', toc(start));
  
  if log.islog
    fclose(log.fileID);
  end

  %% Saving model
  if log.save_model
    save_model(log, model, algo, data);
  end
end


