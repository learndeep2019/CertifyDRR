function attack_script(config)
  if ~nargin
    clear;
    rng(1);
    config = jsondecode(fileread('config.json'));
  end
  [~, model, data, attack, log] = parser_parameter(config);
  
  attack.root = log.root;
  if attack.islog
    if attack.ord == 2
      attack.fileID  = fopen([attack.root '/l2_' attack.method '_' attack.loss '_test_log.txt'],'w');
    elseif isinf(attack.ord)
      attack.fileID  = fopen([attack.root '/linf_' attack.method '_' attack.loss '_test_log.txt'],'w');
    end
  else
    attack.fileID  = 1;
  end
 
  attack_multiclass_clf(model, data, attack);
end



function attack_multiclass_clf(model, data, attack)
  %% load model
  if exist(attack.root, 'dir')
    trained_model = load([attack.root  '/model.mat']);
    model.final_lip  = trained_model.model.final_spec;
    model.W          = trained_model.model.W';
  else
    error('no pretrained model\n')
  end
  
  fprintf(attack.fileID, 'final L%s = %f\n', num2str(attack.ord), model.final_lip);
  
  
  %% Compute training error and test error
  KBX = model.kernel_func(model.B, data.X);
  lastsecondlayer = model.halfinvKBB*KBX;
  logit = model.W*lastsecondlayer;
  [~, largestmargin] = max(logit);
  pred_label = largestmargin-1;
  train_acc = mean(pred_label' == data.Y)*100;

  KBXte = model.kernel_func(model.B, data.Xte);
  lastsecondlayer = model.halfinvKBB*KBXte;
  logit = model.W*lastsecondlayer;
  [~, largestmargin] = max(logit);
  pred_label = largestmargin-1;
  test_acc = mean(pred_label' == data.Yte)*100;

  fprintf(attack.fileID, 'train_acc = %f%%, test_acc = %f%%\n', train_acc, test_acc);


  
  %% Attack
  switch attack.dataname  % Attack train data
    case 'train'
      true_prob = zeros(10, length(data.Y));
      for i=1:length(data.Y)
        true_prob(data.Y(i)+1, i) = 1;
      end
      if strcmp(attack.loss, 'cw')
        attack.lossfunc = @(z)loss_cw(z, true_prob, attack.margin);
      else
        attack.lossfunc = @(z)loss_crossent(z, true_prob, attack.margin);
      end
      if model.normalize
        attack_main(model, data.oriX, data.Y, attack, 'train');
      else
        attack_main(model, data.X, data.Y, attack, 'train');
      end
      
    case 'test'    % Attack test data
      true_prob = zeros(10, length(data.Yte));
      for i=1:length(data.Yte)
        true_prob(data.Yte(i)+1, i) = 1;
      end
      if strcmp(attack.loss, 'cw')
        attack.lossfunc = @(z)loss_cw(z, true_prob, attack.margin);
      else
        attack.lossfunc = @(z)loss_crossent(z, true_prob, attack.margin);
      end
      if model.normalize
        attack_main(model, data.oriXte, data.Yte, attack, 'test');
      else
        attack_main(model, data.Xte, data.Yte, attack, 'test');
      end
      
    case 'plot_grad'
        gradient_plot(model, data.X, data.Y, attack);
  end
end
