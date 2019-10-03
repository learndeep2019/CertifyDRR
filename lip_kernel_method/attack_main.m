function attack_main(model, pureX, pureY, attack, dataname)
  
  fprintf(attack.fileID, '-----------------------ATTACKING start-----------------------\n');
  fprintf(attack.fileID, 'Data   |  dataname  = %s, size = %5d, attack_ord = %d\n', dataname, numel(pureY), attack.ord);
  fprintf(attack.fileID, 'Attack |  method    = %s,  loss = %s, margin = %4d\n', attack.method, attack.loss, attack.margin);
  fprintf(attack.fileID, '       |  iters     = %3d,  rand_init = %s,    clip = %s\n', attack.iters, num2str(attack.rand_init), num2str(attack.clip));
  fprintf(attack.fileID, '------------------------------------------------------------\n');


  % Precompute stats on original data
  if strcmp(dataname, 'train')
    KBx = model.KBX;
  else
    KBx = model.KBXte;
  end
  
  logit              = model.W*(model.halfinvKBB*KBx);
  sorted_logit       = sort(logit, 'descend');
  statOri.margin     = sorted_logit(1,:) - sorted_logit(2,:);
  [~, largestlogit]  = max(logit);
  orix_Pred          = largestlogit'-1;
    
  statOri.c_idx        = (orix_Pred == pureY);  % correctly labeled Index
  statOri.orix_Pred    = orix_Pred;
  statOri.margins_mean = mean(abs(statOri.margin));
  
  normsquare_x         = sum(pureX.^2);
  statOri.norm_x       = sqrt(normsquare_x);
%   statOri.l2normLip    = (norm_x-1)./normsquare_x;
  
  
  % Attacking with different perturbation radius
  eps_sets    = attack.eps;
  num_radius  = numel(eps_sets);
  stat        = cell(num_radius, 1);
  parfor (i = 1:num_radius, str2num(attack.workers))
    eps     = eps_sets(i);
    stat{i} = attack_eps(model, statOri, pureX, pureY, eps, attack);
  end
  
  % Reporting some stats
  fprintf(attack.fileID,   '        def_rate    acc   margin_mean  adv_margins  mean_L2  mean_Linf\n');
  for i = 1:num_radius
    fprintf(attack.fileID, '%.2f     %.2f     %.2f     %.2f         %.2f      %.2f      %.2f\n' , ...
                        stat{i}.eps, stat{i}.defense_rate, stat{i}.total_accuracy, statOri.margins_mean, ...
                        stat{i}.successful_margins, stat{i}.mean_L2, stat{i}.mean_Linf);
  end
  
  fprintf('-------------ATTACKING finished-------------\n\n');
  
  
  % Dump some results into json file
  if attack.islog
    fclose(attack.fileID);
    output.datasize      = numel(pureY);
    output.accuracy      = 100*mean(statOri.c_idx);
    output.final_lip     = model.final_lip;

    output.adv_acc       = zeros(num_radius, 1);
    output.theoretic_acc = zeros(num_radius, 1);
    for i = 1:num_radius
      output.adv_acc(i)       = stat{i}.total_accuracy;
      output.theoretic_acc(i) = stat{i}.theoretic_acc;
    end

    str = jsonencode(output);
    str = strrep(str, ':[', sprintf(':[\r'));
    str = strrep(str, ',',  sprintf(',\r'));
    if attack.ord == 2
      fid = fopen([attack.root '/l2_' attack.method '_' attack.loss '_' dataname '_results.json'], 'w');
    elseif isinf(attack.ord)
      fid = fopen([attack.root '/linf_' attack.method '_' attack.loss '_' dataname '_results.json'], 'w');
    end
    fwrite(fid, str, 'char');
    fclose(fid);
  end
  
  % display some successful attack figures
%   realimgs = pureX(:, s_idx);
%   advimags = adv_x(:, s_idx);
%   
%   if sum(s_idx) > 49
%     realimgs = realimgs(:, 1:49);
%     advimags = advimags(:, 1:49);
%   end
%   display_imgs(realimgs, 'real');
%   display_imgs(advimags, 'adv');
end

function [stat] = attack_eps(model, statOri, pureX, pureY, eps, attack)
    stat.eps   = eps;
    % Generating adversarial examples
    adv_x      = attack_model(model, pureX, pureY, eps, attack);

    % Predict on adversarial examples
    if model.normalize
      KBadv = model.kernel_func(model.B, normc(adv_x));
    else
      KBadv = model.kernel_func(model.B, adv_x);
    end

    adv_logit  = model.W*(model.halfinvKBB*KBadv);
    [~, largestlogit] = max(adv_logit);
    adv_Pred   = largestlogit'-1;
    
    % Compute attack success rate
    s_idx               = (adv_Pred ~= statOri.orix_Pred) & statOri.c_idx;   % Index of successful attacked examples
    success_rate        = sum(s_idx)*100/length(statOri.c_idx);
    stat.defense_rate   = 100 - success_rate;
    stat.total_accuracy = mean(adv_Pred == pureY)*100;
    
    % Compute avg margin
    stat.successful_margins = mean(abs(statOri.margin).*s_idx');
    stat.mean_L2            = mean(sqrt(sum((adv_x-pureX).^2)));
    stat.mean_Linf          = mean(max(abs(adv_x-pureX)));

end
