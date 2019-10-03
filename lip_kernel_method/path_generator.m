function save_path = path_generator(model, algo, data)
  if strcmp(algo.task, 'binary')
    root = ['out/binary/' data.dataname '_' num2str(data.pos_digit)  'vs' num2str(data.neg_digit)];
  elseif strcmp(algo.task, 'multiclass')
    root = ['out/multiclass/' data.dataname];
  end
  if strcmp(model.kernel, 'inverse')
    kernelpara = num2str(model.layers);
  else
    kernelpara = num2str(model.bandwidth);
  end
  
  save_path = [ root ...
                '_size_'   num2str(data.train_size) ...
                '_nystrom_' num2str(model.num_bases) ...
                '_L'       algo.defense_ord '_trained_' ...
                '_kernel_' model.kernel '_' kernelpara ...
                '_lip_'    num2str(algo.max_lip) ...
                '_rkhs_'   num2str(algo.rkhs_norm) ...
                '_margin_' num2str(algo.margin)  ...
                ];
  
  if ~exist(save_path, 'dir')
    mkdir(save_path);
  end
end

