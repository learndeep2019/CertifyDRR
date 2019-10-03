function data = data_preprocess_cifar10(config)
  data.dataname   = config.data.name;
  
  % read raw data
%   batches = ceil(config.data.train_size/10000);
%   data.oriX = [];
%   data.Y    = [];
%   for i = 1:batches
%     trainbatch = ['data/cifar10/data_batch_' num2str(i)];
%     train_matrix  = load(trainbatch);
%     data.oriX     = [data.oriX, train_matrix.data'];
%     data.Y        = [data.Y; train_matrix.labels];
%   end
%   data.oriX   = double(data.oriX(:, 1:config.data.train_size))/255;
%   data.Y      = double(data.Y(1:config.data.train_size));
%   
%   test_matrix = load('data/cifar10/test_batch');
%   data.oriXte = double(test_matrix.data(1:config.data.test_size, :)')/255;
%   data.Yte    = double(test_matrix.labels(1:config.data.test_size));
  
  
  % read embedding
  root    = 'data/cifar10_embedding/';
  data.oriX   = double(readNPY([root 'train_inps_embedding.npy'])');
  data.Y      = readNPY([root 'train_tgts_embedding.npy']);
  data.oriX   = data.oriX(:, 1:config.data.train_size);
  data.Y      = data.Y(1:config.data.train_size);
  
  
  data.oriXte = double(readNPY([root 'test_inps_embedding.npy'])');
  data.Yte    = readNPY([root 'test_tgts_embedding.npy']);
  data.oriXte = data.oriXte(:, 1:config.data.test_size);
  data.Yte    = data.Yte(1:config.data.test_size);
  
  
  % data args
  data.train_size = config.data.train_size;
  data.test_size  = config.data.test_size;
  data.num_fea    = size(data.oriX, 1);
end