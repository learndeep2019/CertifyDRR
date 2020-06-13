function data = data_preprocess_mnist(config)
% load mnist data
% return 
%   data.oriX           original training data
%   data.Y              training labels
%   data.oriXte         original test data
%   data.Yte            test labels
%   data.train_size     training data size
%   data.test_size      training data size
%   data.fea            number of features
  trainingimg    = ['data/' config.data.name '/train-images.idx3-ubyte'];
  traininglabel  = ['data/' config.data.name '/train-labels.idx1-ubyte'];
  testimg        = ['data/' config.data.name '/t10k-images.idx3-ubyte'];
  testlabel      = ['data/' config.data.name '/t10k-labels.idx1-ubyte'];
  [imgs, labels]  = readMNIST(trainingimg, traininglabel, config.data.train_size, 0, config.data.cropborder);
  Xtrain          = reshape(imgs, [], size(imgs, 3));
  
  [testimgs, testlabels]  = readMNIST(testimg, testlabel, config.data.test_size, 0, config.data.cropborder); 
%   [testimgs, testlabels]  = readMNIST(trainingimg, traininglabel, config.data.test_size, 0, config.data.cropborder);
  Xtest                   = reshape(testimgs, [], size(testimgs, 3));
 
  Ytrain = labels;
  Ytest  = testlabels;
  
  data.dataname   = config.data.name;
  data.oriX       = Xtrain;
  data.oriXte     = Xtest;
  data.Y          = Ytrain;
  data.Yte        = Ytest;
    
  data.num_fea    = size(Xtrain, 1);
  data.cropborder = config.data.cropborder; 
  data.train_size = size(Xtrain, 2);
  data.test_size  = size(Xtest, 2);
end
