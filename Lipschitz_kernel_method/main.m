function main()
  % main function only accept one config
  clear;
  clear global;
  rng(1);
  config = jsondecode(fileread('config.json'));
  
  % parse config.json
  [algo, model, data, attack, log] = parser_parameter(config);
  train_multiclass(algo, model, data, attack, log);
  attack_script(config);
end
