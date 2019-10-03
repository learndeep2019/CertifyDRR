% Save model
% To save memory, only dump neccessary data
function save_model(log, model, algo, data)
    
%     if strcmp(algo.task, 'multiclass') && data.pos_digit ~= 0
%       model = rmfield(model, 'B');
%     end
    model = rmfield(model, 'B');
    fields = {'kernel_func', 'num_bases', 'halfinvKBB', 'KBX', 'KBXte', 'PhiX'};
    model  = rmfield(model, fields);
    save([log.root  '/model.mat'], 'model',  '-v7.3')
end

