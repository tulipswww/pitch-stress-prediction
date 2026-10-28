clearvars -except corner_model_cache; clc;
script_dir=fileparts(mfilename('fullpath'));
data_stamp=dir(fullfile(script_dir,'corner_training_data.xlsx'));
script_stamp=dir(fullfile(script_dir,'fit_corner_model.m'));
needs_fit=~exist('corner_model_cache','var') || ~isstruct(corner_model_cache);
if ~needs_fit
    needs_fit=~isfield(corner_model_cache,'source_folder') || ...
        ~isfield(corner_model_cache,'source_data_stamp') || ...
        ~isfield(corner_model_cache,'source_script_stamp');
end
if ~needs_fit
    needs_fit=~strcmpi(corner_model_cache.source_folder,script_dir) || ...
        ~isequal(corner_model_cache.source_data_stamp,[data_stamp.datenum,data_stamp.bytes]) || ...
        ~isequal(corner_model_cache.source_script_stamp,[script_stamp.datenum,script_stamp.bytes]);
end
if needs_fit
    fprintf('工作区无当前模型，先运行fit_corner_model完成拟合。\n');
    run(fullfile(script_dir,'fit_corner_model.m'));
end

model_info=corner_model_cache;
% 使用 fit_corner_model 的同一份清洗样本和完整精度系数，不再独立清洗或重拟合。
tbl = model_info.training_table;
terms = reshape(model_info.model_terms,1,[]);
X_mat = [ones(height(tbl),1),tbl{:,terms}];
c_opt = [model_info.model_intercept;model_info.model_coefficients(:)];
term_names = [{'Constant'},terms];
Y_true = tbl.Y;
Y_pred = X_mat*c_opt;
sigma_true = exp(Y_true);
sigma_pred = exp(Y_pred);
base_R2_log = corner_r2(Y_true,Y_pred);
base_R2 = corner_r2(sigma_true,sigma_pred);
base_MAPE = mean(abs(sigma_pred-sigma_true)./sigma_true)*100;
target_R2 = 0.70;
fprintf('基准 N=%d | R2(Y)=%.8f | R2(sigma)=%.8f | MAPE=%.4f%%\n', ...
    height(tbl),base_R2_log,base_R2,base_MAPE);
if ~isfinite(base_R2) || base_R2 < target_R2
    error('基准真实应力 R2 未达到 %.2f，不能计算该阈值下的系数容许区间。',target_R2);
end
fprintf('单系数扰动：其余系数固定，以真实应力 R2(sigma)>=%.2f 为准。\n',target_R2);
fprintf('参数       | 最优系数 | 下界 | 上界 | 半区间相对宽度ri(%%)\n');
sensitivity_results = table();
for i=1:numel(c_opt)
    [c_min,c_max] = coefficient_bounds(X_mat,c_opt,sigma_true,i,target_R2);
    if abs(c_opt(i))>1e-6
        relative_width = (c_max-c_min)/(2*abs(c_opt(i)))*100;
    else
        relative_width = NaN;
    end
    low = c_opt; low(i)=c_min;
    high = c_opt; high(i)=c_max;
    if isfinite(c_min), R2_lower=corner_r2(sigma_true,exp(X_mat*low)); else, R2_lower=NaN; end
    if isfinite(c_max), R2_upper=corner_r2(sigma_true,exp(X_mat*high)); else, R2_upper=NaN; end
    fprintf('%-10s | %.8f | %.8f | %.8f | %.4f\n', ...
        term_names{i},c_opt(i),c_min,c_max,relative_width);
    sensitivity_results=[sensitivity_results;table(string(term_names{i}), ...
        c_opt(i),c_min,c_max,relative_width,R2_lower,R2_upper, ...
        'VariableNames',{'Term','Coefficient','Lower','Upper','ri_percent','R2_lower','R2_upper'})]; %#ok<AGROW>
end
fprintf('ri为区间半宽/系数绝对值；真实应力尺度的上下界通常不对称。\n');
fprintf('灵敏度结果保留在工作区 sensitivity_results 中，未生成文件。\n');

function [lower,upper]=coefficient_bounds(X,c,sigma_true,index,target)
if all(X(:,index)==0)
    lower=-Inf; upper=Inf; return;
end
base_Y=X*c;
directions=[-1,1]; bounds=zeros(1,2);
for d=1:2
    direction=directions(d);
    distance=0.01*max(abs(c(index)),1);
    bracketed=false;
    for attempt=1:100
        test_R2=corner_r2(sigma_true,exp(base_Y+direction*distance*X(:,index)));
        if ~isfinite(test_R2) || test_R2<target
            bracketed=true; break;
        end
        distance=distance*1.5;
    end
    if ~bracketed, error('系数%d未找到阈值边界。',index); end
    lo=0; hi=distance;
    for iteration=1:60
        mid=(lo+hi)/2;
        test_R2=corner_r2(sigma_true,exp(base_Y+direction*mid*X(:,index)));
        if isfinite(test_R2) && test_R2>=target, lo=mid; else, hi=mid; end
    end
    bounds(d)=c(index)+direction*lo;
end
lower=bounds(1); upper=bounds(2);
end

function value = corner_r2(observed, predicted)
% 在传入的尺度上计算 R²；调用者明确传 Y 或 sigma。
observed = observed(:); predicted = predicted(:);
if numel(observed) ~= numel(predicted)
    error('真实值和预测值的样本数不一致。');
end
if numel(observed) < 2 || any(~isfinite(observed))
    value = NaN;
    return;
end
SS_tot = sum((observed-mean(observed)).^2);
if SS_tot == 0
    value = NaN;
elseif any(~isfinite(predicted))
    value = -Inf;
else
    value = 1-sum((observed-predicted).^2)/SS_tot;
end
end
