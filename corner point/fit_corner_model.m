clear; clc;

% 1. 加载数据与预处理
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
filename = fullfile(script_dir,'corner_training_data.xlsx');
[tbl_fit, cleaning_stats, preprocessing] = prepare_corner_data(filename,3.5);

%% ------------------------------------------------------------------------
%% 2. 🎛️ 设定 R^2 贡献度裁剪门槛
%% ------------------------------------------------------------------------
r2_threshold = 0.005; 

active_terms = {'Z', 'X1', 'X2', 'Z_sq', 'X1_sq', 'X2_sq', 'Z_X1', 'Z_X2', 'X1_X2'};

fprintf('\n=== 🔍 正在执行【R^2 贡献度硬阈值剪枝】 (设定门槛: ΔR^2 >= %.4f) ===\n', r2_threshold);

while length(active_terms) > 1
    full_formula = ['Y ~ ', strjoin(active_terms, ' + ')];
    mdl_curr = fitlm(tbl_fit, full_formula);
    base_r2  = mdl_curr.Rsquared.Ordinary;
    
    min_loss = Inf;
    worst_term_idx = -1;
    
    for i = 1:length(active_terms)
        temp_terms = active_terms;
        temp_terms(i) = [];
        sub_formula = ['Y ~ ', strjoin(temp_terms, ' + ')];
        mdl_sub = fitlm(tbl_fit, sub_formula);
        loss = base_r2 - mdl_sub.Rsquared.Ordinary;
        
        if loss < min_loss
            min_loss = loss;
            worst_term_idx = i;
        end
    end
    
    if min_loss < r2_threshold && length(active_terms) > 1
        fprintf('❌ 剔除项 [%-8s]: 剔除后 R^2 仅降低 %.6f (< %.4f 门槛)\n', ...
            active_terms{worst_term_idx}, min_loss, r2_threshold);
        active_terms(worst_term_idx) = [];
    else
        break;
    end
end

%% ------------------------------------------------------------------------
%% 3. 输出剪枝后的最终精简模型与系数
%% ------------------------------------------------------------------------
final_formula = ['Y ~ ', strjoin(active_terms, ' + ')];
mdl_final = fitlm(tbl_fit, final_formula);

fprintf('\n--------------------------------------------------------------------------------------\n');
fprintf('📊 最终保留的核心项及其真实的 R^2 贡献度：\n');
fprintf('--------------------------------------------------------------------------------------\n');
fprintf('%-15s | %-12s | %-15s | %-20s\n', '保留的核心项', '系数值', '剔除后的 R^2 损失', '保留原因');
fprintf('--------------------------------------------------------------------------------------\n');

c_names = mdl_final.CoefficientNames;
c_vals  = mdl_final.Coefficients.Estimate;
final_r2 = mdl_final.Rsquared.Ordinary;

for i = 1:length(c_names)
    term = c_names{i};
    coef = c_vals(i);
    
    if strcmp(term, '(Intercept)')
        fprintf('%-15s | %-12.4f | %-15s | %-20s\n', term, coef, 'N/A', '🛡️ 基础常数项');
    else
        temp_terms = active_terms;
        temp_terms(strcmp(temp_terms, term)) = [];
        if isempty(temp_terms)
            sub_formula = 'Y ~ 1';
        else
            sub_formula = ['Y ~ ', strjoin(temp_terms, ' + ')];
        end
        mdl_sub = fitlm(tbl_fit, sub_formula);
        loss = final_r2 - mdl_sub.Rsquared.Ordinary;
        
        fprintf('%-15s | %-12.4f | -%-14.6f | ✅ 贡献大 (>= %.4f)\n', term, coef, loss, r2_threshold);
    end
end
fprintf('--------------------------------------------------------------------------------------\n');

%% ------------------------------------------------------------------------
%% 4. 打印带真实系数的精简数学方程
%% ------------------------------------------------------------------------
fprintf('\n🏆 最终提取的【硬核精简数学公式】：\n');

str_eq = sprintf('Y = %.4f', c_vals(1)); 
for k = 2:length(c_vals)
    term_name_disp = strrep(c_names{k}, ':', ' * '); 
    if c_vals(k) >= 0
        str_eq = [str_eq, sprintf(' + %.4f * (%s)', c_vals(k), term_name_disp)];
    else
        str_eq = [str_eq, sprintf(' - %.4f * (%s)', abs(c_vals(k)), term_name_disp)];
    end
end

fprintf('%s\n', str_eq);
fprintf('精简后总项数: %d 项 (含常数项)\n', length(c_vals));
fprintf('精简后 R^2(Y): %.4f (全模型 10 项 R^2(Y): %.4f)\n', final_r2, fitlm(tbl_fit, 'Y ~ Z + X1 + X2 + Z_sq + X1_sq + X2_sq + Z_X1 + Z_X2 + X1_X2').Rsquared.Ordinary);

%% ------------------------------------------------------------------------
%% 5. 每个 H 及总体的对数和真实应力指标
fprintf('\n各厚度拟合指标（Y=ln(sigma)，sigma单位Pa）\n');
fprintf(' H(m)  |   N |   R2(Y)   | R2(sigma)\n');
H_unique_clean = unique(tbl_fit.H);
Y_pred_all = predict(mdl_final,tbl_fit);
sigma_true_all = exp(tbl_fit.Y);
sigma_pred_all = exp(Y_pred_all);
training_metrics = table();
for i=1:numel(H_unique_clean)
    curr_H = H_unique_clean(i);
    k = tbl_fit.H == curr_H;
    r2_log = corner_r2(tbl_fit.Y(k),Y_pred_all(k));
    r2_sig = corner_r2(sigma_true_all(k),sigma_pred_all(k));
    fprintf(' %.3f | %3d | %9.6f | %9.6f\n',curr_H,sum(k),r2_log,r2_sig);
    training_metrics = [training_metrics; table(curr_H,sum(k),r2_log,r2_sig, ...
        'VariableNames',{'H_m','N','R2_log','R2_sigma'})]; %#ok<AGROW>
end
r2_log_global = corner_r2(tbl_fit.Y,Y_pred_all);
r2_sigma_global = corner_r2(sigma_true_all,sigma_pred_all);
fprintf('总体 N=%d | R2(Y)=%.8f | R2(sigma)=%.8f\n', ...
    height(tbl_fit),r2_log_global,r2_sigma_global);

% 拟合结果仅保留在MATLAB工作区，供analyze_corner_sensitivity/validate_corner_model使用，不写入文件。
corner_model_cache=struct();
corner_model_cache.model_intercept=c_vals(1);
corner_model_cache.model_terms=c_names(2:end);
corner_model_cache.model_coefficients=c_vals(2:end);
corner_model_cache.training_table=tbl_fit;
corner_model_cache.cleaning_stats=cleaning_stats;
corner_model_cache.preprocessing=preprocessing;
corner_model_cache.training_metrics=training_metrics;
corner_model_cache.r2_log_global=r2_log_global;
corner_model_cache.r2_sigma_global=r2_sigma_global;
corner_model_cache.source_folder=script_dir;
data_stamp=dir(fullfile(script_dir,'corner_training_data.xlsx'));
script_stamp=dir(fullfile(script_dir,'fit_corner_model.m'));
corner_model_cache.source_data_stamp=[data_stamp.datenum,data_stamp.bytes];
corner_model_cache.source_script_stamp=[script_stamp.datenum,script_stamp.bytes];
fprintf('模型已保留在MATLAB工作区 corner_model_cache 中，未生成文件。\n');

function [tbl_fit, stats, preprocessing] = prepare_corner_data(filename, mad_threshold)
% a/b/c 共用的数据定义：先检查原始符号，再聚合，再按 H 做 scaled MAD。
if nargin < 2, mad_threshold = 3.5; end
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
data = readtable(filename, opts);
vars = data.Properties.VariableNames;
idx_H = find_column(vars, 'h_p', '沥青厚度', 5);
idx_a = find_column(vars, 'l_g', '玻璃边长', 6);
idx_hg = find_column(vars, 'h_g', '玻璃厚度', 4);
idx_w = find_column(vars, '纵向形变', '形变', 8);
idx_sig = find_column(vars, 'mises', '应力', 9);
H_raw = data{:,idx_H}; a_raw = data{:,idx_a}; hg_raw = data{:,idx_hg};
w_orig = data{:,idx_w}; sig_raw = data{:,idx_sig};
valid = isfinite(H_raw) & isfinite(a_raw) & isfinite(hg_raw) & ...
    isfinite(w_orig) & isfinite(sig_raw) & ...
    H_raw > 0 & a_raw > 0 & hg_raw > 0 & w_orig < 0 & sig_raw > 0;
stats.raw_count = height(data);
stats.physical_removed = sum(~valid);
stats.physical_retained = sum(valid);
H_raw = H_raw(valid); a_raw = a_raw(valid); hg_raw = hg_raw(valid);
w_raw = abs(w_orig(valid)); sig_raw = sig_raw(valid);
H_pool = []; Z_pool = []; X1_pool = []; X2_pool = []; Y_pool = [];
per_H = zeros(0,4);
for curr_H = unique(H_raw).'
    idx = H_raw == curr_H;
    [G, a_med, hg_med] = findgroups(a_raw(idx), hg_raw(idx));
    w_med = splitapply(@median, w_raw(idx), G);
    sigma_med = splitapply(@median, sig_raw(idx), G);
    Z = log(w_med ./ curr_H);
    X1 = log(curr_H ./ a_med);
    X2 = log(hg_med ./ curr_H);
    Y = log(sigma_med);
    outlier = isoutlier(Z,'median','ThresholdFactor',mad_threshold) | ...
        isoutlier(Y,'median','ThresholdFactor',mad_threshold);
    keep = ~outlier;
    per_H(end+1,:) = [curr_H, numel(Y), sum(outlier), sum(keep)]; %#ok<AGROW>
    H_pool = [H_pool; repmat(curr_H,sum(keep),1)]; %#ok<AGROW>
    Z_pool = [Z_pool; Z(keep)]; X1_pool = [X1_pool; X1(keep)]; %#ok<AGROW>
    X2_pool = [X2_pool; X2(keep)]; Y_pool = [Y_pool; Y(keep)]; %#ok<AGROW>
end
if isempty(Y_pool), error('清洗后无有效工况：%s',filename); end
tbl_fit = table(H_pool,Z_pool,X1_pool,X2_pool,Y_pool, ...
    'VariableNames',{'H','Z','X1','X2','Y'});
tbl_fit.Z_sq = tbl_fit.Z.^2;
tbl_fit.X1_sq = tbl_fit.X1.^2;
tbl_fit.X2_sq = tbl_fit.X2.^2;
tbl_fit.Z_X1 = tbl_fit.Z .* tbl_fit.X1;
tbl_fit.Z_X2 = tbl_fit.Z .* tbl_fit.X2;
tbl_fit.X1_X2 = tbl_fit.X1 .* tbl_fit.X2;
stats.per_H = array2table(per_H, ...
    'VariableNames',{'H_m','Aggregated','MAD_removed','Retained'});
stats.aggregated_count = sum(per_H(:,2));
stats.mad_removed = sum(per_H(:,3));
stats.retained_count = height(tbl_fit);
preprocessing = struct('mad_threshold',mad_threshold, ...
    'require_negative_w',true,'require_positive_geometry_and_stress',true, ...
    'require_finite',true,'aggregate','median_by_H_a_hg', ...
    'outlier_variables',{{'Z','Y'}},'outlier_group','H');
fprintf('基础筛选：原始 %d 条，排除 %d 条，保留 %d 条。\n', ...
    stats.raw_count,stats.physical_removed,stats.physical_retained);
fprintf('聚合后 %d 个工况，%.1f 倍 scaled MAD 排除 %d 个，最终保留 %d 个。\n', ...
    stats.aggregated_count,mad_threshold,stats.mad_removed,stats.retained_count);
end

function idx = find_column(vars, primary, secondary, fallback)
idx = find(contains(vars,primary,'IgnoreCase',true),1);
if isempty(idx), idx = find(contains(vars,secondary,'IgnoreCase',true),1); end
if isempty(idx)
    if numel(vars) < fallback, error('缺少数据列：%s',primary); end
    idx = fallback;
end
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
