clear; clc;
script_dir = fileparts(mfilename('fullpath'));

filename = fullfile(script_dir, 'center_training_data.xlsx'); 
if ~exist(filename, 'file')
    error('⚠️ 找不到文件: %s，请检查文件路径！', filename);
end

opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve';
data = readtable(filename, opts);

%% ========================================================================
%% 🛠️ 修正后的数据预处理模块（先清洗物理符号，再取绝对值）
%% ========================================================================
hg_raw      = data{:, 4};      
H_raw       = data{:, 5};      
a_raw       = data{:, 6};  
w_raw_orig  = data{:, 8};  
sigma_c_raw = data{:, 9}; 

% 📌 2. 严格物理有效性清洗：形变 w 必须小于 0 
valid_raw_mask = (H_raw > 0) & (a_raw > 0) & (hg_raw > 0) & ...
                 (w_raw_orig < 0) & (sigma_c_raw > 0);

% 📌 3. 过滤非法数据后，再提取合法数据并取绝对值
hg_raw      = hg_raw(valid_raw_mask);
H_raw       = H_raw(valid_raw_mask);
a_raw       = a_raw(valid_raw_mask);
w_raw       = abs(w_raw_orig(valid_raw_mask)); % 仅对物理合法的下挠数据取绝对值
sigma_c_raw = sigma_c_raw(valid_raw_mask);

rho_g = 2500; g = 9.8; 
H_unique = unique(H_raw);

thin_records   = []; % 薄板段   (H <= 0.016m)
trans1_records = []; % 过渡段1 (0.016m <= H <= 0.020m) [共享 0.016, 0.020]
thick1_records = []; % 中厚板段 (0.020m <= H <= 0.024m) [共享 0.020, 0.024]
thick2_records = []; % 过渡段2 (0.024m <= H <= 0.028m) [共享 0.024, 0.028]
thick3_records = []; % 超厚板段 (H >= 0.028m)         [共享 0.028]

segment_names = {'薄板段 (<=0.016m)', ...
                 '第一过渡段 (0.016-0.020m)', ...
                 '中厚板段 (0.020-0.024m)', ...
                 '第二过渡段 (0.024-0.028m)', ...
                 '超厚板段 (>=0.028m)'};

%% ========================================================================
%% 🛠️ 边界数据重叠分发与清洗
%% ========================================================================
for i = 1:length(H_unique)
    curr_H = H_unique(i);
    idx = (H_raw == curr_H);
    if sum(idx) == 0, continue; end
    
    tbl_sub = table(a_raw(idx), hg_raw(idx), w_raw(idx), sigma_c_raw(idx), 'VariableNames', {'a', 'hg', 'w', 'sigma_c'});
    [G, a_med, hg_med] = findgroups(tbl_sub.a, tbl_sub.hg);
    w_med = splitapply(@median, tbl_sub.w, G);
    sigma_c_med = splitapply(@median, tbl_sub.sigma_c, G);
    
    X1 = log(curr_H ./ a_med);
    X2 = log(a_med ./ hg_med);
    Z  = log(w_med ./ curr_H);
    Y  = log(sigma_c_med ./ (rho_g * g .* hg_med));
    
    thresh_factor = 3;
    
    outlier_Z = isoutlier(Z, 'median', 'ThresholdFactor', thresh_factor);
    outlier_Y = isoutlier(Y, 'median', 'ThresholdFactor', thresh_factor);
    valid_mask = ~(outlier_Z | outlier_Y);

    
    X1 = X1(valid_mask); X2 = X2(valid_mask); Z = Z(valid_mask); Y = Y(valid_mask);
    batch_data = [repmat(curr_H, size(X1)), X1, X2, Z, Y];
    
    % 📌 1. 薄板段: H <= 0.016m
    if curr_H <= 0.016 + 1e-4
        thin_records = [thin_records; batch_data];
    end
    
    % 📌 2. 第一过渡段: 0.016m <= H <= 0.020m
    if curr_H >= 0.016 - 1e-4 && curr_H <= 0.020 + 1e-4
        trans1_records = [trans1_records; batch_data];
    end
    
    % 📌 3. 中厚板段: 0.020m <= H <= 0.024m
    if curr_H >= 0.020 - 1e-4 && curr_H <= 0.024 + 1e-4
        thick1_records = [thick1_records; batch_data];
    end
    
    % 📌 4. 第二过渡段: 0.024m <= H <= 0.028m
    if curr_H >= 0.024 - 1e-4 && curr_H <= 0.028 + 1e-4
        thick2_records = [thick2_records; batch_data];
    end
    
    % 📌 5. 超厚板段: H >= 0.028m
    if curr_H >= 0.028 - 1e-4
        thick3_records = [thick3_records; batch_data];
    end
end

%% ========================================================================
%% 1. 薄板区间 (H <= 0.016m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔹 1. 薄板段 (H <= 0.016m) [基于对数空间 R^2(Y) 剪枝] \n');
fprintf('=======================================================================================\n');

T_thin_feats = build_feature_table(thin_records);
[mdl_thin_final, ~] = perform_r2_pruning_soft(T_thin_feats, 0.005, 4, 0);

fprintf('\n📌 【薄板段最终公式】:\n');
print_pretty_formula(mdl_thin_final);

%% ========================================================================
%% 2. 第一过渡段 (0.016m <= H <= 0.020m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔸 2. 第一过渡段 (0.016m <= H <= 0.020m) [基于对数空间 R^2(Y) 剪枝] \n');
fprintf('=======================================================================================\n');

T_trans1_feats = build_feature_table(trans1_records);
[mdl_trans1_final, ~] = perform_r2_pruning_soft(T_trans1_feats, 0.005, 5, 0);

fprintf('\n📌 【第一过渡段最终公式】:\n');
print_pretty_formula(mdl_trans1_final);

%% ========================================================================
%% 3. 中厚板段 (0.020m <= H <= 0.024m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔺 3. 中厚板段 (0.020m <= H <= 0.024m) [允许平方项参与筛选] \n');
fprintf('=======================================================================================\n');

T_thick1_feats = build_feature_table(thick1_records);
% 允许 X1_sq、X2_sq 参与本段筛选，不再预先排除。 

[mdl_thick1_final, ~] = perform_r2_pruning_soft(T_thick1_feats, 0.005, 4, 0.1);

fprintf('\n📌 【中厚板段最终公式】:\n');
print_pretty_formula(mdl_thick1_final);

%% ========================================================================
%% 4. 第二过渡段 (0.024m <= H <= 0.028m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔺 4. 第二过渡段 (0.024m <= H <= 0.028m) [允许平方项参与筛选] \n');
fprintf('=======================================================================================\n');

T_thick2_feats = build_feature_table(thick2_records);
% 允许 X1_sq、X2_sq 参与本段筛选，不再预先排除。 

[mdl_thick2_final, ~] = perform_r2_pruning_soft(T_thick2_feats, 0.005, 4, 0.15);

fprintf('\n📌 【第二过渡段最终公式】:\n');
print_pretty_formula(mdl_thick2_final);

%% ========================================================================
%% 5. 超厚板段 (H >= 0.028m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔺 5. 超厚板段 (H >= 0.028m) [基于对数空间 R^2(Y) 剪枝] \n');
fprintf('=======================================================================================\n');

T_thick3_feats = build_feature_table(thick3_records);
[mdl_thick3_final, ~] = perform_r2_pruning_soft(T_thick3_feats, 0.005, 4, 0.15);

fprintf('\n📌 【超厚板段最终公式】:\n');
print_pretty_formula(mdl_thick3_final);

%% ========================================================================
%% 6. 各厚度 H 下 MAPE 验证与各分段/数据集总体 R^2 汇总
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('                        各厚度 H 下 MAPE 验证汇总表                                 \n');
fprintf('=======================================================================================\n');
fprintf('   H (m)   |  MAPE (真实应力) |     评估路由分段\n');
fprintf('---------------------------------------------------------------------------------------\n');

seg_Y_true   = cell(1, 5);
seg_Y_pred   = cell(1, 5);
seg_sig_true = cell(1, 5);
seg_sig_pred = cell(1, 5);
seg_tables   = cell(1, 5); % 📌 用于保存每个分段互斥评估路由的数据特征表

all_Y_true   = [];
all_Y_pred   = [];
all_sig_true = [];
all_sig_pred = [];

for i = 1:length(H_unique)
    curr_H = H_unique(i);
    
    if curr_H <= 0.016 + 1e-4
        seg_idx = 1; mdl = mdl_thin_final; records = thin_records;
    elseif curr_H <= 0.020 + 1e-4
        seg_idx = 2; mdl = mdl_trans1_final; records = trans1_records;
    elseif curr_H <= 0.024 + 1e-4
        seg_idx = 3; mdl = mdl_thick1_final; records = thick1_records;
    elseif curr_H < 0.028 - 1e-4  % 📌 完全按 zhao，0.028m 归入第 5 段
        seg_idx = 4; mdl = mdl_thick2_final; records = thick2_records;
    else                          % 📌 0.028m 归入第 5 段
        seg_idx = 5; mdl = mdl_thick3_final; records = thick3_records;
    end
    seg_name = segment_names{seg_idx};
    
    sub_mask = (records(:, 1) == curr_H);
    if sum(sub_mask) < 3
        fprintf('  %-8.3f |     -----     | 数据不足无法验证\n', curr_H);
        continue; 
    end
    
    sub_rec = records(sub_mask, :);
    tbl_eval = build_feature_table(sub_rec);
    Y_true = tbl_eval.Y;
    Y_pred = predict_custom(mdl, tbl_eval);
    
    sigma_true = tbl_eval.sigma_c_true;
    sigma_pred = exp(Y_pred) .* (rho_g * g .* tbl_eval.hg);
    
    mape = mean(abs(sigma_true - sigma_pred) ./ sigma_true) * 100;
    
    fprintf('  %-8.3f |  %9.2f%%   | %s\n', curr_H, mape, seg_name);

    seg_Y_true{seg_idx}   = [seg_Y_true{seg_idx}; Y_true(:)];
    seg_Y_pred{seg_idx}   = [seg_Y_pred{seg_idx}; Y_pred(:)];
    seg_sig_true{seg_idx} = [seg_sig_true{seg_idx}; sigma_true(:)];
    seg_sig_pred{seg_idx} = [seg_sig_pred{seg_idx}; sigma_pred(:)];
    
    % 📌 收集当前分段非重叠路由下的完整特征表
    seg_tables{seg_idx}   = [seg_tables{seg_idx}; tbl_eval];

    all_Y_true   = [all_Y_true; Y_true(:)];
    all_Y_pred   = [all_Y_pred; Y_pred(:)];
    all_sig_true = [all_sig_true; sigma_true(:)];
    all_sig_pred = [all_sig_pred; sigma_pred(:)];
end
fprintf('---------------------------------------------------------------------------------------\n');

%% ========================================================================
%% 📊 各分段总体 R^2 与 MAPE 汇总
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('                           各分段总体 R^2 与 MAPE 汇总                             \n');
fprintf('=======================================================================================\n');
for s_idx = 1:5
    if isempty(seg_Y_true{s_idx})
        continue;
    end
    SStot_Y = sum((seg_Y_true{s_idx} - mean(seg_Y_true{s_idx})).^2);
    SSres_Y = sum((seg_Y_true{s_idx} - seg_Y_pred{s_idx}).^2);
    R2_Y    = 1 - (SSres_Y / SStot_Y);

    SStot_sig = sum((seg_sig_true{s_idx} - mean(seg_sig_true{s_idx})).^2);
    SSres_sig = sum((seg_sig_true{s_idx} - seg_sig_pred{s_idx}).^2);
    R2_sig    = 1 - (SSres_sig / SStot_sig);

    seg_MAPE  = mean(abs(seg_sig_true{s_idx} - seg_sig_pred{s_idx}) ./ seg_sig_true{s_idx}) * 100;

    fprintf('  - %-22s: N=%-4d | R^2(Y)=%.4f | R^2(sigma)=%.4f | MAPE(sigma)=%.2f%%\n', ...
        segment_names{s_idx}, length(seg_Y_true{s_idx}), R2_Y, R2_sig, seg_MAPE);
end

if ~isempty(all_Y_true)
    overall_SStot_Y   = sum((all_Y_true - mean(all_Y_true)).^2);
    overall_SSres_Y   = sum((all_Y_true - all_Y_pred).^2);
    overall_R2_Y      = 1 - (overall_SSres_Y / overall_SStot_Y);

    overall_SStot_sig = sum((all_sig_true - mean(all_sig_true)).^2);
    overall_SSres_sig = sum((all_sig_true - all_sig_pred).^2);
    overall_R2_sig    = 1 - (overall_SSres_sig / overall_SStot_sig);

    overall_MAPE_sig  = mean(abs(all_sig_true - all_sig_pred) ./ all_sig_true) * 100;

    fprintf('---------------------------------------------------------------------------------------\n');
    fprintf('📊 【数据集全局评估结果】:\n');
    fprintf('   - 总体样本量 N           : %d\n', length(all_Y_true));
    fprintf('   - 对数空间总体 R^2 (Y)    : %.4f\n', overall_R2_Y);
    fprintf('   - 真实应力总体 R^2 (sigma): %.4f\n', overall_R2_sig);
    fprintf('   - 真实应力总体 MAPE       : %.2f%%\n', overall_MAPE_sig);
    fprintf('=======================================================================================\n\n');
end

%% ========================================================================
%% 【中心点五段式拟合模型——基于真实应力的单参数灵敏度评估】
%% 说明：基于真实应力 R^2(sigma) >= 0.70 约束，采用二分法计算参数浮动边界
%% ========================================================================

target_R2 = 0.70;
models = {mdl_thin_final, mdl_trans1_final, mdl_thick1_final, mdl_thick2_final, mdl_thick3_final};

% 📌 替换为第 6 节互斥评估特征表 seg_tables，保证 N 彻底对齐
feature_tables = seg_tables; 

fprintf('\n===============================================================================\n');
fprintf('     📊 中心点模型五大分段参数灵敏度评估 (标准: 真实应力 R^2(sigma) >= %.2f)\n', target_R2);
fprintf('===============================================================================\n');

for s = 1:length(models)
    mdl = models{s};
    tbl = feature_tables{s};
    seg_name = segment_names{s};
    
    fprintf('\n-------------------------------------------------------------------------------\n');
    fprintf('📌 分段 [%d/5]: %s\n', s, seg_name);
    fprintf('-------------------------------------------------------------------------------\n');
    
    if isempty(tbl)
        fprintf('⚠️ 当前分段无评估样本，跳过灵敏度测试。\n');
        continue;
    end
    
    term_names = mdl.CoefficientNames;
    c_opt = mdl.Coefficients.Estimate;
    N = height(tbl);
    
    [base_R2_sig, base_MAPE, ~] = calc_sigma_metrics(mdl, tbl);
    
    fprintf('基准真实应力 R^2(sigma): %.4f | MAPE: %.2f%% | 样本量 N: %d\n', base_R2_sig, base_MAPE, N);
    
    if base_R2_sig < target_R2
        fprintf('⚠️ 警告: 该分段基准真实应力 R^2 (%.4f) 已低于目标值 %.2f，跳过灵敏度测试。\n', base_R2_sig, target_R2);
        continue;
    end
    
    fprintf('%-18s | %-10s | %-10s | %-10s | %-10s\n', '参数/特征项', '最优系数', '下限边界', '上限边界', '允许浮动比例');
    fprintf('-------------------------------------------------------------------------------\n');
    
    for i = 1:length(c_opt)
        c_val = c_opt(i);
        
        [c_min, c_max] = find_coeff_bounds_sigma(mdl, tbl, i, target_R2);
        
        delta_c = (c_max - c_min) / 2;
        if abs(c_val) > 1e-6
            fluct_ratio = (delta_c / abs(c_val)) * 100;
            fluct_str = sprintf('%5.1f%%', fluct_ratio);
        else
            fluct_str = 'N/A';
        end
        
        fprintf('%-20s | %-12.4f | %-12.4f | %-12.4f | %-10s\n', ...
            term_names{i}, c_val, c_min, c_max, fluct_str);
    end
end
fprintf('===============================================================================\n');

%% ========================================================================
%% 🛠️ 辅助函数模块
%% ========================================================================
function T_feats = build_feature_table(records)
    T_feats = table();
    T_feats.H      = records(:, 1);
    T_feats.X1     = records(:, 2);
    T_feats.X2     = records(:, 3);
    T_feats.Z      = records(:, 4);
    
    T_feats.X1_X2  = T_feats.X1 .* T_feats.X2;
    T_feats.Z_X1   = T_feats.Z  .* T_feats.X1;
    T_feats.Z_X2   = T_feats.Z  .* T_feats.X2;
    T_feats.X1_sq  = T_feats.X1.^2;
    T_feats.X2_sq  = T_feats.X2.^2;
    T_feats.Z_sq   = T_feats.Z.^2;
    
    T_feats.Y      = records(:, 5);
    
    rho_g = 2500; g = 9.8;
    T_feats.hg = T_feats.H ./ exp(T_feats.X1 + T_feats.X2);
    T_feats.sigma_c_true = exp(T_feats.Y) .* (rho_g * g .* T_feats.hg);
end

function [r2_sig, mape_sig, sigma_pred] = calc_sigma_metrics(mdl_struct, tbl)
    rho_g = 2500; g = 9.8;
    Y_pred = predict_custom(mdl_struct, tbl);
    
    sigma_pred = exp(Y_pred) .* (rho_g * g .* tbl.hg);
    sigma_true = tbl.sigma_c_true;
    
    SS_tot = sum((sigma_true - mean(sigma_true)).^2);
    SS_res = sum((sigma_true - sigma_pred).^2);
    
    if SS_tot < 1e-12
        r2_sig = NaN;
    else
        r2_sig = 1 - (SS_res / SS_tot);
    end
    mape_sig = mean(abs(sigma_true - sigma_pred) ./ sigma_true) * 100;
end

function mdl_struct = fit_soft_lm(data_table, terms, lambda_intercept)
    Y = data_table.Y;
    N = height(data_table);
    
    if isempty(terms)
        X = ones(N, 1);
        term_names = {'Intercept'};
    else
        X = [ones(N, 1), data_table{:, terms}];
        term_names = [{'Intercept'}, terms];
    end
    
    p = size(X, 2);
    if lambda_intercept > 0
        X_aug = [X; sqrt(lambda_intercept), zeros(1, p-1)];
        Y_aug = [Y; 0];
        beta = X_aug \ Y_aug;
    else
        beta = X \ Y;
    end
    
    Y_pred = X * beta;
    SS_res = sum((Y - Y_pred).^2);
    SS_tot = sum((Y - mean(Y)).^2);
    if SS_tot < 1e-12
        r2 = NaN;
    else
        r2 = 1 - (SS_res / SS_tot);
    end
    
    mdl_struct = struct();
    mdl_struct.Coefficients.Estimate = beta;
    mdl_struct.CoefficientNames = term_names;
    mdl_struct.Rsquared.Ordinary = r2;
end

function [current_mdl, current_terms] = perform_r2_pruning_soft(data_table, threshold, max_terms, lambda_intercept)
    if nargin < 3, max_terms = Inf; end
    if nargin < 4, lambda_intercept = 0; end

    all_features = data_table.Properties.VariableNames;
    ignore_vars = {'H', 'hg', 'Y', 'sigma_c_true'};
    all_features = setdiff(all_features, ignore_vars, 'stable');
    current_terms = all_features;
    
    current_mdl = fit_soft_lm(data_table, current_terms, lambda_intercept);
    current_r2 = current_mdl.Rsquared.Ordinary; % 基于对数空间 R^2(Y)
    
    while length(current_terms) > 1
        best_remove_idx = -1;
        best_r2_drop = Inf;
        best_test_mdl = [];
        best_test_r2 = -Inf;
        
        for k = 1:length(current_terms)
            test_terms = current_terms;
            test_terms(k) = [];
            
            test_mdl = fit_soft_lm(data_table, test_terms, lambda_intercept);
            test_r2 = test_mdl.Rsquared.Ordinary; % 基于对数空间 R^2(Y)
            r2_drop = current_r2 - test_r2;
            
            if r2_drop < best_r2_drop
                best_r2_drop = r2_drop;
                best_remove_idx = k;
                best_test_mdl = test_mdl;
                best_test_r2 = test_r2;
            end
        end
        
        if best_remove_idx ~= -1
            if (best_r2_drop < threshold) || (length(current_terms) > max_terms)
                removed_term = current_terms{best_remove_idx};
                current_terms(best_remove_idx) = [];
                current_mdl = best_test_mdl;
                current_r2 = best_test_r2;
                
                if best_r2_drop >= threshold
                    fprintf('✂️ 强制删项(超限制项数): %-10s | R^2(Y) 降幅: %.6f | 当前 R^2(Y): %.6f | 截距: %.4f\n', ...
                            removed_term, best_r2_drop, current_r2, current_mdl.Coefficients.Estimate(1));
                else
                    fprintf('✂️ 常规删项: %-10s | R^2(Y) 降幅: %.6f | 当前 R^2(Y): %.6f | 截距: %.4f\n', ...
                            removed_term, best_r2_drop, current_r2, current_mdl.Coefficients.Estimate(1));
                end
            else
                break;
            end
        else
            break;
        end
    end
end

function [c_min, c_max] = find_coeff_bounds_sigma(mdl_struct, tbl, param_idx, target_R2)
    rho_g = 2500; g = 9.8;
    c_opt = mdl_struct.Coefficients.Estimate;
    term_names = mdl_struct.CoefficientNames;
    sigma_true = tbl.sigma_c_true;
    SS_tot = sum((sigma_true - mean(sigma_true)).^2);
    
    N = height(tbl);
    X_mat = ones(N, 1);
    for k = 2:length(term_names)
        X_mat = [X_mat, tbl{:, term_names{k}}];
    end
    
    function r2 = calc_r2_with_c(c_vec)
        Y_p = X_mat * c_vec;
        sig_p = exp(Y_p) .* (rho_g * g .* tbl.hg);
        SS_res = sum((sigma_true - sig_p).^2);
        r2 = 1 - (SS_res / SS_tot);
    end

    % 1. 求解上限 c_max
    step = 0.05;
    delta = step;
    c_test = c_opt;
    c_test(param_idx) = c_opt(param_idx) + delta;
    while calc_r2_with_c(c_test) > target_R2 && delta < 1e5
        delta = delta * 1.5;
        c_test(param_idx) = c_opt(param_idx) + delta;
    end
    
    d_low = 0; d_high = delta;
    for iter = 1:30
        d_mid = (d_low + d_high) / 2;
        c_test(param_idx) = c_opt(param_idx) + d_mid;
        if calc_r2_with_c(c_test) >= target_R2
            d_low = d_mid;
        else
            d_high = d_mid;
        end
    end
    c_max = c_opt(param_idx) + d_low;
    
    % 2. 求解下限 c_min
    delta = step;
    c_test = c_opt;
    c_test(param_idx) = c_opt(param_idx) - delta;
    while calc_r2_with_c(c_test) > target_R2 && delta < 1e5
        delta = delta * 1.5;
        c_test(param_idx) = c_opt(param_idx) - delta;
    end
    
    d_low = 0; d_high = delta;
    for iter = 1:30
        d_mid = (d_low + d_high) / 2;
        c_test(param_idx) = c_opt(param_idx) - d_mid;
        if calc_r2_with_c(c_test) >= target_R2
            d_low = d_mid;
        else
            d_high = d_mid;
        end
    end
    c_min = c_opt(param_idx) - d_low;
end

function Y_pred = predict_custom(mdl, tbl)
    coefs = mdl.Coefficients.Estimate;
    names = mdl.CoefficientNames;
    Y_pred = coefs(1) * ones(height(tbl), 1);
    for i = 2:length(names)
        term_name = names{i};
        Y_pred = Y_pred + coefs(i) * tbl{:, term_name};
    end
end

function print_pretty_formula(mdl)
    coefs = mdl.Coefficients.Estimate;
    names = mdl.CoefficientNames;
    
    formula_str = sprintf('Y = %.4f', coefs(1));
    for i = 2:length(names)
        val = coefs(i);
        term_name = names{i};
        term_name = strrep(term_name, '_sq', '^2');
        term_name = strrep(term_name, '_X1', '*X1');
        term_name = strrep(term_name, '_X2', '*X2');
        
        if val >= 0
            formula_str = sprintf('%s + %.4f * (%s)', formula_str, val, term_name);
        else
            formula_str = sprintf('%s - %.4f * (%s)', formula_str, abs(val), term_name);
        end
    end
    disp(formula_str);
end