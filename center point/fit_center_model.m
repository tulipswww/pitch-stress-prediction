
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
%% 🛠️ 修正后的数据预处理模块
%% ========================================================================
hg_raw      = data{:, 4};      
H_raw       = data{:, 5};      
a_raw       = data{:, 6};  
w_raw_orig  = data{:, 8};  
sigma_c_raw = data{:, 9}; 

% 📌 严格物理有效性清洗：形变 w 必须小于 0 
valid_raw_mask = (H_raw > 0) & (a_raw > 0) & (hg_raw > 0) & ...
                 (w_raw_orig < 0) & (sigma_c_raw > 0);

% 📌 过滤非法数据后，再提取合法数据并取绝对值
hg_raw      = hg_raw(valid_raw_mask);
H_raw       = H_raw(valid_raw_mask);
a_raw       = a_raw(valid_raw_mask);
w_raw       = abs(w_raw_orig(valid_raw_mask)); % 仅对物理合法的下挠数据取绝对值
sigma_c_raw = sigma_c_raw(valid_raw_mask);

rho_g = 2500; g = 9.8; 
H_unique = unique(H_raw);

thin_records   = []; % 薄板段   (H <= 0.016m)
trans1_records = []; % 过渡段1 (0.016m <= H <= 0.020m)
thick1_records = []; % 中厚板段 (0.020m <= H <= 0.024m)
thick2_records = []; % 过渡段2 (0.024m <= H <= 0.028m)
thick3_records = []; % 超厚板段 (H >= 0.028m)



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
    
    % 📌 2. 第一过渡段: 0.016m < H <= 0.020m
    if curr_H >= 0.016 - 1e-4 && curr_H <= 0.020 + 1e-4
        trans1_records = [trans1_records; batch_data];
    end
    
    % 📌 3. 中厚板段: 0.020m < H <= 0.024m
    if curr_H >= 0.020 - 1e-4 && curr_H <= 0.024 + 1e-4
        thick1_records = [thick1_records; batch_data];
    end
    
    % 📌 4. 第二过渡段: 0.024m < H < 0.028m
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
fprintf('        🔹 1. 薄板段 (H <= 0.016m) [训练包含: H <= 0.016m] \n');
fprintf('=======================================================================================\n');

if isempty(thin_records)
    fprintf('⚠️ 薄板段无有效数据，跳过拟合。\n');
    mdl_thin_final = [];
else
    T_thin_feats = build_feature_table(thin_records);
    [mdl_thin_final, ~] = perform_r2_pruning_soft(T_thin_feats, 0.005, 4, 0);
    
    fprintf('\n📌 【薄板段最终公式】:\n');
    print_pretty_formula(mdl_thin_final);
end

%% ========================================================================
%% 2. 第一过渡段 (0.016m <= H <= 0.020m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔸 2. 第一过渡段 (0.016m <= H <= 0.020m) [训练包含: 0.016, 0.018, 0.020m] \n');
fprintf('=======================================================================================\n');

if isempty(trans1_records)
    fprintf('⚠️ 第一过渡段无有效数据，跳过拟合。\n');
    mdl_trans1_final = [];
else
    T_trans1_feats = build_feature_table(trans1_records);
    [mdl_trans1_final, ~] = perform_r2_pruning_soft(T_trans1_feats, 0.005, 5, 0);
    fprintf('\n📌 【第一过渡段最终公式】:\n');
    print_pretty_formula(mdl_trans1_final);
end

%% ========================================================================
%% 3. 中厚板段 (0.020m <= H <= 0.024m) 拟合 - 完整二次候选集合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔺 3. 中厚板段 (0.020m <= H <= 0.024m) [允许平方项参与筛选] \n');
fprintf('=======================================================================================\n');

if isempty(thick1_records)
    fprintf('⚠️ 中厚板段无有效数据，跳过拟合。\n');
    mdl_thick1_final = [];
else
    T_thick1_feats = build_feature_table(thick1_records);
    % 允许 X1_sq、X2_sq 参与本段筛选，不再预先排除。 
    [mdl_thick1_final, ~] = perform_r2_pruning_soft(T_thick1_feats, 0.005, 4, 0.1);
    fprintf('\n📌 【中厚板段最终公式】:\n');
    print_pretty_formula(mdl_thick1_final);
end

%% ========================================================================
%% 4. 第二过渡段 (0.024m <= H <= 0.028m) 拟合 - 完整二次候选集合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔺 4. 第二过渡段 (0.024m <= H <= 0.028m) [允许平方项参与筛选] \n');
fprintf('=======================================================================================\n');

if isempty(thick2_records)
    fprintf('⚠️ 第二过渡段无有效数据，跳过拟合。\n');
    mdl_thick2_final = [];
else
    T_thick2_feats = build_feature_table(thick2_records);
    % 允许 X1_sq、X2_sq 参与本段筛选，不再预先排除。 
    [mdl_thick2_final, ~] = perform_r2_pruning_soft(T_thick2_feats, 0.005, 4, 0.15);
    fprintf('\n📌 【第二过渡段最终公式】:\n');
    print_pretty_formula(mdl_thick2_final);
end

%% ========================================================================
%% 5. 超厚板段 (H >= 0.028m) 拟合
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('        🔺 5. 超厚板段 (H >= 0.028m) [训练包含: H >= 0.028m] \n');
fprintf('=======================================================================================\n');

if isempty(thick3_records)
    fprintf('⚠️ 超厚板段无有效数据，跳过拟合。\n');
    mdl_thick3_final = [];
else
    T_thick3_feats = build_feature_table(thick3_records);
    [mdl_thick3_final, ~] = perform_r2_pruning_soft(T_thick3_feats, 0.005, 4, 0.15);
    fprintf('\n📌 【超厚板段最终公式】:\n');
    print_pretty_formula(mdl_thick3_final);
end

%% ========================================================================
%% 各厚度 H 下 R^2 汇总（Y空间与sigma空间）
%% ========================================================================
fprintf('\n=======================================================================================\n');
fprintf('                        各厚度 H 下 R^2 汇总表 (Y空间与sigma空间)                \n');
fprintf('=======================================================================================\n');
fprintf('   H (m)   |  R^2 (Y)   | R^2 (sigma)\n');
fprintf('---------------------------------------------\n');

for i = 1:length(H_unique)
    curr_H = H_unique(i);
    % 路由
    if curr_H <= 0.016 + 1e-4
        mdl = mdl_thin_final; records = thin_records;
    elseif curr_H <= 0.020 + 1e-4
        mdl = mdl_trans1_final; records = trans1_records;
    elseif curr_H <= 0.024 + 1e-4
        mdl = mdl_thick1_final; records = thick1_records;
    elseif curr_H < 0.028 - 1e-4
        mdl = mdl_thick2_final; records = thick2_records;
    else
        mdl = mdl_thick3_final; records = thick3_records;
    end
    
    sub_mask = (records(:, 1) == curr_H);
    if sum(sub_mask) < 3 || isempty(mdl)
        fprintf('  %-8.3f |    -----   |   -----\n', curr_H);
        continue;
    end
    
    sub_rec = records(sub_mask, :);
    tbl_eval = build_feature_table(sub_rec);
    Y_true = tbl_eval.Y;
    Y_pred = predict_custom(mdl, tbl_eval);
    
    % R²(Y)
    SStot_Y = sum((Y_true - mean(Y_true)).^2);
    SSres_Y = sum((Y_true - Y_pred).^2);
    if SStot_Y < 1e-12
        R2_Y = NaN;
    else
        R2_Y = 1 - SSres_Y / SStot_Y;
    end
    
    % R²(sigma)
    X1_sub = tbl_eval.X1;
    X2_sub = tbl_eval.X2;
    hg_sub = curr_H ./ exp(X1_sub + X2_sub);
    sigma_true = exp(Y_true) .* (rho_g * g .* hg_sub);
    sigma_pred = exp(Y_pred) .* (rho_g * g .* hg_sub);
    
    SStot_sig = sum((sigma_true - mean(sigma_true)).^2);
    SSres_sig = sum((sigma_true - sigma_pred).^2);
    if SStot_sig < 1e-12
        R2_sig = NaN;
    else
        R2_sig = 1 - SSres_sig / SStot_sig;
    end
    
    fprintf('  %-8.3f |  %8.4f  |  %8.4f\n', curr_H, R2_Y, R2_sig);
end
fprintf('---------------------------------------------\n');

%% 整理MAD筛选后的独立工况（仅保留在工作区）
% 合并五段已经完成MAD筛选的数据
export_records = [thin_records; trans1_records; thick1_records; ...
                  thick2_records; thick3_records];

if isempty(export_records)
    warning('MAD筛选后无数据。');
else
    % 每行依次为 [H, X1, X2, Z, Y]
    % 去除相邻分段共享造成的重复记录
    export_records = unique(export_records, 'rows', 'stable');

    export_H  = export_records(:, 1);
    export_X1 = export_records(:, 2);
    export_X2 = export_records(:, 3);
    export_Z  = export_records(:, 4);
    export_Y  = export_records(:, 5);

    % 还原物理量，长度单位均为m，应力单位为Pa
    export_a = export_H ./ exp(export_X1);
    export_hg = export_H ./ exp(export_X1 + export_X2);
    export_w = export_H .* exp(export_Z); % 向下位移幅值，为正数
    export_sigma = rho_g * g .* export_hg .* exp(export_Y);

    % 生成结果表
    MAD_clean_table = table( ...
        export_H, export_hg, export_a, export_w, export_sigma, ...
        'VariableNames', {'H_m', 'hg_m', 'a_m', 'w_m', 'sigma_Pa'});

    MAD_clean_table = sortrows( ...
        MAD_clean_table, {'H_m', 'hg_m', 'a_m'});

    % 保留筛选结果在工作区，不自动生成额外Excel文件。
    fprintf('\nMAD筛选后保留 %d 个独立工况，结果保留在 MAD_clean_table 中。\n', ...
        height(MAD_clean_table));
end
%% ========================================================================
%% 🛠️ 辅助函数模块
%% ========================================================================

function T_feats = build_feature_table(records)
    % 空输入保护
    if isempty(records)
        T_feats = table();
        return;
    end
    T_feats = table();
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
    all_features(strcmp(all_features, 'Y')) = [];
    current_terms = all_features;
    
    current_mdl = fit_soft_lm(data_table, current_terms, lambda_intercept);
    current_r2 = current_mdl.Rsquared.Ordinary;
    
    while length(current_terms) > 1
        best_remove_idx = -1;
        best_r2_drop = Inf;
        best_test_mdl = [];
        
        for k = 1:length(current_terms)
            test_terms = current_terms;
            test_terms(k) = [];
            
            test_mdl = fit_soft_lm(data_table, test_terms, lambda_intercept);
            r2_drop = current_r2 - test_mdl.Rsquared.Ordinary;
            
            if r2_drop < best_r2_drop
                best_r2_drop = r2_drop;
                best_remove_idx = k;
                best_test_mdl = test_mdl;
            end
        end
        
        if best_remove_idx ~= -1
            if (best_r2_drop < threshold) || (length(current_terms) > max_terms)
                removed_term = current_terms{best_remove_idx};
                current_terms(best_remove_idx) = [];
                current_mdl = best_test_mdl;
                current_r2 = current_mdl.Rsquared.Ordinary;
                
                if best_r2_drop >= threshold
                    fprintf('✂️ 强制删项(超限制项数): %-10s | R^2 降幅: %.6f | 当前 R^2: %.6f | 截距: %.4f\n', ...
                            removed_term, best_r2_drop, current_r2, current_mdl.Coefficients.Estimate(1));
                else
                    fprintf('✂️ 常规删项: %-10s | R^2 降幅: %.6f | 当前 R^2: %.6f | 截距: %.4f\n', ...
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