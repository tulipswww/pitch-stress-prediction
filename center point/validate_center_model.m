clear; clc;
script_dir = fileparts(mfilename('fullpath'));

% 1. 📢 打印中心点五段式预测公式
fprintf('==================== 中心点五段式预测公式 ====================\n');
fprintf('📌 [薄板段 H <= 0.016m]:\n   Y = -0.6420 - 0.0143*(Z*X1) + 0.0270*(X1^2) + 0.0111*(X2^2) + 0.0010*(Z^2)\n');
fprintf('📌 [第一过渡段 0.016m <= H <= 0.020m]:\n   Y = -1.6623 - 0.3647*X1 - 0.0435*X2 - 0.0538*Z - 0.0194*(X1*X2) - 0.1311*(X1^2)\n');
fprintf('📌 [中厚板段 0.020m <= H <= 0.024m]:\n   Y = -0.0562 + 1.7766*(X1) + 1.7546*(X2) - 2.1121*(Z) - 0.1471*(Z^2)\n');
fprintf('📌 [第二过渡段 0.024m <= H < 0.028m]:\n   Y = -0.1098 + 1.9142*(X1) + 1.9129*(X2) - 2.6424*(Z) - 0.1865*(Z^2)\n');
fprintf('📌 [超厚板段 H >= 0.028m]:\n   Y = -0.0838 + 2.7391*X1 + 2.5143*X2 - 3.4848*Z - 0.2459*(Z^2)\n');
fprintf('=============================================================\n\n');

% 2. 📁 文件列表与对应的验证标题 (中心点文件)
file_list = {'center_validation_S1.xlsx', ...
             'center_validation_S2.xlsx', ...
             'center_validation_S3.xlsx'};
             
title_list = {'沥青厚度 hp 内插数据集验证结果', ...
              '玻璃厚度 hg 外推数据集验证结果', ...
              '内插与外推组合数据集验证结果'};

rho_g = 2500; 
g = 9.8;

segment_names = {'薄板段 (<=0.016m)', ...
                 '第一过渡段 (0.016-0.020m)', ...
                 '中厚板段 (0.020-0.024m)', ...
                 '第二过渡段 (0.024-0.028m)', ...
                 '超厚板段 (>=0.028m)'};

validation_summary = table(); % 仅在工作区汇总，不生成文件
% 遍历每个文件进行测试
for f_idx = 1:length(file_list)
    filename = fullfile(script_dir, file_list{f_idx});
    
    fprintf('================ %d. %s ================\n', f_idx, title_list{f_idx});
    fprintf('%-10s %-14s %s\n', 'H(m)', '    MAPE(应力)', '    所属评估路由分段');
    fprintf('--------------------------------------------------------------------------------\n');
    
    if ~exist(filename, 'file')
        fprintf('⚠️ 找不到文件: %s\n\n', filename);
        continue;
    end
    
    % 读取数据
    opts = detectImportOptions(filename);
    opts.VariableNamesRange = '1:1'; % 第 1 行作为列名范围
    opts.DataRange = 'A2';           % 从 A2 单元格起读取数据
    opts.VariableNamingRule = 'preserve';
    data = readtable(filename, opts);
    
    % 关键词模糊匹配
    vars = data.Properties.VariableNames;
    idx_H   = find(contains(vars, 'h_p', 'IgnoreCase', true) | contains(vars, '沥青', 'IgnoreCase', true));
    idx_a   = find(contains(vars, 'l_g', 'IgnoreCase', true) | contains(vars, '边长', 'IgnoreCase', true));
    idx_hg  = find(contains(vars, 'h_g', 'IgnoreCase', true) | contains(vars, '玻璃', 'IgnoreCase', true));
    idx_w   = find(contains(vars, 'w', 'IgnoreCase', true)   | contains(vars, '形变', 'IgnoreCase', true));
    idx_sig = find(contains(vars, 'mises', 'IgnoreCase', true)| contains(vars, '应力', 'IgnoreCase', true));
    
    % 兜底机制：若匹配失败，自动使用固定列号 (4:hg, 5:hp, 6:lg, 8:w, 9:sig)
    if isempty(idx_hg),  idx_hg  = 4; else, idx_hg  = idx_hg(1);  end
    if isempty(idx_H),   idx_H   = 5; else, idx_H   = idx_H(1);   end
    if isempty(idx_a),   idx_a   = 6; else, idx_a   = idx_a(1);   end
    if isempty(idx_w),   idx_w   = 8; else, idx_w   = idx_w(1);   end
    if isempty(idx_sig), idx_sig = 9; else, idx_sig = idx_sig(1); end
    
    % 提取真实数据
    H_raw       = data{:, idx_H}; 
    a_raw       = data{:, idx_a}; 
    hg_raw      = data{:, idx_hg};
    w_raw_orig  = data{:, idx_w};  
    sig_raw     = data{:, idx_sig};
    
    H_unique = unique(H_raw);
    
    % 📌 用于存储各个分段的汇总数据容器 (1~5段)
    seg_Y_true   = cell(1, 5);
    seg_sig_true = cell(1, 5);
    seg_sig_pred = cell(1, 5);
    seg_group_mape = cell(1, 5);
    group_metrics = table();
    
    % 📌 用于汇总整个数据集的预测与真实值容器
    all_Y_true   = [];
    all_sig_true = [];
    all_sig_pred = [];
    
    % 3. 针对每个不同的厚度进行单独处理与评估
    for h_idx = 1:length(H_unique)
        curr_H = H_unique(h_idx);
        
        % 🔍 【中心点基础过滤条件：物理符号筛选 w < 0】
        idx = (H_raw == curr_H) & (H_raw > 0) & (a_raw > 0) & (hg_raw > 0) & ...
              (w_raw_orig < 0) & (sig_raw > 0); 
        
        if sum(idx) == 0, continue; end
        
        % 校验通过后，取形变绝对值
        w_valid = abs(w_raw_orig(idx));
        
        % 按 (a, hg) 聚合并取中位数
        tbl_sub = table(a_raw(idx), hg_raw(idx), w_valid, sig_raw(idx), ...
            'VariableNames', {'a', 'hg', 'w', 'sigma_c'});
        [G, a_med, hg_med] = findgroups(tbl_sub.a, tbl_sub.hg);
        w_med = splitapply(@median, tbl_sub.w, G); 
        sigma_c_med = splitapply(@median, tbl_sub.sigma_c, G);
        
        % 计算中心点特征自变量与因变量
        X1 = log(curr_H ./ a_med); 
        X2 = log(a_med ./ hg_med); 
        Z  = log(w_med ./ curr_H); 
        Y_true = log(sigma_c_med ./ (rho_g * g .* hg_med));
        
        % 📌 对齐 z2 的 MAD 清洗阈值 factor = 3
        thresh_factor = 3; 
        
        outlier_mask = isoutlier(Z, 'median', 'ThresholdFactor', thresh_factor) | ...
                       isoutlier(Y_true, 'median', 'ThresholdFactor', thresh_factor);

        
        X1 = X1(~outlier_mask);
        X2 = X2(~outlier_mask);
        Z  = Z(~outlier_mask);
        Y_true = Y_true(~outlier_mask);
        hg_med_valid = hg_med(~outlier_mask);
        
        if length(Y_true) < 3
            fprintf('%-10.3f %-14s %s\n', curr_H, '----', '数据不足无法验证');
            continue;
        end
        
        if curr_H <= 0.016 + 1e-4
            seg_idx  = 1;
            Y_pred   = -0.6420 - 0.0143.*(Z.*X1) + 0.0270.*(X1.^2) + 0.0111.*(X2.^2) + 0.0010.*(Z.^2);
            
        elseif curr_H <= 0.020 + 1e-4
            seg_idx  = 2;
            Y_pred   = -1.6623 - 0.3647.*X1 - 0.0435.*X2 - 0.0538.*Z - 0.0194.*(X1.*X2) - 0.1311.*(X1.^2);
            
        elseif curr_H <= 0.024 + 1e-4
            seg_idx  = 3;
            Y_pred   = -0.0562 + 1.7766.*X1 + 1.7546.*X2 - 2.1121.*Z - 0.1471.*(Z.^2);
            
        elseif curr_H < 0.028 - 1e-4  
            seg_idx  = 4;
            Y_pred   = -0.1098 + 1.9142.*X1 + 1.9129.*X2 - 2.6424.*Z - 0.1865.*(Z.^2);
            
        else                          
            seg_idx  = 5;
            Y_pred   = -0.0838 + 2.7391.*X1 + 2.5143.*X2 - 3.4848.*Z - 0.2459.*(Z.^2);
        end
        seg_name = segment_names{seg_idx};
        
        % --- 计算真实的应力与预测应力 ---
        sig_true = exp(Y_true) .* (rho_g * g .* hg_med_valid);
        sig_pred = exp(Y_pred) .* (rho_g * g .* hg_med_valid);
        
        % 计算当前单 H 厚度下的 MAPE
        MAPE_sig = mean(abs(sig_true - sig_pred) ./ sig_true) * 100;
        group_metrics = [group_metrics; table(curr_H, numel(sig_true), MAPE_sig, ...
            'VariableNames', {'H_m','N','MAPE_sigma'})]; %#ok<AGROW>
        seg_group_mape{seg_idx} = [seg_group_mape{seg_idx}; MAPE_sig];

        % 打印单个 H 切片结果（只保留 MAPE）
        fprintf('H = %-6.3f m | MAPE = %6.2f%%   | %s\n', curr_H, MAPE_sig, seg_name);

        % 📌 将当前厚度切片的数据追加至对应分段容器
        seg_Y_true{seg_idx}   = [seg_Y_true{seg_idx}; Y_true(:)];
        seg_sig_true{seg_idx} = [seg_sig_true{seg_idx}; sig_true(:)];
        seg_sig_pred{seg_idx} = [seg_sig_pred{seg_idx}; sig_pred(:)];

        % 📌 将当前厚度切片的数据追加至数据集全局数组
        all_Y_true   = [all_Y_true; Y_true(:)];
        all_sig_true = [all_sig_true; sig_true(:)];
        all_sig_pred = [all_sig_pred; sig_pred(:)];
    end
    
    % 4. 各分段总体 R^2 与各厚度组等权平均 MAPE
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('📌 【各分段总体评估结果】:\n');
    for s_idx = 1:5
        if isempty(seg_Y_true{s_idx})
            continue;
        end

        % 真实应力空间 (sigma) 的分段 R^2
        SStot_sig = sum((seg_sig_true{s_idx} - mean(seg_sig_true{s_idx})).^2);
        SSres_sig = sum((seg_sig_true{s_idx} - seg_sig_pred{s_idx}).^2);
        R2_sig    = 1 - (SSres_sig / SStot_sig);

        % 每个厚度组权重相同，不按组内样本数加权
        seg_MAPE = mean(seg_group_mape{s_idx});

        fprintf('  - %-22s: R^2=%.4f | 各厚度组等权平均MAPE=%.2f%%\n', ...
            segment_names{s_idx}, R2_sig, seg_MAPE);
    end

    % 5. 整个数据集的总体 R^2 与各厚度组等权平均 MAPE
    if ~isempty(all_Y_true)
        overall_SStot_sig = sum((all_sig_true - mean(all_sig_true)).^2);
        overall_SSres_sig = sum((all_sig_true - all_sig_pred).^2);
        overall_R2_sig    = 1 - (overall_SSres_sig / overall_SStot_sig);

        mean_group_MAPE = mean(group_metrics.MAPE_sigma);
        validation_summary = [validation_summary; table(f_idx, numel(all_sig_true), overall_R2_sig, mean_group_MAPE, ...
            'VariableNames', {'Scenario','N','R2_sigma','Mean_group_MAPE'})]; %#ok<AGROW>

        fprintf('--------------------------------------------------------------------------------\n');
        fprintf('📊 【数据集整体评估结果】:\n');
        fprintf('   - 真实应力总体 R^2 (sigma): %.4f\n', overall_R2_sig);
        fprintf('   - 各厚度组等权平均 MAPE   : %.2f%%\n', mean_group_MAPE);
        fprintf('================================================================================\n\n');
    end
end