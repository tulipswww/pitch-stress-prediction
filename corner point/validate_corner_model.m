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
script_dir=fileparts(mfilename('fullpath'));
model_info=corner_model_cache;
terms=reshape(model_info.model_terms,1,[]);
coefficients=[model_info.model_intercept;model_info.model_coefficients(:)];
file_list={ ...
    'corner_validation_S1.xlsx', ...
    'corner_validation_S2.xlsx', ...
    'corner_validation_S3.xlsx'};
title_list={'S1 沥青厚度内插','S2 玻璃厚度外推','S3 组合验证'};
validation_results=struct([]);
for f_idx=1:numel(file_list)
    fprintf('\n========== %s ==========\n',title_list{f_idx});
    filename=fullfile(script_dir,file_list{f_idx});
    if ~isfile(filename), error('找不到验证文件：%s',filename); end
    % 全厚度采用 fit_corner_model 相同清洗规则，不再附加21~23mm特例或应力限值。
    [tbl,stats,~]=prepare_corner_data(filename,model_info.preprocessing.mad_threshold);
    X=[ones(height(tbl),1),tbl{:,terms}];
    Y_pred=X*coefficients;
    sigma_true=exp(tbl.Y); sigma_pred=exp(Y_pred);
    H_values=unique(tbl.H);
    group_metrics=table();
    fprintf(' H(m)  |   N | MAPE(sigma)\n');
    for h_idx=1:numel(H_values)
        h=H_values(h_idx); k=tbl.H==h;
        mape=mean(abs(sigma_pred(k)-sigma_true(k))./sigma_true(k))*100;
        fprintf(' %.3f | %3d | %9.4f%%\n',h,sum(k),mape);
        group_metrics=[group_metrics;table(h,sum(k),mape, ...
            'VariableNames',{'H_m','N','MAPE_sigma'})]; %#ok<AGROW>
    end
    mean_group_MAPE=mean(group_metrics.MAPE_sigma);
    fprintf('总体 N=%d | 各厚度组等权平均MAPE=%.4f%%\n',height(tbl),mean_group_MAPE);
    validation_results(f_idx).scenario=title_list{f_idx};
    validation_results(f_idx).file=filename;
    validation_results(f_idx).cleaning_stats=stats;
    validation_results(f_idx).group_metrics=group_metrics;
    validation_results(f_idx).mean_group_MAPE=mean_group_MAPE;
    validation_results(f_idx).samples=table(tbl.H,tbl.Z,tbl.X1,tbl.X2, ...
        sigma_true,sigma_pred,'VariableNames',{'H_m','Z','X1','X2','Sigma_true_Pa','Sigma_pred_Pa'});
end
fprintf('验证结果保留在工作区 validation_results 中，未生成文件。\n');

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
