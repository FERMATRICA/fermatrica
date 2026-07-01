import hashlib
import os
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _dhash(s):
    """Deterministic string hash — stable across Python processes."""
    return int(hashlib.md5(str(s).encode()).digest().hex(), 16)


def write_sheet(ws, headers, rows, col_widths):
    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h)
    for ri, row in enumerate(rows, 2):
        for ci, val in enumerate(row, 1):
            ws.cell(row=ri, column=ci, value=val)
    for ci, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(ci)].width = w


dt = pd.read_excel(
    os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'data_processed', 'data_for_sample_p04.xlsx'),
    parse_dates=['date'])
train_mask = dt['listed'].isin([2])


def train_mean(var):
    m = (dt[var] > 0.0) & train_mask
    if m.sum() == 0:
        return dt.loc[train_mask, var].mean()
    return dt.loc[m, var].mean()


def train_std(var):
    m = (dt[var] > 0.0) & train_mask
    if m.sum() == 0:
        return dt.loc[train_mask, var].std()
    return dt.loc[m, var].std()


# Brand media stats
tv_avg = train_mean('brnd_tv_ots_clip')
tv_std = train_std('brnd_tv_ots_clip')
ooh_avg = train_mean('brnd_ots_ooh')
ooh_std = train_std('brnd_ots_ooh')
dgt_perf_avg = train_mean('brnd_cl_dgtind_perform')
dgt_perf_std = train_std('brnd_cl_dgtind_perform')
dgt_disp_avg = train_mean('brnd_cl_dgtind_display')
dgt_disp_std = train_std('brnd_cl_dgtind_display')
dgt_vid_avg = train_mean('brnd_cl_dgtind_video')
dgt_vid_std = train_std('brnd_cl_dgtind_video')
dgt_spec_avg = train_mean('brnd_cl_dgtind_special')
dgt_spec_std = train_std('brnd_cl_dgtind_special')
dgt_indoor_avg = train_mean('brnd_cl_dgtind_indoor')
dgt_indoor_std = train_std('brnd_cl_dgtind_indoor')

# Region media stats
reg_tv_avg = train_mean('reg_tv_ots_clip')
reg_tv_std = train_std('reg_tv_ots_clip')
reg_ooh_avg = train_mean('reg_ots_ooh')
reg_ooh_std = train_std('reg_ots_ooh')
reg_dgt_perf_avg = train_mean('reg_cl_dgtind_perform')
reg_dgt_perf_std = train_std('reg_cl_dgtind_perform')
reg_dgt_disp_avg = train_mean('reg_cl_dgtind_display')
reg_dgt_disp_std = train_std('reg_cl_dgtind_display')
reg_dgt_vid_avg = train_mean('reg_cl_dgtind_video')
reg_dgt_vid_std = train_std('reg_cl_dgtind_video')
reg_dgt_spec_avg = train_mean('reg_cl_dgtind_special')
reg_dgt_spec_std = train_std('reg_cl_dgtind_special')
reg_dgt_indoor_avg = train_mean('reg_cl_dgtind_indoor')
reg_dgt_indoor_std = train_std('reg_cl_dgtind_indoor')

ecom_avg = train_mean('e_com')
ecom_std = train_std('e_com')
pos_avg = train_mean('pos')
pos_std = train_std('pos')
pr_avg = train_mean('pr_costs_sm')
pr_std = train_std('pr_costs_sm')

di_avg = train_mean('di_soph_ssn_median_sm')
di_std = train_std('di_soph_ssn_median_sm')
di_share_avg = train_mean('di_soph_ssn_median_sm_share')
di_share_std = train_std('di_soph_ssn_median_sm_share')

price_mar_n = 17
price_mar_avg = train_mean('price_ret_rel_market')
price_mar_std = train_std('price_ret_rel_market')

print('Stats computed')

out_path = os.path.join(_SCRIPT_DIR, '..', 'model_data', 'model_conf.xlsx')
wb = Workbook()

params_headers = ['variable', 'fun', 'arg', 'lower', 'upper', 'value', 'type', 'fixed', 'if_active',
                  'index_vars', 'index_aggr', 'index_free_var', 'set_start', 'formula']

FM = 'float64'

params_rows = [
    # === Distribution: di_soph_ssn_median_sm (brand-consistent, softmax) ===
    ['di_soph_ssn_median_sm', 'softmax', 'lambda', 5.0, 40.0, 33.446043, FM, 0, 1],
    ['di_soph_ssn_median_sm', 'softmax', 'avg', 0.0, 100.0, di_avg, FM, 0, 1, None, None, None, 0,
     '''ds.loc[(ds["di_soph_ssn_median_sm"] > 0.0) & (ds["listed"].isin([2])), "di_soph_ssn_median_sm"].mean()'''],
    ['di_soph_ssn_median_sm', 'softmax', 'std', None, None, di_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["di_soph_ssn_median_sm"] > 0.0) & (ds["listed"].isin([2])), "di_soph_ssn_median_sm"].std()'''],

    # DI share with kpi_coef weighting (regional share)
    ['di_soph_ssn_median_sm', 'share', 'coef_var', None, None, 'kpi_coef', 'str', 1, 1],
    ['di_soph_ssn_median_sm', 'share', 'include_coef_var', None, None, True, 'bool', 1, 1],
    ['di_soph_ssn_median_sm', 'share', 'index_vars', None, None, '"market","date","listed"', 'str', 1, 1],

    ['di_soph_ssn_median_sm_share', 'softmax', 'lambda', 2.0, 30.0, 11.021992, FM, 1, 1],
    ['di_soph_ssn_median_sm_share', 'softmax', 'avg', 0.0, 1.0, di_share_avg, FM, 1, 1],
    ['di_soph_ssn_median_sm_share', 'softmax', 'std', None, None, di_share_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["di_soph_ssn_median_sm_share"] > 0.0) & (ds["listed"].isin([2])), "di_soph_ssn_median_sm_share"].std()'''],

    # === Price: price_ret (region-specific: index_vars=bs_key) ===
    ['price_ret', 'mar', 'n', 10.0, 18.0, price_mar_n, 'int', 0, 1, '"bs_key"'],
    ['price_ret', 'mar', 'if_scale', None, None, False, 'bool', 1, 1, '"bs_key"'],
    ['price_ret', 'price', 'share_threshold', 0.0, 0.075, 0.001, FM, 0, 1],
    ['price_ret', 'price', 'threshold_var', None, None, '"units_base_share"', 'str', 1, 1],
    ['price_ret', 'price', 'weighted_by', None, None, '"di_soph_ssn_median"', 'str', 1, 1],
    ['price_ret', 'price', 'index_vars_dim', None, None, '"market"', 'str', 1, 1],
    ['price_ret', 'price', 'index_vars_time', None, None, '"date"', 'str', 1, 1],
    ['price_ret', 'price', 'year_start', None, None, 2021, 'int', 1, 1],
    ['price_ret', 'price', 'if_clutter', None, None, False, 'bool', 1, 1],
    ['price_ret', 'price', 'inn_coef', 0.0, 3.0, 0.0, FM, 1, 1],

    # === adhoc_unparam_fin (unparametrized transformations) ===
    ['adhoc_unparam_fin', 'code_py.adhoc.model.adhoc_unparam_fin', 'dummy', None, None, False, 'bool', 1, 1],

    # === Brand TV softmax (global curve, per-region lag/adstock) ===
    ['brnd_tv_ots_clip', 'softmax', 'lambda', 5.0, 20.0, 9.0, FM, 0, 1],
    ['brnd_tv_ots_clip', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_tv_ots_clip', 'softmax', 'avg', None, None, tv_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_tv_ots_clip"] > 0.0) & (ds["listed"].isin([2])), "brnd_tv_ots_clip"].mean()'''],
    ['brnd_tv_ots_clip', 'softmax', 'std', None, None, tv_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_tv_ots_clip"] > 0.0) & (ds["listed"].isin([2])), "brnd_tv_ots_clip"].std()'''],
    ['brnd_tv_ots_clip_softmax', 'lag', 'n', 0.0, 3.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_tv_ots_clip_softmax_lag', 'adstockp', 'a', 0.15, 0.95, 0.321491, FM, 0, 1, '"bs_key"', 'max'],

    # === Brand OOH softmax ===
    ['brnd_ots_ooh', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 0, 1],
    ['brnd_ots_ooh', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_ots_ooh', 'softmax', 'avg', None, None, ooh_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_ots_ooh"] > 0.0) & (ds["listed"].isin([2])), "brnd_ots_ooh"].mean()'''],
    ['brnd_ots_ooh', 'softmax', 'std', None, None, ooh_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_ots_ooh"] > 0.0) & (ds["listed"].isin([2])), "brnd_ots_ooh"].std()'''],
    ['brnd_ots_ooh_softmax', 'lag', 'n', 0.0, 4.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_ots_ooh_softmax_lag', 'adstockp', 'a', 0.0, 0.95, 0.484365, FM, 0, 1, '"bs_key"', 'max'],

    # === Brand Digital Performance ===
    ['brnd_cl_dgtind_perform', 'lag', 'n', 0.0, 3.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_cl_dgtind_perform_lag', 'softmax', 'lambda', 6.0, 30.0, 15.0, FM, 0, 1],
    ['brnd_cl_dgtind_perform_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_cl_dgtind_perform_lag', 'softmax', 'avg', None, None, dgt_perf_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_perform"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_perform"].mean()'''],
    ['brnd_cl_dgtind_perform_lag', 'softmax', 'std', None, None, dgt_perf_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_perform"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_perform"].std()'''],
    ['brnd_cl_dgtind_perform_lag_softmax', 'adstockp', 'a', 0.0, 0.25, 0.135582, FM, 0, 1, '"bs_key"', 'max'],

    # === Brand Digital Display ===
    ['brnd_cl_dgtind_display', 'lag', 'n', 0.0, 2.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_cl_dgtind_display_lag', 'softmax', 'lambda', 4.0, 30.0, 5.0, FM, 0, 1],
    ['brnd_cl_dgtind_display_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_cl_dgtind_display_lag', 'softmax', 'avg', None, None, dgt_disp_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_display"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_display"].mean()'''],
    ['brnd_cl_dgtind_display_lag', 'softmax', 'std', None, None, dgt_disp_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_display"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_display"].std()'''],
    ['brnd_cl_dgtind_display_lag_softmax', 'adstockp', 'a', 0.0, 0.85, 0.18, FM, 0, 1, '"bs_key"', 'max'],

    # === Brand Digital Video ===
    ['brnd_cl_dgtind_video', 'lag', 'n', 0.0, 2.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_cl_dgtind_video_lag', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 0, 1],
    ['brnd_cl_dgtind_video_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_cl_dgtind_video_lag', 'softmax', 'avg', None, None, dgt_vid_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_video"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_video"].mean()'''],
    ['brnd_cl_dgtind_video_lag', 'softmax', 'std', None, None, dgt_vid_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_video"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_video"].std()'''],
    ['brnd_cl_dgtind_video_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.068242, FM, 0, 1, '"bs_key"', 'max'],

    # === Brand Digital Special ===
    ['brnd_cl_dgtind_special', 'lag', 'n', 0.0, 3.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_cl_dgtind_special_lag', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 0, 1],
    ['brnd_cl_dgtind_special_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_cl_dgtind_special_lag', 'softmax', 'avg', None, None, dgt_spec_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_special"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_special"].mean()'''],
    ['brnd_cl_dgtind_special_lag', 'softmax', 'std', None, None, dgt_spec_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_special"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_special"].std()'''],
    ['brnd_cl_dgtind_special_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.000232, FM, 0, 1, '"bs_key"', 'max'],

    # === Brand Digital Indoor ===
    ['brnd_cl_dgtind_indoor', 'lag', 'n', 0.0, 3.0, 3, 'int', 0, 1, '"bs_key"', 'max'],
    ['brnd_cl_dgtind_indoor_lag', 'softmax', 'lambda', 2.0, 30.0, 5.0, FM, 0, 1],
    ['brnd_cl_dgtind_indoor_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['brnd_cl_dgtind_indoor_lag', 'softmax', 'avg', None, None, dgt_indoor_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_indoor"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_indoor"].mean()'''],
    ['brnd_cl_dgtind_indoor_lag', 'softmax', 'std', None, None, dgt_indoor_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["brnd_cl_dgtind_indoor"] > 0.0) & (ds["listed"].isin([2])), "brnd_cl_dgtind_indoor"].std()'''],
    ['brnd_cl_dgtind_indoor_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.8, FM, 0, 1, '"bs_key"', 'max'],

    # === Region TV softmax (per-region) ===
    ['reg_tv_ots_clip', 'softmax', 'lambda', 5.0, 20.0, 9.0, FM, 0, 1],
    ['reg_tv_ots_clip', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_tv_ots_clip', 'softmax', 'avg', None, None, reg_tv_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_tv_ots_clip"] > 0.0) & (ds["listed"].isin([2])), "reg_tv_ots_clip"].mean()'''],
    ['reg_tv_ots_clip', 'softmax', 'std', None, None, reg_tv_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_tv_ots_clip"] > 0.0) & (ds["listed"].isin([2])), "reg_tv_ots_clip"].std()'''],
    ['reg_tv_ots_clip_softmax', 'lag', 'n', 0.0, 3.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_tv_ots_clip_softmax_lag', 'adstockp', 'a', 0.15, 0.95, 0.321491, FM, 0, 1, '"bs_key"', 'max'],

    # === Region OOH softmax ===
    ['reg_ots_ooh', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 0, 1, None],
    ['reg_ots_ooh', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_ots_ooh', 'softmax', 'avg', None, None, reg_ooh_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_ots_ooh"] > 0.0) & (ds["listed"].isin([2])), "reg_ots_ooh"].mean()'''],
    ['reg_ots_ooh', 'softmax', 'std', None, None, reg_ooh_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_ots_ooh"] > 0.0) & (ds["listed"].isin([2])), "reg_ots_ooh"].std()'''],
    ['reg_ots_ooh_softmax', 'lag', 'n', 0.0, 4.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_ots_ooh_softmax_lag', 'adstockp', 'a', 0.0, 0.95, 0.484365, FM, 0, 1, '"bs_key"', 'max'],

    # === Region Digital Performance ===
    ['reg_cl_dgtind_perform', 'lag', 'n', 0.0, 3.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_cl_dgtind_perform_lag', 'softmax', 'lambda', 6.0, 30.0, 15.0, FM, 0, 1, None],
    ['reg_cl_dgtind_perform_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_cl_dgtind_perform_lag', 'softmax', 'avg', None, None, reg_dgt_perf_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_perform"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_perform"].mean()'''],
    ['reg_cl_dgtind_perform_lag', 'softmax', 'std', None, None, reg_dgt_perf_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_perform"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_perform"].std()'''],
    ['reg_cl_dgtind_perform_lag_softmax', 'adstockp', 'a', 0.0, 0.25, 0.135582, FM, 0, 1, '"bs_key"', 'max'],

    # === Region Digital Display ===
    ['reg_cl_dgtind_display', 'lag', 'n', 0.0, 2.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_cl_dgtind_display_lag', 'softmax', 'lambda', 4.0, 30.0, 5.0, FM, 0, 1, None],
    ['reg_cl_dgtind_display_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_cl_dgtind_display_lag', 'softmax', 'avg', None, None, reg_dgt_disp_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_display"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_display"].mean()'''],
    ['reg_cl_dgtind_display_lag', 'softmax', 'std', None, None, reg_dgt_disp_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_display"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_display"].std()'''],
    ['reg_cl_dgtind_display_lag_softmax', 'adstockp', 'a', 0.0, 0.85, 0.18, FM, 0, 1, '"bs_key"', 'max'],

    # === Region Digital Video ===
    ['reg_cl_dgtind_video', 'lag', 'n', 0.0, 2.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_cl_dgtind_video_lag', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 0, 1, None],
    ['reg_cl_dgtind_video_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_cl_dgtind_video_lag', 'softmax', 'avg', None, None, reg_dgt_vid_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_video"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_video"].mean()'''],
    ['reg_cl_dgtind_video_lag', 'softmax', 'std', None, None, reg_dgt_vid_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_video"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_video"].std()'''],
    ['reg_cl_dgtind_video_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.068242, FM, 0, 1, '"bs_key"', 'max'],

    # === Region Digital Special ===
    ['reg_cl_dgtind_special', 'lag', 'n', 0.0, 3.0, 0, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_cl_dgtind_special_lag', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 0, 1, None],
    ['reg_cl_dgtind_special_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_cl_dgtind_special_lag', 'softmax', 'avg', None, None, reg_dgt_spec_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_special"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_special"].mean()'''],
    ['reg_cl_dgtind_special_lag', 'softmax', 'std', None, None, reg_dgt_spec_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_special"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_special"].std()'''],
    ['reg_cl_dgtind_special_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.000232, FM, 0, 1, '"bs_key"', 'max'],

    # === Region Digital Indoor ===
    ['reg_cl_dgtind_indoor', 'lag', 'n', 0.0, 3.0, 3, 'int', 0, 1, '"bs_key"', 'max'],
    ['reg_cl_dgtind_indoor_lag', 'softmax', 'lambda', 2.0, 30.0, 5.0, FM, 0, 1, None],
    ['reg_cl_dgtind_indoor_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['reg_cl_dgtind_indoor_lag', 'softmax', 'avg', None, None, reg_dgt_indoor_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_indoor"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_indoor"].mean()'''],
    ['reg_cl_dgtind_indoor_lag', 'softmax', 'std', None, None, reg_dgt_indoor_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["reg_cl_dgtind_indoor"] > 0.0) & (ds["listed"].isin([2])), "reg_cl_dgtind_indoor"].std()'''],
    ['reg_cl_dgtind_indoor_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.8, FM, 0, 1, '"bs_key"', 'max'],

    # === E-com ===
    ['e_com', 'infl', 'infl', 0.9, 0.95, 0.95, FM, 1, 1],
    ['e_com', 'infl', 'year_start', None, None, 2021, FM, 1, 1],
    ['e_com_infl', 'lag', 'n', 0.0, 3.0, 2.967409, 'int', 1, 1, '"bs_key"', 'max'],
    ['e_com_infl_lag', 'softmax', 'lambda', 2.0, 30.0, 5.0, FM, 1, 1],
    ['e_com_infl_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['e_com_infl_lag', 'softmax', 'avg', None, None, ecom_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["e_com_infl"] > 0.0) & (ds["listed"].isin([2])), "e_com_infl"].mean()'''],
    ['e_com_infl_lag', 'softmax', 'std', None, None, ecom_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["e_com_infl"] > 0.0) & (ds["listed"].isin([2])), "e_com_infl"].std()'''],
    ['e_com_infl_lag_softmax', 'adstockp', 'a', 0.0, 0.85, 0.789203, FM, 1, 1, '"bs_key"', 'max'],

    # === POS ===
    ['pos', 'infl', 'infl', 0.85, 1.0, 0.9, FM, 1, 1],
    ['pos', 'infl', 'year_start', None, None, 2021, FM, 1, 1],
    ['pos_infl', 'lag', 'n', 0.0, 6.0, 2, 'int', 1, 1, '"bs_key"', 'max'],
    ['pos_infl_lag', 'softmax', 'lambda', 2.0, 30.0, 10.0, FM, 1, 1],
    ['pos_infl_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['pos_infl_lag', 'softmax', 'avg', None, None, pos_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["pos_infl"] > 0.0) & (ds["listed"].isin([2])), "pos_infl"].mean()'''],
    ['pos_infl_lag', 'softmax', 'std', None, None, pos_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["pos_infl"] > 0.0) & (ds["listed"].isin([2])), "pos_infl"].std()'''],
    ['pos_infl_lag_softmax', 'adstockp', 'a', 0.0, 0.85, 0.1, FM, 1, 1, '"bs_key"', 'max'],

    # === PR costs SM ===
    ['pr_costs_sm', 'infl', 'infl', 0.85, 1.0, 0.85, FM, 1, 1],
    ['pr_costs_sm', 'infl', 'year_start', None, None, 2021, FM, 1, 1],
    ['pr_costs_sm_infl', 'lag', 'n', 0.0, 8.0, 2, 'int', 1, 1, '"bs_key"', 'max'],
    ['pr_costs_sm_infl_lag', 'softmax', 'lambda', 3.0, 30.0, 5.0, FM, 1, 1],
    ['pr_costs_sm_infl_lag', 'softmax', 'ec50', 0, 3, 1, FM, 0, 1],
    ['pr_costs_sm_infl_lag', 'softmax', 'avg', None, None, pr_avg, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["pr_costs_sm_infl"] > 0.0) & (ds["listed"].isin([2])), "pr_costs_sm_infl"].mean()'''],
    ['pr_costs_sm_infl_lag', 'softmax', 'std', None, None, pr_std, FM, 1, 1, None, None, None, 1,
     '''ds.loc[(ds["pr_costs_sm_infl"] > 0.0) & (ds["listed"].isin([2])), "pr_costs_sm_infl"].std()'''],
    ['pr_costs_sm_infl_lag_softmax', 'adstockp', 'a', 0.0, 0.95, 0.319055, FM, 1, 1, '"bs_key"', 'max'],

    # === Coefs (LHS) ===
    [None, 'coef', 'tt_pk_tr_ssn', 0.1, 7.0, 1.5, FM, 0, 1],
    [None, 'coef', 'hdays', -0.1, 0.0, -0.0394, FM, 0, 1],
    [None, 'coef', 'price_index', -0.99, 0.0, -0.2, FM, 0, 1],
]

for r in params_rows:
    while len(r) < 14:
        r.append(None)

ws_params = wb.active
ws_params.title = 'params'
write_sheet(ws_params, params_headers, params_rows, [30, 55, 18, 10, 18, 18, 10, 8, 10, 25, 10, 15, 8, 100])

# ═══════════════════════════════════════════
# Sheet: RHS
# ═══════════════════════════════════════════
rhs_rows = [
    # Distribution
    ['di_soph_ssn_median_sm_softmax', 1, 1, 1, 1, 1, 2, '01_Distribution'],

    # Brand media (with kpi_coef interaction to distribute brand-level spend to regions)
    ['brnd_tv_ots_clip_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '06_Brand_TV'],
    ['brnd_ots_ooh_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '07_Brand_OOH'],
    ['brnd_cl_dgtind_perform_lag_softmax_adstockp:kpi_coef', 1, 1, 1, 1, 1, 2, '10_Brand_Digital_Performance'],
    ['brnd_cl_dgtind_display_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '11_Brand_Digital_DA'],
    ['brnd_cl_dgtind_video_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '11_Brand_Digital_Video'],
    ['brnd_cl_dgtind_special_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '11_Brand_Digital_Special'],
    ['brnd_cl_dgtind_indoor_lag_softmax_adstockp:kpi_coef', 1, 1, 1, 1, 1, 2, '11_Brand_Digital_Indoor'],

    # Region media (already region-specific, no kpi_coef)
    ['reg_tv_ots_clip_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax', 1, 1, 1, 1, 1, 2, '06_Region_TV'],
    ['reg_ots_ooh_softmax_lag_adstockp:di_soph_ssn_median_sm_softmax', 1, 1, 1, 1, 1, 2, '07_Region_OOH'],
    ['reg_cl_dgtind_perform_lag_softmax_adstockp', 1, 1, 1, 1, 1, 2, '10_Region_Digital_Performance'],
    ['reg_cl_dgtind_display_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax', 1, 1, 1, 1, 1, 2, '11_Region_Digital_DA'],
    ['reg_cl_dgtind_video_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax', 1, 1, 1, 1, 1, 2, '11_Region_Digital_Video'],
    ['reg_cl_dgtind_special_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax', 1, 1, 1, 1, 1, 2, '11_Region_Digital_Special'],
    ['reg_cl_dgtind_indoor_lag_softmax_adstockp', 1, 1, 1, 1, 1, 2, '11_Region_Digital_Indoor'],

    # Brand-level non-media (with kpi_coef)
    ['e_com_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '13_E-com'],
    ['pr_costs_sm_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '14_PR'],
    ['pos_infl_lag_softmax_adstockp:di_soph_ssn_median_sm_softmax:kpi_coef', 1, 1, 1, 1, 1, 2, '14_POS'],
]

ws_rhs = wb.create_sheet('RHS')
write_sheet(ws_rhs, ['token', 'fixed', 'if_active', 'sign', 'sign_weight', 'signif_weight', 'signif_threshold', 'display_var'],
            rhs_rows, [80, 8, 10, 8, 14, 14, 18, 30])

# ═══════════════════════════════════════════
# Sheet: setup
# ═══════════════════════════════════════════
ws_setup = wb.create_sheet('setup')
write_sheet(ws_setup, ['key', 'value'], [
    ['Y_var', 'units'],
    ['model_type', 'LME'],
    ['fixed_effect_var', 'bs_key'],
    ['target_superbrand', 'brand_1'],
    ['period_var', 'week'],
    ['price_var', 'price_ret'],
    ['conversion_var', None],
    ['subfolder_orig', '20260617'],
    ['custom_scoring', 'rmse_bs_key * ((sign_t_value_st + 1) ** 10)'],
    ['target_audience', 'all_20_45'],
    ['bs_key', 'superbrand'],
    ['bs_key', 'master'],
    ['bs_key', 'market'],
    ['lme_method', 'powell'],
    ['lme_method', 'lbfgs'],
], [20, 80])

# ═══════════════════════════════════════════
# Sheet: scoring
# ═══════════════════════════════════════════
ws_scoring = wb.create_sheet('scoring')
write_sheet(ws_scoring, ['metrics', 'if_invert', 'width', 'weight', 'if_active'], [
    ['rmse_bs_key', 0, 0.3, 0.166667, 1],
    ['r_squared', 1, 0.3, 1.0, 0],
    ['r_squared_adjusted', 0, 0.3, 1.0, 0],
    ['t_value', 1, 0.3, 0.5, 1],
    ['p_value', 0, 0.3, 0.166667, 1],
    ['sign_simple', 0, 0.3, 1.0, 0],
    ['sign_t_value_st', 0, 0.3, 0.166667, 1],
    ['vif', 0, 0.3, 1.0, 0],
    ['durbin_watson', 0, 0.3, 1.0, 0],
], [20, 10, 8, 10, 10])

# ═══════════════════════════════════════════
# Sheet: LHS
# ═══════════════════════════════════════════
ws_lhs = wb.create_sheet('LHS')
write_sheet(ws_lhs, ['type', 'name', 'token', 'if_active', 'display_var'], [
    ['multiplicative', 'ssn_mrk', '(coef.tt_pk_tr_ssn + 1) / (coef.tt_pk_tr_ssn * ds.tt_pk_tr_ssn  + 1)', 1, '20_seasonality'],
    ['multiplicative', 'hdays', '1 + ds.hdays * coef.hdays', 1, '20_seasonality'],
    ['multiplicative', 'price_index', '1 + coef.price_index * (ds.price_index - 1)', 1, '05_Price index'],
], [18, 16, 80, 10, 25])

# ═══════════════════════════════════════════
# Sheet: trans_path_df
# ═══════════════════════════════════════════
ws_tpdf = wb.create_sheet('trans_path_df')
write_sheet(ws_tpdf, ['variable', 'variable_fin', 'price', 'display_var'], [
    ['brnd_tv_ots_clip', None, 100.0, None],
    ['brnd_ots_ooh', None, 100.0, None],
    ['brnd_cl_dgtind_perform', None, 10.0, None],
    ['brnd_cl_dgtind_display', None, 0.1, None],
    ['brnd_cl_dgtind_video', None, 0.3, None],
    ['brnd_cl_dgtind_special', None, 0.2, None],
    ['brnd_cl_dgtind_indoor', None, 0.1, None],
    ['reg_tv_ots_clip', None, 100.0, None],
    ['reg_ots_ooh', None, 100.0, None],
    ['reg_cl_dgtind_perform', None, 10.0, None],
    ['reg_cl_dgtind_display', None, 0.1, None],
    ['reg_cl_dgtind_video', None, 0.3, None],
    ['reg_cl_dgtind_special', None, 0.2, None],
    ['reg_cl_dgtind_indoor', None, 0.1, None],
    ['e_com', None, 1.0, None],
    ['pr_costs_sm', None, 1.0, None],
    ['pos', None, 1.0, None],
], [50, 70, 10, 70])

wb.save(out_path)
print(f'Created: {out_path}')

# ═══════════════════════════════════════════
# options.xlsx — per-master budgets, translation, media_prices
# ═══════════════════════════════════════════

options_pth = os.path.join(_SCRIPT_DIR, '..', 'model_data', 'options', 'options.xlsx')
wb_o = Workbook()

# master-specific media multipliers
regions = ['Moscow', 'Paris', 'Warsaw', 'Cairo']
media_mult = {'Moscow': 0.5, 'Paris': 0.4, 'Warsaw': 0.3, 'Cairo': 0.25}

# defaults for brand-level budgets (same across all masters)
brnd_budgets = {
    'bdg_brnd_tv': 100,
    'bdg_brnd_ooh': 20,
    'bdg_brnd_dgt_perform': 10,
    'bdg_brnd_dgt_display': 20,
    'bdg_brnd_dgt_video': 20,
    'bdg_brnd_dgt_special': 10,
    'bdg_brnd_dgt_indoor': 10,
    'bdg_ecom': 5,
    'bdg_pr': 5,
    'bdg_pos': 10,
}

# regional budget keys and their brand-level reference for scaling
reg_budget_keys = ['bdg_reg_tv', 'bdg_reg_ooh', 'bdg_reg_dgt_perform',
                   'bdg_reg_dgt_display', 'bdg_reg_dgt_video',
                   'bdg_reg_dgt_special', 'bdg_reg_dgt_indoor']
reg_ref_keys = ['bdg_brnd_tv', 'bdg_brnd_ooh', 'bdg_brnd_dgt_perform',
                'bdg_brnd_dgt_display', 'bdg_brnd_dgt_video',
                'bdg_brnd_dgt_special', 'bdg_brnd_dgt_indoor']

nobdg_keys = ['nobdg_price_index']

all_budget_keys = list(brnd_budgets.keys()) + reg_budget_keys + nobdg_keys

# ═══ Sheet: translation ═══
translation_headers = ['item', 'function', 'variable', 'max_budg', 'max_obs_budg']

translation_rows = [
    # Brand-level
    ['bdg_brnd_tv', 'basic', 'brnd_tv_ots_clip', 500, round(500 / 1.35)],
    ['bdg_brnd_ooh', 'basic', 'brnd_ots_ooh', 100, round(100 / 1.35)],
    ['bdg_brnd_dgt_perform', 'basic', 'brnd_cl_dgtind_perform', 50, round(50 / 1.35)],
    ['bdg_brnd_dgt_display', 'basic', 'brnd_cl_dgtind_display', 100, round(100 / 1.35)],
    ['bdg_brnd_dgt_video', 'basic', 'brnd_cl_dgtind_video', 100, round(100 / 1.35)],
    ['bdg_brnd_dgt_special', 'basic', 'brnd_cl_dgtind_special', 100, round(100 / 1.35)],
    ['bdg_brnd_dgt_indoor', 'basic', 'brnd_cl_dgtind_indoor', 100, round(100 / 1.35)],
    # Regional
    ['bdg_reg_tv', 'basic', 'reg_tv_ots_clip', 500, round(500 / 1.35)],
    ['bdg_reg_ooh', 'basic', 'reg_ots_ooh', 100, round(100 / 1.35)],
    ['bdg_reg_dgt_perform', 'basic', 'reg_cl_dgtind_perform', 50, round(50 / 1.35)],
    ['bdg_reg_dgt_display', 'basic', 'reg_cl_dgtind_display', 100, round(100 / 1.35)],
    ['bdg_reg_dgt_video', 'basic', 'reg_cl_dgtind_video', 100, round(100 / 1.35)],
    ['bdg_reg_dgt_special', 'basic', 'reg_cl_dgtind_special', 100, round(100 / 1.35)],
    ['bdg_reg_dgt_indoor', 'basic', 'reg_cl_dgtind_indoor', 100, round(100 / 1.35)],
    # Non-media brand-level
    ['bdg_ecom', 'basic', 'e_com', 30, round(30 / 1.35)],
    ['bdg_pr', 'basic', 'pr_costs_sm', 30, round(30 / 1.35)],
    ['bdg_pos', 'basic', 'pos', 50, round(50 / 1.35)],
    # Price index (nobdg — no budget, fixed coefficient)
    ['nobdg_price_index', 'basic_fix', 'price_index', None, None],
]

ws_tr = wb_o.active
ws_tr.title = 'translation'
write_sheet(ws_tr, translation_headers, translation_rows,
            [22, 14, 30, 10, 14])

# ═══ Sheet: main ═══
main_headers = ['option', 'text_flags'] + all_budget_keys
main_rows = []

for opt in ['zero', 'predefined']:
    for rgn in regions:
        if opt == 'zero':
            row = ['zero', rgn]
            for k in all_budget_keys:
                if k.startswith('nobdg_'):
                    row.append(1.0)
                else:
                    row.append(0)
        else:
            row = ['predefined', rgn]
            mm = media_mult[rgn]
            for k in all_budget_keys:
                if k.startswith('nobdg_'):
                    row.append(1.0)
                elif k in brnd_budgets:
                    row.append(brnd_budgets[k])
                else:
                    idx = reg_budget_keys.index(k)
                    ref_key = reg_ref_keys[idx]
                    row.append(round(brnd_budgets[ref_key] * mm))
        main_rows.append(row)

ws_main = wb_o.create_sheet('main')
write_sheet(ws_main, main_headers, main_rows,
            [14, 16] + [14] * len(all_budget_keys))

# ═══ Sheet: current ═══
current_rows = []
for opt in ['zero', 'predefined']:
    for rgn in regions:
        if opt == 'zero':
            row = ['zero', rgn]
            for k in all_budget_keys:
                if k.startswith('nobdg_'):
                    row.append(1.0)
                else:
                    row.append(0)
        else:
            row = ['predefined', rgn]
            mm = media_mult[rgn]
            for k in all_budget_keys:
                if k.startswith('nobdg_'):
                    row.append(1.0)
                elif k in brnd_budgets:
                    row.append(brnd_budgets[k])
                else:
                    idx = reg_budget_keys.index(k)
                    ref_key = reg_ref_keys[idx]
                    row.append(round(brnd_budgets[ref_key] * mm))
        current_rows.append(row)

ws_current = wb_o.create_sheet('current')
write_sheet(ws_current, main_headers, current_rows,
            [14, 16] + [14] * len(all_budget_keys))

# ═══ Sheet: media_prices ═══
price_vars = ['brnd_tv_ots_clip', 'brnd_ots_ooh',
              'brnd_cl_dgtind_perform', 'brnd_cl_dgtind_display', 'brnd_cl_dgtind_video',
              'brnd_cl_dgtind_special', 'brnd_cl_dgtind_indoor',
              'reg_tv_ots_clip', 'reg_ots_ooh',
              'reg_cl_dgtind_perform', 'reg_cl_dgtind_display', 'reg_cl_dgtind_video',
              'reg_cl_dgtind_special', 'reg_cl_dgtind_indoor',
              'e_com', 'pr_costs_sm', 'pos']

mp_headers = ['master'] + price_vars

prices_from_tpdf = {
    'brnd_tv_ots_clip': 100.0, 'brnd_ots_ooh': 100.0,
    'brnd_cl_dgtind_perform': 10.0, 'brnd_cl_dgtind_display': 0.1,
    'brnd_cl_dgtind_video': 0.3, 'brnd_cl_dgtind_special': 0.2,
    'brnd_cl_dgtind_indoor': 1.0,
    'reg_tv_ots_clip': 100.0, 'reg_ots_ooh': 100.0,
    'reg_cl_dgtind_perform': 10.0, 'reg_cl_dgtind_display': 0.1,
    'reg_cl_dgtind_video': 0.3, 'reg_cl_dgtind_special': 0.2,
    'reg_cl_dgtind_indoor': 1.0,
    'e_com': 1.0, 'pr_costs_sm': 1.0, 'pos': 1.0,
}

mp_rows = []
for rgn in regions:
    row = [rgn] + [prices_from_tpdf[v] for v in price_vars]
    mp_rows.append(row)

ws_mp = wb_o.create_sheet('media_prices')
write_sheet(ws_mp, mp_headers, mp_rows, [14] + [14] * len(price_vars))

wb_o.save(options_pth)
print(f'Created: {options_pth}')

# ═══════════════════════════════════════════
# seasonality.xlsx — weekly seasonality coefficients
# ═══════════════════════════════════════════

ssn_pth = os.path.join(_SCRIPT_DIR, '..', 'model_data', 'options', 'seasonality.xlsx')
wb_s = Workbook()

ssn_cols = ['week_number',
            'brnd_tv', 'reg_tv',
            'brnd_ooh', 'reg_ooh',
            'brnd_dgt_perform', 'reg_dgt_perform',
            'brnd_dgt_display', 'reg_dgt_display',
            'brnd_dgt_video', 'reg_dgt_video',
            'brnd_dgt_special', 'reg_dgt_special',
            'brnd_dgt_indoor', 'reg_dgt_indoor',
            'ecom', 'pr', 'pos',
            'price_index']

n_weeks = 52
ssn_rows = []

for wk in range(1, n_weeks + 1):
    t = 2 * np.pi * wk / n_weeks
    row = [wk]
    for col in ssn_cols[1:]:
        if col.startswith('brnd_'):
            v = max(0.005, (0.5 + 0.3 * np.sin(t + 2.0)) / n_weeks)
        elif col.startswith('reg_'):
            v = max(0.005, (0.5 + 0.4 * np.sin(t + 1.5 + _dhash(col) % 3)) / n_weeks)
        else:
            v = max(0.005, (0.5 + 0.3 * np.sin(t + 3.0)) / n_weeks)
        row.append(round(v, 6))
    ssn_rows.append(row)

ws_ssn = wb_s.active
ws_ssn.title = 'data'
write_sheet(ws_ssn, ssn_cols, ssn_rows, [14] + [16] * (len(ssn_cols) - 1))

wb_s.save(ssn_pth)
print(f'Created: {ssn_pth}')

print('\nAll files generated successfully.')
