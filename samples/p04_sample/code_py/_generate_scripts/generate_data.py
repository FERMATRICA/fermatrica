import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

# ---------- config ----------
regions = {
    'Moscow': {'bs_key': 1, 'price_base': 220, 'media_mult': 1.0},
    'Paris':  {'bs_key': 2, 'price_base': 280, 'media_mult': 0.8},
    'Warsaw': {'bs_key': 3, 'price_base': 200, 'media_mult': 0.6},
    'Cairo':  {'bs_key': 4, 'price_base': 180, 'media_mult': 0.5},
}

date_start = datetime(2019, 1, 7)
date_end = datetime(2030, 12, 31)
n_weeks = (date_end - date_start).days // 7 + 1
dates = [date_start + timedelta(weeks=i) for i in range(n_weeks)]

N = len(regions) * len(dates)


def campaign_spikes(n, n_max=1.0, prob_on=0.25, spike_duration=2):
    arr = np.zeros(n)
    i = 0
    while i < n:
        if np.random.rand() < prob_on:
            height = np.random.uniform(0.3, n_max)
            dur = min(spike_duration + np.random.randint(0, 3), n - i)
            arr[i:i + dur] = height
            i += dur
        else:
            i += 1
    return arr


def random_walk(n, start=0.0, step_std=0.02):
    changes = np.random.randn(n) * step_std
    series = start + np.cumsum(changes)
    return series


t_all = np.arange(n_weeks)

# ---------- region-independent (brand-consistent) series ----------
trend = 0.4 + 0.4 * t_all / n_weeks
di_raw_brand = np.clip(trend + 0.1 * np.random.randn(n_weeks), 0.05, 0.95) * 100
di_soph_ssn_raw_brand = di_raw_brand

# Brand media (consistent per week across all regions)
tv_ots = 50000 + np.random.randn(n_weeks) * 5000 + campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 800000
tv_ots = np.clip(tv_ots, 0, 1.3e9).astype(int)

dgt_perf = campaign_spikes(n_weeks, n_max=1, prob_on=0.3) * 30000 + np.random.randn(n_weeks) * 1000
dgt_perf = np.clip(dgt_perf, 0, 40000).astype(int)

dgt_disp = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 8000000 + np.random.randn(n_weeks) * 200000
dgt_disp = np.clip(dgt_disp, 0, 9700000).astype(int)

dgt_vid = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 8000000 + np.random.randn(n_weeks) * 200000
dgt_vid = np.clip(dgt_vid, 0, 9700000).astype(int)

dgt_spec = campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 15000000 + np.random.randn(n_weeks) * 500000
dgt_spec = np.clip(dgt_spec, 0, 22000000).astype(int)

dgt_indoor = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 1200000 + np.random.randn(n_weeks) * 50000
dgt_indoor = np.clip(dgt_indoor, 0, 1600000).astype(int)

ooh_ots = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 10000 + np.random.randn(n_weeks) * 500
ooh_ots = np.clip(ooh_ots, 0, 16000).astype(int)

ecom = campaign_spikes(n_weeks, n_max=1, prob_on=0.3) * 900000 + np.random.randn(n_weeks) * 30000
ecom = np.clip(ecom, 0, 1180000).astype(int)

pos_val = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 4000000 + np.random.randn(n_weeks) * 100000
pos_val = np.clip(pos_val, 0, 5000000).astype(int)

pr_cost = campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 2000000 + np.random.randn(n_weeks) * 50000
pr_cost = np.clip(pr_cost, 0, 3000000).astype(int)

# ---------- region-specific series ----------
rows = []

for region_name, rcfg in regions.items():
    bs = rcfg['bs_key']
    mmlt = rcfg['media_mult']
    price_base = rcfg['price_base']

    # kpi_coef: regional sales share by date, generated as random walk + softmax per date
    rw = random_walk(n_weeks, start=0.0, step_std=0.03)
    kpi_raw = np.exp(rw)

    # Seasonality per region (different climate patterns)
    if region_name == 'Moscow':
        ssn_amp = 0.5
        ssn_phase = 0
        ssn_mean = 0.3
    elif region_name == 'Paris':
        ssn_amp = 0.35
        ssn_phase = 0.5
        ssn_mean = 0.5
    elif region_name == 'Warsaw':
        ssn_amp = 0.55
        ssn_phase = 0.2
        ssn_mean = 0.25
    else:
        ssn_amp = 0.2
        ssn_phase = 3.0
        ssn_mean = 0.7

    ssn_reg = ssn_mean + ssn_amp * np.sin(2 * np.pi * t_all / 52 + ssn_phase)
    tt_pk_reg = 0.3 + 0.7 * ssn_reg
    tt_pk_reg = tt_pk_reg - np.min(tt_pk_reg) + 1

    # Holidays per region
    hdays_reg = np.zeros(n_weeks, dtype=int)
    for i, d in enumerate(dates):
        if region_name == 'Moscow':
            if d.month == 1 and d.day <= 14:
                hdays_reg[i] = np.random.randint(1, 4)
            elif d.month == 5 and d.day <= 14:
                hdays_reg[i] = np.random.randint(1, 3)
            elif d.month == 3 and 8 <= d.day <= 10:
                hdays_reg[i] = np.random.randint(1, 2)
        elif region_name == 'Paris':
            if d.month == 1 and d.day <= 2:
                hdays_reg[i] = np.random.randint(1, 2)
            elif d.month == 5 and d.day <= 2:
                hdays_reg[i] = np.random.randint(1, 2)
            elif d.month == 7 and 14 <= d.day <= 16:
                hdays_reg[i] = np.random.randint(1, 3)
        elif region_name == 'Warsaw':
            if d.month == 1 and d.day <= 2:
                hdays_reg[i] = np.random.randint(1, 2)
            elif d.month == 5 and 1 <= d.day <= 5:
                hdays_reg[i] = np.random.randint(1, 3)
            elif d.month == 11 and 11 <= d.day <= 13:
                hdays_reg[i] = np.random.randint(1, 2)
        else:
            if d.month == 1 and d.day <= 2:
                hdays_reg[i] = np.random.randint(1, 2)
            elif d.month == 4 and 25 <= d.day <= 27:
                hdays_reg[i] = np.random.randint(1, 2)
            elif d.month == 7 and 23 <= d.day <= 25:
                hdays_reg[i] = np.random.randint(1, 2)

    # Price per region
    price_noise = np.random.randn(n_weeks) * 5
    price_val = np.clip(price_base + 40 * np.sin(2 * np.pi * t_all / 52 + ssn_phase) + price_noise, 80, 500)

    market_price = 250 + 40 * np.sin(2 * np.pi * t_all / 52) + 10 * np.random.randn(n_weeks)
    market_price = pd.Series(market_price).rolling(13, min_periods=1, center=True).mean().values
    price_rel_market = (price_val - market_price) / market_price

    p_change = np.diff(price_val, prepend=price_val[0])
    price_volat = pd.Series(np.abs(p_change / price_val)).rolling(13, min_periods=1, center=True).mean().values
    price_volat_rel = (price_volat - price_volat.mean()) / price_volat.std()
    price_volat_rel = np.clip(price_volat_rel, -1.5, 1.5)

    # Region media (independent per region)
    reg_tv_ots = 20000 + np.random.randn(n_weeks) * 3000 + campaign_spikes(n_weeks, n_max=1, prob_on=0.12) * 400000
    reg_tv_ots = np.clip(reg_tv_ots * mmlt, 0, 6e8).astype(int)

    reg_dgt_perf = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 15000 * mmlt + np.random.randn(n_weeks) * 800
    reg_dgt_perf = np.clip(reg_dgt_perf, 0, 20000).astype(int)

    reg_dgt_disp = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 4000000 * mmlt + np.random.randn(n_weeks) * 100000
    reg_dgt_disp = np.clip(reg_dgt_disp, 0, 5000000).astype(int)

    reg_dgt_vid = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 4000000 * mmlt + np.random.randn(n_weeks) * 100000
    reg_dgt_vid = np.clip(reg_dgt_vid, 0, 5000000).astype(int)

    reg_dgt_spec = campaign_spikes(n_weeks, n_max=1, prob_on=0.12) * 8000000 * mmlt + np.random.randn(n_weeks) * 300000
    reg_dgt_spec = np.clip(reg_dgt_spec, 0, 12000000).astype(int)

    reg_dgt_indoor = campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 600000 * mmlt + np.random.randn(n_weeks) * 30000
    reg_dgt_indoor = np.clip(reg_dgt_indoor, 0, 800000).astype(int)

    reg_ooh_ots = campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 5000 * mmlt + np.random.randn(n_weeks) * 300
    reg_ooh_ots = np.clip(reg_ooh_ots, 0, 8000).astype(int)

    for i, d in enumerate(dates):
        rows.append({
            'superbrand': 'brand_1',
            'bs_key': bs,
            'master': region_name,
            'market': 'market_1',
            'date': d,
            'listed': 1,
            'is_inn': 1,
            'media': 1,
            'kpi_coef': 0.0,
            'di_soph_ssn_median_sm': round(di_raw_brand[i], 6),
            'di_soph_ssn_median': round(di_soph_ssn_raw_brand[i], 2),
            'units_base_share': round(np.clip(di_raw_brand[i] * np.random.uniform(0.4, 0.8), 0.01, 0.95), 6),
            'price_ret': round(price_val[i], 2),
            'price_index': round(np.random.uniform(0.85, 1.15), 4),
            'brnd_tv_ots_clip': int(tv_ots[i]),
            'brnd_cl_dgtind_perform': int(dgt_perf[i]),
            'brnd_cl_dgtind_display': int(dgt_disp[i]),
            'brnd_cl_dgtind_video': int(dgt_vid[i]),
            'brnd_cl_dgtind_special': int(dgt_spec[i]),
            'brnd_cl_dgtind_indoor': int(dgt_indoor[i]),
            'brnd_ots_ooh': int(ooh_ots[i]),
            'reg_tv_ots_clip': int(reg_tv_ots[i]),
            'reg_cl_dgtind_perform': int(reg_dgt_perf[i]),
            'reg_cl_dgtind_display': int(reg_dgt_disp[i]),
            'reg_cl_dgtind_video': int(reg_dgt_vid[i]),
            'reg_cl_dgtind_special': int(reg_dgt_spec[i]),
            'reg_cl_dgtind_indoor': int(reg_dgt_indoor[i]),
            'reg_ots_ooh': int(reg_ooh_ots[i]),
            'e_com': int(ecom[i]),
            'pos': int(pos_val[i]),
            'other': 0,
            'pr_costs_sm': int(pr_cost[i]),
            'tt_pk_tr_ssn': round(tt_pk_reg[i], 6),
            'hdays': int(hdays_reg[i]),
            'price_ret_rel_market': round(price_rel_market[i], 6),
            'price_ret_volat_rel': round(price_volat_rel[i], 6),
        })

dt_p = pd.DataFrame(rows)

# Compute kpi_coef: softmax over raw kpi per date
for idx, d in enumerate(dates):
    mask = dt_p['date'] == d
    n_reg = mask.sum()
    raw = np.random.randn(n_reg) * 0.3 + np.arange(n_reg) * 0.2
    kp = np.exp(raw) / np.sum(np.exp(raw))
    dt_p.loc[mask, 'kpi_coef'] = kp

# ---------- listed split ----------
dt_p.loc[dt_p['date'] > '2019-12-30', 'listed'] = 2
dt_p.loc[dt_p['date'] > '2024-12-25', 'listed'] = 3
dt_p.loc[dt_p['date'] > '2025-06-24', 'listed'] = 4

dt_p = dt_p.sort_values(['master', 'date']).reset_index(drop=True)

# ---------- compute di_soph_ssn_median_sm_share ----------
# Share = di * kpi_coef / sum(di * kpi_coef) within market+date+listed
# Since di is brand-consistent (same per date), this equals kpi_coef
dt_p['di_raw_weighted'] = dt_p['di_soph_ssn_median_sm'] * dt_p['kpi_coef']
di_total = dt_p.groupby(['market', 'date', 'listed'])['di_raw_weighted'].transform('sum')
dt_p['di_soph_ssn_median_sm_share'] = dt_p['di_raw_weighted'] / di_total
dt_p['di_soph_ssn_median_sm_share'] = dt_p['di_soph_ssn_median_sm_share'].fillna(0).clip(0, 1)

del dt_p['di_raw_weighted']

# ---------- save ----------
tmp_path = os.path.join(tempfile.gettempdir(), 'tmp_data_p04.xlsx')
out_path = os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'data_processed', 'data_for_sample_p04.xlsx')
dt_p.to_excel(tmp_path, index=False)
shutil.copy2(tmp_path, out_path)
os.remove(tmp_path)
print(f'Generated {len(dt_p)} rows')
print(f'Shape: {dt_p.shape}')
print(f'Columns: {list(dt_p.columns)}')
print(f'Regions: {dt_p["master"].unique()}')
print(f'Date range: {dt_p["date"].min()} to {dt_p["date"].max()}')
print(f'Listed: {dt_p["listed"].value_counts().to_dict()}')
print(f'kpi_coef sum per date (check): {dt_p.groupby("date")["kpi_coef"].sum().describe()}')
