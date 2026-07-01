import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

# ---------- config ----------
brands = ['brand_1', 'brand_2', 'brand_3']
brand_base_volume = {'brand_1': 3000, 'brand_2': 2000, 'brand_3': 1500}
brand_master = {'brand_1': 'master_a', 'brand_2': 'master_b', 'brand_3': 'master_c'}
brand_bs_key = {'brand_1': 1, 'brand_2': 2, 'brand_3': 3}

date_start = datetime(2019, 1, 7)
date_end = datetime(2030, 12, 31)
n_weeks = (date_end - date_start).days // 7 + 1
dates = [date_start + timedelta(weeks=i) for i in range(n_weeks)]

N = len(brands) * len(dates)


# ---------- brand-independent series (consistent across brands) ----------
t_all = np.arange(len(dates))

# Seasonality trough-to-peak (same for all brands on same date)
ssn_all = 0.5 + 0.4 * np.sin(2 * np.pi * t_all / 52)
tt_pk_all = 0.3 + 0.7 * ssn_all

tt_pk_all = tt_pk_all - np.min(tt_pk_all) + 1  # normalize to start at 1

# Holidays (same for all brands on same date)
hdays_all = np.zeros(len(dates), dtype=int)
for i, d in enumerate(dates):
    if d.month == 1 and d.day <= 14:
        hdays_all[i] = np.random.randint(1, 4)
    elif d.month == 5 and d.day <= 14:
        hdays_all[i] = np.random.randint(1, 3)
    else:
        hdays_all[i] = np.random.randint(0, 1)

# ---------- helpers ----------
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


# ---------- build rows ----------
rows = []

for brand in brands:
    bs = brand_bs_key[brand]

    t = t_all

    # distribution
    trend = 0.4 + 0.4 * t / n_weeks
    di_raw = np.clip(trend + 0.1 * np.random.randn(len(dates)), 0.05, 0.95) * 100
    di_soph_ssn_raw = di_raw

    # price
    price_base = {'brand_1': 200, 'brand_2': 300, 'brand_3': 250}[brand]
    price_noise = np.random.randn(len(dates)) * 5
    price_val = np.clip(price_base + 40 * np.sin(2 * np.pi * t / 52) + price_noise, 80, 500)

    # media: brand_1 gets highest, brand_2 moderate, brand_3 lowest
    if brand == 'brand_1':
        media_mult = 1.0
    elif brand == 'brand_2':
        media_mult = 0.6
    else:
        media_mult = 0.3

    # TV
    tv_ots = 50000 + np.random.randn(len(dates)) * 5000 + campaign_spikes(len(dates), n_max=1, prob_on=0.15) * 800000
    tv_ots = np.clip(tv_ots * media_mult, 0, 1.3e9).astype(int)

    # Digital performance
    dgt_perf = campaign_spikes(len(dates), n_max=1, prob_on=0.3) * 30000 * media_mult + np.random.randn(len(dates)) * 1000
    dgt_perf = np.clip(dgt_perf, 0, 40000).astype(int)

    # Digital display
    dgt_disp = campaign_spikes(len(dates), n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(len(dates)) * 200000
    dgt_disp = np.clip(dgt_disp, 0, 9700000).astype(int)

    # Digital video
    dgt_vid = campaign_spikes(len(dates), n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(len(dates)) * 200000
    dgt_vid = np.clip(dgt_vid, 0, 9700000).astype(int)

    # Digital special
    dgt_spec = campaign_spikes(len(dates), n_max=1, prob_on=0.15) * 15000000 * media_mult + np.random.randn(len(dates)) * 500000
    dgt_spec = np.clip(dgt_spec, 0, 22000000).astype(int)

    # Digital indoor (OOH digital)
    dgt_indoor = campaign_spikes(len(dates), n_max=1, prob_on=0.2) * 1200000 * media_mult + np.random.randn(len(dates)) * 50000
    dgt_indoor = np.clip(dgt_indoor, 0, 1600000).astype(int)

    # OOH
    ooh_ots = campaign_spikes(len(dates), n_max=1, prob_on=0.2) * 10000 * media_mult + np.random.randn(len(dates)) * 500
    ooh_ots = np.clip(ooh_ots, 0, 16000).astype(int)

    # E-com
    ecom = campaign_spikes(len(dates), n_max=1, prob_on=0.3) * 900000 * media_mult + np.random.randn(len(dates)) * 30000
    ecom = np.clip(ecom, 0, 1180000).astype(int)

    # POS
    pos_val = campaign_spikes(len(dates), n_max=1, prob_on=0.25) * 4000000 * media_mult + np.random.randn(len(dates)) * 100000
    pos_val = np.clip(pos_val, 0, 5000000).astype(int)

    # Other
    other_val = campaign_spikes(len(dates), n_max=1, prob_on=0.2) * 4000000 * media_mult + np.random.randn(len(dates)) * 100000
    other_val = np.clip(other_val, 0, 5000000).astype(int)

    # PR costs social media
    pr_cost = campaign_spikes(len(dates), n_max=1, prob_on=0.15) * 2000000 * media_mult + np.random.randn(len(dates)) * 50000
    pr_cost = np.clip(pr_cost, 0, 3000000).astype(int)

    # Price relative to market
    market_price = 250 + 40 * np.sin(2 * np.pi * t / 52) + 10 * np.random.randn(len(dates))
    market_price = pd.Series(market_price).rolling(13, min_periods=1, center=True).mean().values
    price_rel_market = (price_val - market_price) / market_price

    # Price volatility
    p_change = np.diff(price_val, prepend=price_val[0])
    price_volat = pd.Series(np.abs(p_change / price_val)).rolling(13, min_periods=1, center=True).mean().values
    price_volat_rel = (price_volat - price_volat.mean()) / price_volat.std()
    price_volat_rel = np.clip(price_volat_rel, -1.5, 1.5)

    # kpi_coef: weight for adstock aggregation by superbrand
    kpi_coef_val = {'brand_1': 1, 'brand_2': 1, 'brand_3': 1}[brand]

    for i, d in enumerate(dates):
        rows.append({
            'superbrand': brand,
            'bs_key': bs,
            'master': brand_master[brand],
            'market': 'market_1',
            'date': d,
            'listed': 1,
            'is_inn': 1,
            'media': 1,
            'kpi_coef': kpi_coef_val,
            'di_soph_ssn_median_sm': round(di_raw[i], 6),
            'di_soph_ssn_median': round(di_soph_ssn_raw[i], 2),
            'units_base_share': round(np.clip(di_raw[i] * np.random.uniform(0.4, 0.8), 0.01, 0.95), 6),
            'price_ret': round(price_val[i], 2),
            'brnd_tv_ots_clip': int(tv_ots[i]),
            'brnd_cl_dgtind_perform': int(dgt_perf[i]),
            'brnd_cl_dgtind_display': int(dgt_disp[i]),
            'brnd_cl_dgtind_video': int(dgt_vid[i]),
            'brnd_cl_dgtind_special': int(dgt_spec[i]),
            'brnd_cl_dgtind_indoor': int(dgt_indoor[i]),
            'brnd_ots_ooh': int(ooh_ots[i]),
            'e_com': int(ecom[i]),
            'pos': int(pos_val[i]),
            'other': 0,
            'pr_costs_sm': int(pr_cost[i]),
            'tt_pk_tr_ssn': round(tt_pk_all[i], 6),
            'hdays': int(hdays_all[i]),
            'price_ret_rel_market': round(price_rel_market[i], 6),
            'price_ret_volat_rel': round(price_volat_rel[i], 6),
        })


dt_p = pd.DataFrame(rows)

# ---------- listed split ----------
dt_p.loc[dt_p['date'] > '2019-12-30', 'listed'] = 2
dt_p.loc[dt_p['date'] > '2024-12-25', 'listed'] = 3
dt_p.loc[dt_p['date'] > '2025-06-24', 'listed'] = 4

dt_p = dt_p.sort_values(['superbrand', 'date']).reset_index(drop=True)

# ---------- save ----------
tmp_path = os.path.join(tempfile.gettempdir(), 'tmp_data_p01.xlsx')
out_path = os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'data_processed', 'data_for_sample_p01.xlsx')
dt_p.to_excel(tmp_path, index=False)
shutil.copy2(tmp_path, out_path)
os.remove(tmp_path)
print(f'Generated {len(dt_p)} rows')
print(f'Shape: {dt_p.shape}')
print(f'Columns: {list(dt_p.columns)}')
print(f'Brands: {dt_p["superbrand"].unique()}')
print(f'Date range: {dt_p["date"].min()} to {dt_p["date"].max()}')
print(f'Listed: {dt_p["listed"].value_counts().to_dict()}')
