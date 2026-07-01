import hashlib
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _dhash(s):
    """Deterministic string hash — stable across Python processes."""
    return int(hashlib.md5(str(s).encode()).digest().hex(), 16)


np.random.seed(42)

# ---------- config ----------
brands = ['brand_1', 'brand_2', 'brand_3']
brand_base_volume = {'brand_1': 3000, 'brand_2': 2000, 'brand_3': 1500}

# Each brand has multiple masters/forms forming unique SKUs
# Master names are prefixed with brand to keep them unique per brand
brand_sku_map = {
    'brand_1': {
        'brand_1_master_a': ['form_a', 'form_b', 'form_c'],
        'brand_1_master_b': ['form_a', 'form_b'],
        'brand_1_master_c': ['form_a'],
    },
    'brand_2': {
        'brand_2_master_a': ['form_a', 'form_b'],
        'brand_2_master_b': ['form_a', 'form_b', 'form_c'],
        'brand_2_master_c': ['form_a'],
        'brand_2_master_d': ['form_a', 'form_b', 'form_c'],
    },
    'brand_3': {
        'brand_3_master_a': ['form_a', 'form_b'],
        'brand_3_master_b': ['form_a', 'form_b', 'form_c', 'form_d'],
    },
}

date_start = datetime(2019, 1, 7)
date_end = datetime(2030, 12, 31)
n_weeks = (date_end - date_start).days // 7 + 1
dates = [date_start + timedelta(weeks=i) for i in range(n_weeks)]

# Build SKU list: (brand, master, form, bs_key)
sku_list = []
bs_key = 0
sku_count_per_brand = {}
for b in brands:
    sku_count_per_brand[b] = 0
    for m, forms in brand_sku_map[b].items():
        for f in forms:
            bs_key += 1
            sku_list.append((b, m, f, bs_key))
            sku_count_per_brand[b] += 1

sku_df = pd.DataFrame(sku_list, columns=['superbrand', 'master', 'form', 'bs_key'])
N_skus = len(sku_list)
print(f'Total SKUs: {N_skus}')
print(f'SKUs per brand: {sku_count_per_brand}')
print(f'Total rows: {N_skus * n_weeks}')

# ---------- brand-independent series (consistent across brands) ----------
t_all = np.arange(len(dates))

ssn_all = 0.5 + 0.4 * np.sin(2 * np.pi * t_all / 52)
tt_pk_all = 0.3 + 0.7 * ssn_all
tt_pk_all = tt_pk_all - np.min(tt_pk_all) + 1

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

# ---------- build per-brand time series (media, DI, brand-level price) ----------
# These are generated once per brand and broadcast to all SKUs

brand_ts = {}  # brand -> dict of arrays

for brand in brands:
    t = t_all
    n = len(dates)

    trend = 0.4 + 0.4 * t / n_weeks
    di_raw = np.clip(trend + 0.1 * np.random.randn(n), 0.05, 0.95) * 100
    di_soph_ssn_raw = di_raw

    price_base = {'brand_1': 200, 'brand_2': 300, 'brand_3': 250}[brand]
    price_noise = np.random.randn(n) * 5
    price_val = np.clip(price_base + 40 * np.sin(2 * np.pi * t / 52) + price_noise, 80, 500)

    if brand == 'brand_1':
        media_mult = 1.0
    elif brand == 'brand_2':
        media_mult = 0.6
    else:
        media_mult = 0.3

    tv_ots = 50000 + np.random.randn(n) * 5000 + campaign_spikes(n, n_max=1, prob_on=0.15) * 800000
    tv_ots = np.clip(tv_ots * media_mult, 0, 1.3e9).astype(int)

    dgt_perf = campaign_spikes(n, n_max=1, prob_on=0.3) * 30000 * media_mult + np.random.randn(n) * 1000
    dgt_perf = np.clip(dgt_perf, 0, 40000).astype(int)

    dgt_disp = campaign_spikes(n, n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(n) * 200000
    dgt_disp = np.clip(dgt_disp, 0, 9700000).astype(int)

    dgt_vid = campaign_spikes(n, n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(n) * 200000
    dgt_vid = np.clip(dgt_vid, 0, 9700000).astype(int)

    dgt_spec = campaign_spikes(n, n_max=1, prob_on=0.15) * 15000000 * media_mult + np.random.randn(n) * 500000
    dgt_spec = np.clip(dgt_spec, 0, 22000000).astype(int)

    dgt_indoor = campaign_spikes(n, n_max=1, prob_on=0.2) * 1200000 * media_mult + np.random.randn(n) * 50000
    dgt_indoor = np.clip(dgt_indoor, 0, 1600000).astype(int)

    ooh_ots = campaign_spikes(n, n_max=1, prob_on=0.2) * 10000 * media_mult + np.random.randn(n) * 500
    ooh_ots = np.clip(ooh_ots, 0, 16000).astype(int)

    ecom = campaign_spikes(n, n_max=1, prob_on=0.3) * 900000 * media_mult + np.random.randn(n) * 30000
    ecom = np.clip(ecom, 0, 1180000).astype(int)

    pos_val = campaign_spikes(n, n_max=1, prob_on=0.25) * 4000000 * media_mult + np.random.randn(n) * 100000
    pos_val = np.clip(pos_val, 0, 5000000).astype(int)

    other_val = campaign_spikes(n, n_max=1, prob_on=0.2) * 4000000 * media_mult + np.random.randn(n) * 100000
    other_val = np.clip(other_val, 0, 5000000).astype(int)

    pr_cost = campaign_spikes(n, n_max=1, prob_on=0.15) * 2000000 * media_mult + np.random.randn(n) * 50000
    pr_cost = np.clip(pr_cost, 0, 3000000).astype(int)

    market_price = 250 + 40 * np.sin(2 * np.pi * t / 52) + 10 * np.random.randn(n)
    market_price = pd.Series(market_price).rolling(13, min_periods=1, center=True).mean().values
    price_rel_market = (price_val - market_price) / market_price

    p_change = np.diff(price_val, prepend=price_val[0])
    price_volat = pd.Series(np.abs(p_change / price_val)).rolling(13, min_periods=1, center=True).mean().values
    price_volat_rel = (price_volat - price_volat.mean()) / price_volat.std()
    price_volat_rel = np.clip(price_volat_rel, -1.5, 1.5)

    brand_ts[brand] = {
        'di_raw': di_raw,
        'di_soph_ssn_raw': di_soph_ssn_raw,
        'price_val': price_val,
        'tv_ots': tv_ots,
        'dgt_perf': dgt_perf,
        'dgt_disp': dgt_disp,
        'dgt_vid': dgt_vid,
        'dgt_spec': dgt_spec,
        'dgt_indoor': dgt_indoor,
        'ooh_ots': ooh_ots,
        'ecom': ecom,
        'pos_val': pos_val,
        'other_val': other_val,
        'pr_cost': pr_cost,
        'price_rel_market': price_rel_market,
        'price_volat_rel': price_volat_rel,
    }

# ---------- build rows ----------
rows = []

# Pre-compute brand-level price_rel_market per week for DI share computation
for brand in brands:
    bt = brand_ts[brand]
    mkt = 'market_1'

    skus = sku_df[sku_df['superbrand'] == brand]
    n_sku = len(skus)

    # kpi_coef: brand base volume split evenly across SKUs so aggregation is correct
    kpi_coef_val = brand_base_volume[brand] / brand_base_volume[brand] / n_sku

    for _, sku in skus.iterrows():
        master = sku['master']
        form = sku['form']
        bs_key = sku['bs_key']

        for i, d in enumerate(dates):
            # Price per SKU: brand-level price + small form-level noise
            form_noise = _dhash(f"{brand}_{master}_{form}_{i}") % 21 - 10
            price_sku = bt['price_val'][i] + form_noise
            price_sku = np.clip(price_sku, 80, 500)

            rows.append({
                'superbrand': brand,
                'bs_key': int(bs_key),
                'master': master,
                'form': form,
                'market': 'market_1',
                'date': d,
                'listed': 1,
                'is_inn': 1,
                'media': 1,
                'kpi_coef': round(kpi_coef_val, 6),
                'di_soph_ssn_median_sm': round(bt['di_raw'][i], 6),
                'di_soph_ssn_median': round(bt['di_soph_ssn_raw'][i], 2),
                'units_base_share': round(np.clip(bt['di_raw'][i] * np.random.uniform(0.4, 0.8), 0.01, 0.95), 6),
                'price_ret': round(price_sku, 2),
                'brnd_tv_ots_clip': int(bt['tv_ots'][i]),
                'brnd_cl_dgtind_perform': int(bt['dgt_perf'][i]),
                'brnd_cl_dgtind_display': int(bt['dgt_disp'][i]),
                'brnd_cl_dgtind_video': int(bt['dgt_vid'][i]),
                'brnd_cl_dgtind_special': int(bt['dgt_spec'][i]),
                'brnd_cl_dgtind_indoor': int(bt['dgt_indoor'][i]),
                'brnd_ots_ooh': int(bt['ooh_ots'][i]),
                'e_com': int(bt['ecom'][i]),
                'pos': int(bt['pos_val'][i]),
                'other': 0,
                'pr_costs_sm': int(bt['pr_cost'][i]),
                'tt_pk_tr_ssn': round(tt_pk_all[i], 6),
                'hdays': int(hdays_all[i]),
                'price_ret_rel_market': round(bt['price_rel_market'][i], 6),
                'price_ret_volat_rel': round(bt['price_volat_rel'][i], 6),
            })

dt_p = pd.DataFrame(rows)

# ---------- listed split ----------
dt_p.loc[dt_p['date'] > '2019-12-30', 'listed'] = 2
dt_p.loc[dt_p['date'] > '2024-12-25', 'listed'] = 3
dt_p.loc[dt_p['date'] > '2025-06-24', 'listed'] = 4

dt_p = dt_p.sort_values(['superbrand', 'master', 'form', 'date']).reset_index(drop=True)

# ---------- compute di_soph_ssn_median_sm_share ----------
# Share = brand's DI / sum of all brands' DI (weighted by kpi_coef)
# Since kpi_coef splits brand volume across SKUs, aggregating kpi_coef * di gives brand-level DI
# Share within market+date+listed
dt_p['di_raw_weighted'] = dt_p['di_soph_ssn_median_sm'] * dt_p['kpi_coef']
di_total = dt_p.groupby(['market', 'date', 'listed'])['di_raw_weighted'].transform('sum')
dt_p['di_soph_ssn_median_sm_share'] = dt_p['di_raw_weighted'] / di_total
dt_p['di_soph_ssn_median_sm_share'] = dt_p['di_soph_ssn_median_sm_share'].fillna(0).clip(0, 1)

del dt_p['di_raw_weighted']

# ---------- save ----------
tmp_path = os.path.join(tempfile.gettempdir(), 'tmp_data_p02.xlsx')
out_path = os.path.join(_SCRIPT_DIR, '..', '..', 'data', 'data_processed', 'data_for_sample_p02.xlsx')
dt_p.to_excel(tmp_path, index=False)
shutil.copy2(tmp_path, out_path)
if os.path.exists(tmp_path):
    os.remove(tmp_path)

print(f'Generated {len(dt_p)} rows')
print(f'Shape: {dt_p.shape}')
print(f'Columns: {list(dt_p.columns)}')
print(f'SKUs: {dt_p["bs_key"].nunique()}')
print(f'Brands: {dt_p["superbrand"].unique()}')
print(f'Masters: {dt_p["master"].unique()}')
print(f'Forms: {dt_p["form"].unique()}')
print(f'Date range: {dt_p["date"].min()} to {dt_p["date"].max()}')
print(f'Listed: {dt_p["listed"].value_counts().to_dict()}')
print(f'DI share range: {dt_p["di_soph_ssn_median_sm_share"].min():.6f} to {dt_p["di_soph_ssn_median_sm_share"].max():.6f}')
