import hashlib
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p05_root = os.path.join(_SCRIPT_DIR, '..', '..')


def _dhash(s):
    """Deterministic string hash — stable across Python processes."""
    return int(hashlib.md5(str(s).encode()).digest().hex(), 16)


np.random.seed(42)

# ---------- config ----------
brands = ['brand_1', 'brand_2', 'brand_3']
brand_base_aba = {'brand_1': 65, 'brand_2': 45, 'brand_3': 35}
brand_base_volume = {'brand_1': 3000, 'brand_2': 2000, 'brand_3': 1500}

# Same SKU map as p02
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

# ---- Date ranges ----
# Monthly for BHT
date_start_m = datetime(2019, 1, 1)
date_end_m = datetime(2030, 12, 1)
dates_m = pd.date_range(start=date_start_m, end=date_end_m, freq='MS').to_pydatetime().tolist()
n_months = len(dates_m)

# Weekly for Sales
date_start_w = datetime(2019, 1, 7)
date_end_w = datetime(2030, 12, 31)
n_weeks = (date_end_w - date_start_w).days // 7 + 1
dates_w = [date_start_w + timedelta(weeks=i) for i in range(n_weeks)]

# ---- Build SKU list ----
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

# ---- Brand-independent time series ----
t_m = np.arange(n_months)
t_w = np.arange(n_weeks)

# Monthly seasonality
ssn_m = 0.5 + 0.4 * np.sin(2 * np.pi * t_m / 12)
tt_pk_m = 0.3 + 0.7 * ssn_m
tt_pk_m = tt_pk_m - np.min(tt_pk_m) + 1

# Weekly seasonality
ssn_w = 0.5 + 0.4 * np.sin(2 * np.pi * t_w / 52)
tt_pk_w = 0.3 + 0.7 * ssn_w
tt_pk_w = tt_pk_w - np.min(tt_pk_w) + 1

# Holidays (monthly)
hdays_m = np.zeros(n_months, dtype=int)
for i, d in enumerate(dates_m):
    if d.month == 1 and d.day <= 14:
        hdays_m[i] = np.random.randint(1, 4)
    elif d.month == 5 and d.day <= 14:
        hdays_m[i] = np.random.randint(1, 3)
    else:
        hdays_m[i] = np.random.randint(0, 1)

# Holidays (weekly)
hdays_w = np.zeros(n_weeks, dtype=int)
for i, d in enumerate(dates_w):
    if d.month == 1 and d.day <= 14:
        hdays_w[i] = np.random.randint(1, 4)
    elif d.month == 5 and d.day <= 14:
        hdays_w[i] = np.random.randint(1, 3)
    else:
        hdays_w[i] = np.random.randint(0, 1)


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


# ---- Generate brand-level time series (shared) ----
brand_ts_m = {}
brand_ts_w = {}

for brand in brands:
    media_mult = {'brand_1': 1.0, 'brand_2': 0.6, 'brand_3': 0.3}[brand]

    # Monthly series
    trend_m = 0.4 + 0.4 * t_m / n_months
    di_raw_m = np.clip(trend_m + 0.1 * np.random.randn(n_months), 0.05, 0.95) * 100

    price_base = {'brand_1': 200, 'brand_2': 300, 'brand_3': 250}[brand]
    price_noise_m = np.random.randn(n_months) * 5
    price_val_m = np.clip(price_base + 40 * np.sin(2 * np.pi * t_m / 12) + price_noise_m, 80, 500)

    tv_ots_m = 50000 + np.random.randn(n_months) * 5000 + campaign_spikes(n_months, n_max=1, prob_on=0.15) * 800000
    tv_ots_m = np.clip(tv_ots_m * media_mult, 0, 1.3e9).astype(int)

    dgt_perf_m = campaign_spikes(n_months, n_max=1, prob_on=0.3) * 30000 * media_mult + np.random.randn(n_months) * 1000
    dgt_perf_m = np.clip(dgt_perf_m, 0, 40000).astype(int)

    dgt_disp_m = campaign_spikes(n_months, n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(n_months) * 200000
    dgt_disp_m = np.clip(dgt_disp_m, 0, 9700000).astype(int)

    dgt_vid_m = campaign_spikes(n_months, n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(n_months) * 200000
    dgt_vid_m = np.clip(dgt_vid_m, 0, 9700000).astype(int)

    dgt_spec_m = campaign_spikes(n_months, n_max=1, prob_on=0.15) * 15000000 * media_mult + np.random.randn(n_months) * 500000
    dgt_spec_m = np.clip(dgt_spec_m, 0, 22000000).astype(int)

    dgt_indoor_m = campaign_spikes(n_months, n_max=1, prob_on=0.2) * 1200000 * media_mult + np.random.randn(n_months) * 50000
    dgt_indoor_m = np.clip(dgt_indoor_m, 0, 1600000).astype(int)

    ooh_ots_m = campaign_spikes(n_months, n_max=1, prob_on=0.2) * 10000 * media_mult + np.random.randn(n_months) * 500
    ooh_ots_m = np.clip(ooh_ots_m, 0, 16000).astype(int)

    ecom_m = campaign_spikes(n_months, n_max=1, prob_on=0.3) * 900000 * media_mult + np.random.randn(n_months) * 30000
    ecom_m = np.clip(ecom_m, 0, 1180000).astype(int)

    pos_val_m = campaign_spikes(n_months, n_max=1, prob_on=0.25) * 4000000 * media_mult + np.random.randn(n_months) * 100000
    pos_val_m = np.clip(pos_val_m, 0, 5000000).astype(int)

    other_val_m = campaign_spikes(n_months, n_max=1, prob_on=0.2) * 4000000 * media_mult + np.random.randn(n_months) * 100000
    other_val_m = np.clip(other_val_m, 0, 5000000).astype(int)

    pr_cost_m = campaign_spikes(n_months, n_max=1, prob_on=0.15) * 2000000 * media_mult + np.random.randn(n_months) * 50000
    pr_cost_m = np.clip(pr_cost_m, 0, 3000000).astype(int)

    market_price_m = 250 + 40 * np.sin(2 * np.pi * t_m / 12) + 10 * np.random.randn(n_months)
    market_price_m = pd.Series(market_price_m).rolling(3, min_periods=1, center=True).mean().values
    price_rel_market_m = (price_val_m - market_price_m) / market_price_m

    p_change_m = np.diff(price_val_m, prepend=price_val_m[0])
    price_volat_m = pd.Series(np.abs(p_change_m / price_val_m)).rolling(3, min_periods=1, center=True).mean().values
    price_volat_rel_m = (price_volat_m - price_volat_m.mean()) / price_volat_m.std()
    price_volat_rel_m = np.clip(price_volat_rel_m, -1.5, 1.5)

    brand_ts_m[brand] = {
        'di_raw': di_raw_m,
        'price_val': price_val_m,
        'tv_ots': tv_ots_m,
        'dgt_perf': dgt_perf_m,
        'dgt_disp': dgt_disp_m,
        'dgt_vid': dgt_vid_m,
        'dgt_spec': dgt_spec_m,
        'dgt_indoor': dgt_indoor_m,
        'ooh_ots': ooh_ots_m,
        'ecom': ecom_m,
        'pos_val': pos_val_m,
        'other_val': other_val_m,
        'pr_cost': pr_cost_m,
        'price_rel_market': price_rel_market_m,
        'price_volat_rel': price_volat_rel_m,
    }

    # Weekly series (interpolate monthly media to weekly, then add noise)
    def monthly_to_weekly(monthly_arr, n_weeks_val):
        idx_m = np.linspace(0, n_months - 1, n_months)
        idx_w = np.linspace(0, n_months - 1, n_weeks_val)
        return np.round(np.interp(idx_w, idx_m, monthly_arr)).astype(int)

    def monthly_to_weekly_float(monthly_arr, n_weeks_val):
        idx_m = np.linspace(0, n_months - 1, n_months)
        idx_w = np.linspace(0, n_months - 1, n_weeks_val)
        return np.interp(idx_w, idx_m, monthly_arr)

    trend_w = 0.4 + 0.4 * t_w / n_weeks
    di_raw_w = np.clip(trend_w + 0.1 * np.random.randn(n_weeks), 0.05, 0.95) * 100

    price_noise_w = np.random.randn(n_weeks) * 5
    price_val_w = np.clip(price_base + 40 * np.sin(2 * np.pi * t_w / 52) + price_noise_w, 80, 500)

    tv_ots_w = 50000 + np.random.randn(n_weeks) * 5000 + campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 800000
    tv_ots_w = np.clip(tv_ots_w * media_mult, 0, 1.3e9).astype(int)

    dgt_perf_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.3) * 30000 * media_mult + np.random.randn(n_weeks) * 1000
    dgt_perf_w = np.clip(dgt_perf_w, 0, 40000).astype(int)

    dgt_disp_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(n_weeks) * 200000
    dgt_disp_w = np.clip(dgt_disp_w, 0, 9700000).astype(int)

    dgt_vid_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 8000000 * media_mult + np.random.randn(n_weeks) * 200000
    dgt_vid_w = np.clip(dgt_vid_w, 0, 9700000).astype(int)

    dgt_spec_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 15000000 * media_mult + np.random.randn(n_weeks) * 500000
    dgt_spec_w = np.clip(dgt_spec_w, 0, 22000000).astype(int)

    dgt_indoor_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 1200000 * media_mult + np.random.randn(n_weeks) * 50000
    dgt_indoor_w = np.clip(dgt_indoor_w, 0, 1600000).astype(int)

    ooh_ots_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 10000 * media_mult + np.random.randn(n_weeks) * 500
    ooh_ots_w = np.clip(ooh_ots_w, 0, 16000).astype(int)

    ecom_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.3) * 900000 * media_mult + np.random.randn(n_weeks) * 30000
    ecom_w = np.clip(ecom_w, 0, 1180000).astype(int)

    pos_val_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.25) * 4000000 * media_mult + np.random.randn(n_weeks) * 100000
    pos_val_w = np.clip(pos_val_w, 0, 5000000).astype(int)

    other_val_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.2) * 4000000 * media_mult + np.random.randn(n_weeks) * 100000
    other_val_w = np.clip(other_val_w, 0, 5000000).astype(int)

    pr_cost_w = campaign_spikes(n_weeks, n_max=1, prob_on=0.15) * 2000000 * media_mult + np.random.randn(n_weeks) * 50000
    pr_cost_w = np.clip(pr_cost_w, 0, 3000000).astype(int)

    market_price_w = 250 + 40 * np.sin(2 * np.pi * t_w / 52) + 10 * np.random.randn(n_weeks)
    market_price_w = pd.Series(market_price_w).rolling(13, min_periods=1, center=True).mean().values
    price_rel_market_w = (price_val_w - market_price_w) / market_price_w

    p_change_w = np.diff(price_val_w, prepend=price_val_w[0])
    price_volat_w = pd.Series(np.abs(p_change_w / price_val_w)).rolling(13, min_periods=1, center=True).mean().values
    price_volat_rel_w = (price_volat_w - price_volat_w.mean()) / price_volat_w.std()
    price_volat_rel_w = np.clip(price_volat_rel_w, -1.5, 1.5)

    brand_ts_w[brand] = {
        'di_raw': di_raw_w,
        'price_val': price_val_w,
        'tv_ots': tv_ots_w,
        'dgt_perf': dgt_perf_w,
        'dgt_disp': dgt_disp_w,
        'dgt_vid': dgt_vid_w,
        'dgt_spec': dgt_spec_w,
        'dgt_indoor': dgt_indoor_w,
        'ooh_ots': ooh_ots_w,
        'ecom': ecom_w,
        'pos_val': pos_val_w,
        'other_val': other_val_w,
        'pr_cost': pr_cost_w,
        'price_rel_market': price_rel_market_w,
        'price_volat_rel': price_volat_rel_w,
    }


# ============================================================
# Build BHT dataset (brand-month)
# ============================================================
rows_bht = []

for brand in brands:
    bt = brand_ts_m[brand]
    bs = brands.index(brand) + 1

    kpi_coef_val = 1

    for i, d in enumerate(dates_m):
        rows_bht.append({
            'superbrand': brand,
            'bs_key': bs,
            'master': 'master_' + brand,
            'market': 'market_1',
            'date': d,
            'listed': 1,
            'is_inn': 1,
            'media': 1,
            'kpi_coef': kpi_coef_val,
            'di_soph_ssn_median_sm': round(bt['di_raw'][i], 6),
            'di_soph_ssn_median': round(bt['di_raw'][i], 2),
            'units_base_share': round(np.clip(bt['di_raw'][i] * np.random.uniform(0.4, 0.8), 0.01, 0.95), 6),
            'price_ret': round(bt['price_val'][i], 2),
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
            'tt_pk_tr_ssn': round(tt_pk_m[i], 6),
            'hdays': int(hdays_m[i]),
            'price_ret_rel_market': round(bt['price_rel_market'][i], 6),
            'price_ret_volat_rel': round(bt['price_volat_rel'][i], 6),
        })

dt_bht = pd.DataFrame(rows_bht)

# listed split for BHT
dt_bht.loc[dt_bht['date'] > '2019-12-30', 'listed'] = 2
dt_bht.loc[dt_bht['date'] > '2024-12-25', 'listed'] = 3
dt_bht.loc[dt_bht['date'] > '2025-06-24', 'listed'] = 4

dt_bht = dt_bht.sort_values(['superbrand', 'date']).reset_index(drop=True)

# Initial ABA (will be overwritten later)
base_vec = np.array([brand_base_aba[b] for b in dt_bht['superbrand']])
media_contrib = (
    0.15 * np.log1p(dt_bht['brnd_tv_ots_clip'].values / 5e7) +
    0.08 * np.log1p(dt_bht['brnd_cl_dgtind_perform'].values / 1e4) +
    0.06 * np.log1p(dt_bht['brnd_cl_dgtind_display'].values / 3e6) +
    0.06 * np.log1p(dt_bht['brnd_cl_dgtind_video'].values / 3e6) +
    0.06 * np.log1p(dt_bht['brnd_cl_dgtind_special'].values / 5e6) +
    0.04 * np.log1p(dt_bht['brnd_cl_dgtind_indoor'].values / 5e5) +
    0.08 * np.log1p(dt_bht['brnd_ots_ooh'].values / 3e3) +
    0.03 * np.log1p(dt_bht['e_com'].values / 3e5) +
    0.03 * np.log1p(dt_bht['pr_costs_sm'].values / 5e5) +
    0.03 * np.log1p(dt_bht['pos'].values / 1e6)
)
media_contrib = np.clip(media_contrib, -2, 5)
aba_init = np.clip(base_vec + media_contrib * 25 + np.random.randn(len(dt_bht)) * 3, 0.5, 99.5)
dt_bht['aba'] = np.round(aba_init, 2)

# aba_lag
dt_bht['aba_lag'] = dt_bht.groupby('superbrand')['aba'].shift(1)
dt_bht['aba_lag'] = dt_bht.groupby('superbrand')['aba_lag'].bfill()

print(f'BHT dataset: {dt_bht.shape}, brands={dt_bht["superbrand"].nunique()}, months={dt_bht["date"].nunique()}')

# ============================================================
# Build Sales dataset (SKU-week)
# ============================================================
rows_sales = []

di_share_df = pd.DataFrame()

for brand in brands:
    bt_w = brand_ts_w[brand]
    skus = sku_df[sku_df['superbrand'] == brand]
    n_sku = len(skus)

    kpi_coef_val = brand_base_volume[brand] / brand_base_volume[brand] / n_sku

    # Pre-compute ABA for each week from monthly BHT values (interpolated)
    ibm_brand = dt_bht[dt_bht['superbrand'] == brand]['aba'].values
    idx_m = np.linspace(0, n_months - 1, n_months)
    idx_w = np.linspace(0, n_months - 1, n_weeks)
    aba_interp = np.interp(idx_w, idx_m, ibm_brand)

    for _, sku in skus.iterrows():
        master = sku['master']
        form = sku['form']
        bs_key_val = sku['bs_key']

        for i, d in enumerate(dates_w):
            form_noise = _dhash(f"{brand}_{master}_{form}_{i}") % 21 - 10
            price_sku = bt_w['price_val'][i] + form_noise
            price_sku = np.clip(price_sku, 80, 500)

            rows_sales.append({
                'superbrand': brand,
                'bs_key': int(bs_key_val),
                'master': master,
                'form': form,
                'market': 'market_1',
                'date': d,
                'listed': 1,
                'is_inn': 1,
                'media': 1,
                'kpi_coef': round(kpi_coef_val, 6),
                'di_soph_ssn_median_sm': round(bt_w['di_raw'][i], 6),
                'di_soph_ssn_median': round(bt_w['di_raw'][i], 2),
                'units_base_share': round(np.clip(bt_w['di_raw'][i] * np.random.uniform(0.4, 0.8), 0.01, 0.95), 6),
                'price_ret': round(price_sku, 2),
                'brnd_tv_ots_clip': int(bt_w['tv_ots'][i]),
                'brnd_cl_dgtind_perform': int(bt_w['dgt_perf'][i]),
                'brnd_cl_dgtind_display': int(bt_w['dgt_disp'][i]),
                'brnd_cl_dgtind_video': int(bt_w['dgt_vid'][i]),
                'brnd_cl_dgtind_special': int(bt_w['dgt_spec'][i]),
                'brnd_cl_dgtind_indoor': int(bt_w['dgt_indoor'][i]),
                'brnd_ots_ooh': int(bt_w['ooh_ots'][i]),
                'e_com': int(bt_w['ecom'][i]),
                'pos': int(bt_w['pos_val'][i]),
                'other': 0,
                'pr_costs_sm': int(bt_w['pr_cost'][i]),
                'tt_pk_tr_ssn': round(tt_pk_w[i], 6),
                'hdays': int(hdays_w[i]),
                'price_ret_rel_market': round(bt_w['price_rel_market'][i], 6),
                'price_ret_volat_rel': round(bt_w['price_volat_rel'][i], 6),
                'aba_predicted': round(aba_interp[i], 2),
            })

dt_sales = pd.DataFrame(rows_sales)

# listed split for Sales
dt_sales.loc[dt_sales['date'] > '2019-12-30', 'listed'] = 2
dt_sales.loc[dt_sales['date'] > '2024-12-25', 'listed'] = 3
dt_sales.loc[dt_sales['date'] > '2025-06-24', 'listed'] = 4

dt_sales = dt_sales.sort_values(['superbrand', 'master', 'form', 'date']).reset_index(drop=True)

# Compute DI share for sales
dt_sales['di_raw_weighted'] = dt_sales['di_soph_ssn_median_sm'] * dt_sales['kpi_coef']
di_total_s = dt_sales.groupby(['market', 'date', 'listed'])['di_raw_weighted'].transform('sum')
dt_sales['di_soph_ssn_median_sm_share'] = dt_sales['di_raw_weighted'] / di_total_s
dt_sales['di_soph_ssn_median_sm_share'] = dt_sales['di_soph_ssn_median_sm_share'].fillna(0).clip(0, 1)
del dt_sales['di_raw_weighted']

# Placeholder units (will be overwritten)
dt_sales['units'] = 0

print(f'Sales dataset: {dt_sales.shape}, SKUs={dt_sales["bs_key"].nunique()}, brands={dt_sales["superbrand"].nunique()}, weeks={dt_sales["date"].nunique()}')

# ============================================================
# Save both datasets
# ============================================================
tmp_bht = os.path.join(tempfile.gettempdir(), 'tmp_data_p05_bht.xlsx')
tmp_sales = os.path.join(tempfile.gettempdir(), 'tmp_data_p05_sales.xlsx')

out_bht = os.path.join(p05_root, 'data', 'data_processed', 'data_for_sample_p05_bht.xlsx')
out_sales = os.path.join(p05_root, 'data', 'data_processed', 'data_for_sample_p05_sales.xlsx')

dt_bht.to_excel(tmp_bht, index=False)
shutil.copy2(tmp_bht, out_bht)
os.remove(tmp_bht)

dt_sales.to_excel(tmp_sales, index=False)
shutil.copy2(tmp_sales, out_sales)
os.remove(tmp_sales)

print(f'\nGenerated BHT: {len(dt_bht)} rows, columns={len(dt_bht.columns)}')
print(f'Generated Sales: {len(dt_sales)} rows, columns={len(dt_sales.columns)}')
print(f'BHT columns: {list(dt_bht.columns)}')
print(f'Sales columns: {list(dt_sales.columns)}')
print(f'BHT ABA range: {dt_bht["aba"].min():.2f} to {dt_bht["aba"].max():.2f}')
