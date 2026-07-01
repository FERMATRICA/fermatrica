import copy
import os
import re
import time
import warnings
from datetime import datetime

import pandas as pd
import plotly.io as pio
import winsound

import fermatrica as fm
import fermatrica_rep as fmr
import fermatrica_rep.export.export_pptx as export_pptx

import code_py.adhoc.translators
from code_py.adhoc.reporting import pre_reporting

warnings.simplefilter(action='ignore', category=FutureWarning)

pio.renderers.default = "browser"

if __name__ == '__main__':

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 2000)

    if os.path.basename(os.getcwd()) == 'scripts':
        os.chdir("../../")
    elif os.path.basename(os.getcwd()) == 'code_py':
        os.chdir("../")

    if_visual = False
    if_curves = False
    if_export = True
    if_save = False
    if_debug = True

    # -------------------- load options per master ------------------------ #

    options_m_raw = pd.read_excel(os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                  sheet_name='main')

    options_m = {}
    for ms in options_m_raw['text_flags'].unique():
        if pd.isna(ms):
            continue
        options_m[ms] = fmr.budget_dict_create(
            dct=options_m_raw[options_m_raw['text_flags'] == ms])

    options_c_raw = pd.read_excel(os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                  sheet_name='current')

    options_c = {}
    for ms in options_c_raw['text_flags'].unique():
        if pd.isna(ms):
            continue
        options_c[ms] = fmr.budget_dict_create(
            dct=options_c_raw[options_c_raw['text_flags'] == ms])

    # media prices per master
    media_prices = pd.read_excel(os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                 sheet_name='media_prices')
    media_prices = pd.melt(media_prices, id_vars='master', var_name='variable', value_name='price')

    masters = media_prices['master'].unique().tolist()

    # -------------------- load model ------------------------ #

    model_tag = "2026-06-24_161827"
    pth = os.path.join('code_py', 'model_data', 'model_' + model_tag)

    model, dt_p, return_state = fm.model_load_ext(pth, missed_stop=True)
    print(return_state)

    trans_dict = fmr.trans_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                       sheet='translation')

    model_rep = fmr.ModelRep(dt_p, trans_dict=trans_dict,
                             adhoc_code=[code_py.adhoc.translators], language='english')

    dt_p = fmr.media_ssn_apply(dt_p, path=os.path.join('code_py', 'model_data', 'options', 'seasonality.xlsx'),
                               sheet='data')

    # pre-reporting: back-fill future-period variables
    model, dt_p, model_rep = pre_reporting(model, dt_p, model_rep)

    # ------------------ run options (per master) ---------------------- #

    # 2025
    for ms in masters:
        print(f'Option run 2025: {ms}')

        opt_set = fmr.OptionSettings(target=[['brand_1'], [ms]]
                                     , date_start=datetime.strptime('2025-06-30', '%Y-%m-%d').date()
                                     , date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                     , ref_date_start=datetime.strptime('2024-01-01', '%Y-%m-%d').date()
                                     , ref_date_end=datetime.strptime('2024-12-31', '%Y-%m-%d').date()
                                     , apply_vars=["superbrand", "master"]
                                     , plan_period='exact')

        model.conf.trans_path_df = media_prices[media_prices['master'] == ms]

        dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                                    options_c[ms]['predefined'],
                                                    opt_set, if_exact=False)

    # 2026
    for ms in masters:
        print(f'Option run 2026: {ms}')

        opt_set = fmr.OptionSettings(target=[['brand_1'], [ms]]
                                     , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                     , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                     , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                     , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                     , apply_vars=["superbrand", "master"]
                                     , plan_period='exact')

        model.conf.trans_path_df = media_prices[media_prices['master'] == ms]

        dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                                    options_m[ms]['predefined'],
                                                    opt_set, if_exact=False)

    # --------------------- output ----------------------- #

    if if_visual:

        print('R^2 by region train volume')
        print(
            dt_pred[(dt_pred['listed'] == 2)].groupby(['master', 'date']).sum().groupby(['master']).apply(
                lambda x: fm.metrics.r_squared(x['observed'], x['predicted']))
        )

        print('Accuracy by region train volume')
        print(
            dt_pred[(dt_pred['listed'] == 2)].groupby(['master', 'date']).sum().groupby(['master']).apply(
                lambda x: 100 - fm.metrics.mapef(x['observed'], x['predicted']))
        )

        print('Accuracy by region test volume')
        print(
            dt_pred[(dt_pred['listed'] == 3)].groupby(['master', 'date']).sum().groupby(['master']).apply(
                lambda x: 100 - fm.metrics.mapef(x['observed'], x['predicted']))
        )

        fig = fmr.fit_mult_plot_vol(model, dt_pred, model_rep, period='week',
                                    show_future=True, group_var=['superbrand', 'master'])
        fig.show()

        split_m_m = fmr.extract_effect(model, dt_p, model_rep)
        split_m_m = split_m_m.sort_values(['superbrand', 'bs_key', 'date'])

        fig = fmr.decompose_sub_plot(split_m_m=split_m_m, brands=None, model_rep=model_rep
                                     , sku_var=['superbrand', 'master'], period='m', show_future=True
                                     , contour_line=True)
        fig.show()

        fig = fmr.waterfall_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep
                                 , date_start='2024-01-01', date_end='2024-12-31')
        fig.show()

    if if_curves:

        for ms in masters:
            print(f'Curves: {ms}')

            opt_set_crv = fmr.OptionSettings(target=[['brand_1'], [ms]]
                                             , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                             , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                             , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                             , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                             , apply_vars=["superbrand", "master"]
                                             , plan_period='exact')

            model.conf.trans_path_df = media_prices[media_prices['master'] == ms]

            dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                                        options_m[ms]['zero'],
                                                        opt_set_crv, if_exact=True)

            print(datetime.now())
            curves_full_data = fmr.option_report_multi_post(
                model=model
                , ds=dt_p
                , model_rep=model_rep
                , opt_set=opt_set_crv
                , budget_step=5
                , bdg_max=300
                , adhoc_curves_max_costs=None
                , if_exact=True
                , cores=10)
            print(datetime.now())

            winsound.Beep(400, 1000)

            curves_full_figs = fmr.curves_full_plot_short(curves_full_data, model_rep)

            for fig in curves_full_figs.values():
                fig.show()

            time.sleep(5)

            curves_full_figs = fmr.curves_full_plot_long(curves_full_data, model_rep)

            for fig in curves_full_figs.values():
                fig.show()

    if if_export:

        for ms in masters:
            print(f'Export: {ms}')

            model.conf.trans_path_df = media_prices[media_prices['master'] == ms]

            model_rep, prs = export_pptx.set_config(model_rep, template_name='blanc')

            trg_vars = [['brand_1'], [ms]]
            group_vars = ['superbrand', 'master']

            opt_set = fmr.OptionSettings(target=trg_vars
                                         , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                         , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                         , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                         , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                         , plan_period='exact'
                                         , apply_vars=group_vars)

            opt_set_retro = fmr.OptionSettings(target=[['brand_1']]
                                               , date_start=datetime.strptime('2024-01-01', '%Y-%m-%d').date()
                                               , date_end=datetime.strptime('2024-12-31', '%Y-%m-%d').date()
                                               , ref_date_start=datetime.strptime('2024-01-01', '%Y-%m-%d').date()
                                               , ref_date_end=datetime.strptime('2024-12-31', '%Y-%m-%d').date()
                                               , plan_period='exact'
                                               , apply_vars=['superbrand'])

            prs = export_pptx.export_option_detail(prs, model, model_rep
                                                   , dt_pred=dt_pred[dt_pred['listed'].isin([2, 3])]
                                                   , ds=dt_p[dt_p['listed'].isin([2, 3])]
                                                   , opt_set=opt_set_retro
                                                   , sku_var=['superbrand', 'master']
                                                   , options_m=None
                                                   , period='month'
                                                   , if_volume=True
                                                   , if_sku_fit_predict=True
                                                   , if_sku_decompose=True
                                                   , if_sku_waterfall=True
                                                   )

            opt_set_crv = fmr.OptionSettings(target=trg_vars,
                                             date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date(),
                                             date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date(),
                                             ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date(),
                                             ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date(),
                                             plan_period='exact',
                                             apply_vars=group_vars)

            dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                                        options_m[ms]['predefined'],
                                                        opt_set_crv, if_exact=False)

            prs = export_pptx.export_curves(prs, model, model_rep
                                            , ds=dt_p
                                            , trans_dict=None
                                            , opt_set_crv=opt_set_crv
                                            , budget_step=2
                                            , bdg_max=400
                                            , fixed_vars=None
                                            , cores=10
                                            , if_exact=True
                                            , adhoc_curves_max_costs=None
                                            )

            prs = export_pptx.export_option_table(prs, model, model_rep
                                                  , dt_pred=dt_pred
                                                  , ds=dt_p
                                                  , opt_set=opt_set
                                                  , options_m=options_m[ms]
                                                  , if_exact=False
                                                  )

            prs.save(
                "output/_Econometrics_" + model.conf.target_superbrand.capitalize()
                + "_" + ms + "_"
                + pd.Timestamp('today').strftime('%Y%m%d') + ".pptx")

            winsound.Beep(400, 1000)

    if if_save:

        model_rep = fmr.ModelRep(dt_p, trans_dict=trans_dict,
                                 adhoc_code=[code_py.adhoc.reporting], language='english')

        model.save(dt_p
                   , model_rep=model_rep
                   , path=os.path.join(os.getcwd(), 'code_py', 'model_data')
                   , save_tag=model_tag + '_dash')

    if if_debug:
        breakpoint()
        1
