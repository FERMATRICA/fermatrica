import os
import time
import warnings
from datetime import datetime

import fermatrica as fm
import fermatrica_rep as fmr
import fermatrica_rep.export.export_pptx as export_pptx
import pandas as pd
import plotly.io as pio

import code_py.adhoc.translators

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

    # -------------------- load options ------------------------ #

    options_c = fmr.budget_dict_create(
        path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
        sheet='current', vat_rate=0)
    options_m = fmr.budget_dict_create(
        path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
        sheet='main', vat_rate=0)

    # Shared translation dict
    trans_dict = fmr.trans_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                       sheet='translation_1')

    # -------------------- load model ------------------------ #

    pth_bht = os.path.join('code_py', 'model_data_bht')
    saved_dirs = [d for d in os.listdir(pth_bht)
                  if d.startswith('model_') and os.path.isdir(os.path.join(pth_bht, d))]
    saved_dirs.sort(reverse=True)
    model_tag = saved_dirs[0].replace('model_', '') if saved_dirs else None
    print(f'Using BHT model tag: {model_tag}')

    pth = os.path.join(pth_bht, 'model_' + model_tag)
    model, dt_p, return_state = fm.model_load_ext(pth, missed_stop=True)
    print(return_state)

    dt_p = pd.read_excel('data/data_processed/data_for_sample_p05_bht.xlsx', parse_dates=['date'])
    fm.transform(ds=dt_p, model=model, set_start=False, if_by_ref=True)

    model_rep = fmr.ModelRep(dt_p, trans_dict=trans_dict,
                             adhoc_code=[code_py.adhoc.translators], language='english')

    dt_p = fmr.media_ssn_apply(
        dt_p, path=os.path.join('code_py', 'model_data', 'options', 'seasonality.xlsx'),
        sheet='data')

    # ------------------ run option ---------------------- #

    # 2025
    opt_set = fmr.OptionSettings(target=[['brand_1']]
                                 , date_start=datetime.strptime('2025-06-30', '%Y-%m-%d').date()
                                 , date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                 , ref_date_start=datetime.strptime('2024-01-01', '%Y-%m-%d').date()
                                 , ref_date_end=datetime.strptime('2024-12-31', '%Y-%m-%d').date()
                                 , apply_vars=["superbrand"]
                                 , plan_period='exact')

    dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                               options_c['predefined'], opt_set, if_exact=False)

    # 2026
    opt_set = fmr.OptionSettings(target=[['brand_1']]
                                 , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                 , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                 , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                 , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                 , apply_vars=["superbrand"]
                                 , plan_period='exact')

    dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                               options_m['predefined'], opt_set, if_exact=False)

    # --------------------- output ----------------------- #

    if if_visual:

        print('R^2 by superbrand train volume')
        print(
            dt_pred[(dt_pred['listed'] == 2)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(
                lambda x: fm.metrics.r_squared(x['observed'], x['predicted']))
        )

        print('Accuracy by superbrand train volume')
        print(
            dt_pred[(dt_pred['listed'] == 2)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(
                lambda x: 100 - fm.metrics.mapef(x['observed'], x['predicted']))
        )

        print('Accuracy by superbrand test volume')
        print(
            dt_pred[(dt_pred['listed'] == 3)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(
                lambda x: 100 - fm.metrics.mapef(x['observed'], x['predicted']))
        )

        fig = fmr.fit_main_plot_vol(model, dt_pred, model_rep, period='month', show_future=True,
                                    target_superbrand='brand_1')
        fig.show()

        fig = fmr.fit_mult_plot_vol(model, dt_pred, model_rep, period='month', show_future=True,
                                    group_var=['superbrand'])
        fig.show()

        split_m_m = fmr.extract_effect(model, dt_p, model_rep, listed_filter=[1, 2, 3, 4])
        split_m_m = split_m_m.sort_values(['superbrand', 'bs_key', 'date'])

        fig = fmr.decompose_main_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep
                                      , period='m', show_future=True, contour_line=True)
        fig.show()

        fig = fmr.waterfall_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep
                                 , date_start='2024-01-01', date_end='2024-12-31')
        fig.show()

    if if_curves:

        opt_set_crv = fmr.OptionSettings(target=['brand_1']
                                         , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                         , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                         , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                         , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                         , plan_period='exact')

        dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                                   options_m['zero'], opt_set_crv, if_exact=True)

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
            , cores=14)
        print(datetime.now())

        curves_full_figs = fmr.curves_full_plot_short(curves_full_data, model_rep)
        for fig in curves_full_figs.values():
            fig.show()
            break

        time.sleep(5)

        curves_full_figs = fmr.curves_full_plot_long(curves_full_data, model_rep)
        for fig in curves_full_figs.values():
            fig.show()
            break

    if if_export:

        model_rep, prs = export_pptx.set_config(model_rep, template_name='blanc')

        trg_vars = [['brand_1']]
        group_vars = ['superbrand']

        opt_set = fmr.OptionSettings(target=trg_vars
                                     , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                     , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                     , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                     , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                     , plan_period='exact'
                                     , apply_vars=group_vars)

        opt_set_retro = fmr.OptionSettings(target=trg_vars
                                           , date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                           , date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                           , ref_date_start=datetime.strptime('2024-01-01', '%Y-%m-%d').date()
                                           , ref_date_end=datetime.strptime('2024-12-31', '%Y-%m-%d').date()
                                           , plan_period='exact'
                                           , apply_vars=group_vars)

        prs = export_pptx.export_option_detail(prs, model, model_rep
                                               , dt_pred=dt_pred[dt_pred['listed'].isin([2, 3])]
                                               , ds=dt_p[dt_p['listed'].isin([2, 3])]
                                               , opt_set=opt_set_retro
                                               , sku_var=['superbrand']
                                               , options_m=None
                                               , period='month'
                                               , if_volume=True
                                               , if_sku_fit_predict=False
                                               , if_sku_decompose=False
                                               , if_sku_waterfall=False
                                                )

        # curves

        opt_set_crv = fmr.OptionSettings(target=[['brand_1']]
                                         , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                         , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                         , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                         , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                         , plan_period='exact'
                                         , apply_vars=['superbrand'])
        
        dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep,
                                                   options_m['zero'], opt_set_crv, if_exact=True)
        
        prs = export_pptx.export_curves(prs, model, model_rep
                                        , ds=dt_p
                                        , opt_set_crv=opt_set_crv
                                        , budget_step=5
                                        , bdg_max=300
                                        , fixed_vars=None
                                        , cores=14
                                        , if_exact=True
                                        , adhoc_curves_max_costs=None
                                        , if_volume=True
                                        )

        # Options table
        prs = export_pptx.export_option_table(prs, model, model_rep
                                              , dt_pred=dt_pred
                                              , ds=dt_p
                                              , opt_set=opt_set
                                              , options_m=options_m
                                              , if_exact=False
                                              )

        prs.save(
            "output/_Econometrics_" + model.conf.target_superbrand.capitalize() + "_BHT_"
            + pd.Timestamp('today').strftime('%Y%m%d') + ".pptx")

    if if_save:

        model_rep = fmr.ModelRep(dt_p, trans_dict=trans_dict,
                                 adhoc_code=[code_py.adhoc.translators], language='english')

        model.save(dt_p
                   , model_rep=model_rep
                   , path=os.path.join(os.getcwd(), 'code_py', 'model_data_bht')
                   , save_tag=model_tag + '_dash')

    if if_debug:
        breakpoint()
        1
