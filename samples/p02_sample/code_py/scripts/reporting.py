import os
import re
import time
import warnings
from datetime import datetime

import fermatrica as fm
import fermatrica_rep as fmr
import fermatrica_rep.export.export_pptx as export_pptx
import pandas as pd
import plotly.io as pio
import winsound

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

    # script settings

    if_correct = False
    if_visual = False
    if_curves = False
    if_export = True
    if_save = False
    if_debug = True

    # -------------------- load ------------------------ #

    options_c = fmr.budget_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'), sheet='current', vat_rate=0)
    options_m = fmr.budget_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'), sheet='main', vat_rate=0)

    # load model

    model_tag = "2026-06-24_161124"  #"YYYY-MM-DD_hhmmss"
    pth = os.path.join('code_py', 'model_data', 'model_' + model_tag)

    model, dt_p, return_state = fm.model_load_ext(pth, missed_stop=True)
    print(return_state)

    # load reporting environment

    trans_dict = fmr.trans_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                       sheet='translation')

    model_rep = fmr.ModelRep(dt_p, trans_dict=trans_dict,
                             adhoc_code=[code_py.adhoc.translators], language='english')

    dt_p = fmr.media_ssn_apply(dt_p, path=os.path.join('code_py', 'model_data', 'options', 'seasonality.xlsx'),
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

    dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep, options_c['predefined'], opt_set, if_exact=False)

    # 2026

    opt_set = fmr.OptionSettings(target=[['brand_1']]
                                 , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                 , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                 , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                 , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                 , apply_vars=["superbrand"]
                                 , plan_period='exact')

    dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep, options_m['predefined'], opt_set, if_exact=False)

    # --------------------- output ----------------------- #

    if if_visual:

        print('R^2 by superbrand train volume')
        print(
            dt_pred[(dt_pred['listed'] == 2)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(lambda x: fm.metrics.r_squared(x['observed'], x['predicted']))
        )

        print('Accuracy by superbrand train volume')
        print(
            dt_pred[(dt_pred['listed'] == 2)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(lambda x: 100 - fm.metrics.mapef(x['observed'], x['predicted']))
        )

        print('Accuracy by superbrand test volume')
        print(
            dt_pred[(dt_pred['listed'] == 3)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(lambda x: 100 - fm.metrics.mapef(x['observed'], x['predicted']))
        )

        # fit / predict

        fig = fmr.fit_mult_plot_vol(model, dt_pred, model_rep, period='week', show_future=True, group_var=['superbrand'])
        fig.show()

        # decomposition

        split_m_m = fmr.extract_effect(model, dt_p, model_rep)
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

        dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep, options_m['zero'], opt_set_crv, if_exact=True)

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

        # retro

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
                                               , sku_var=['superbrand', 'master', 'form']
                                               , options_m=None
                                               , period='month'
                                               , if_volume=True
                                               , if_sku_fit_predict=True
                                               , if_sku_decompose=True
                                               , if_sku_waterfall=True
                                               )

        # # curves

        # opt_set_crv = fmr.OptionSettings(target=trg_vars,
        #                                  date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date(),
        #                                  date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date(),
        #                                  ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date(),
        #                                  ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date(),
        #                                  plan_period='exact',
        #                                  apply_vars=group_vars)

        # dt_p, dt_pred, opt_sum = fmr.option_report(model, dt_p, model_rep, options_m['predefined'], opt_set_crv, if_exact=False)

        # prs = export_pptx.export_curves(prs, model, model_rep
        #                                 , ds=dt_p
        #                                 , trans_dict=None
        #                                 , opt_set_crv=opt_set_crv
        #                                 , budget_step=2
        #                                 , bdg_max=400
        #                                 , fixed_vars=None
        #                                 , cores=10
        #                                 , if_exact=True
        #                                 , adhoc_curves_max_costs=None
        #                                 )

        # options table

        prs = export_pptx.export_option_table(prs, model, model_rep
                                              , dt_pred=dt_pred
                                              , ds=dt_p
                                              , opt_set=opt_set
                                              , options_m=options_m
                                              , if_exact=False
                                              )

        prs.save(
            "output/_Econometrics_" + model.conf.target_superbrand.capitalize() + "_" + pd.Timestamp('today').strftime(
                '%Y%m%d') + ".pptx")

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
