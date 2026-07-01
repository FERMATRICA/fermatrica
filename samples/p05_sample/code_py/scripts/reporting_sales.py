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
from code_py.adhoc.reporting import pred_merge_bht

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
    if_visual = False
    if_curves = False
    if_export = True
    if_save = False
    if_debug = True

    # -------------------- load options ------------------------ #

    # Translation dicts per node
    trans_dict_1 = fmr.trans_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                         sheet='translation_1')
    trans_dict_2 = fmr.trans_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                         sheet='translation_2')

    # options
    options_m = fmr.budget_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                           sheet='main', vat_rate=0)
    options_c = fmr.budget_dict_create(path=os.path.join('code_py', 'model_data', 'options', 'options.xlsx'),
                                           sheet='current', vat_rate=0)

    # ------------------ load model_1 (BHT) ---------------------- #

    model_tag_1 = None  # Will be auto-detected
    pth_bht = os.path.join('code_py', 'model_data_bht')

    # Auto-detect latest saved model
    if model_tag_1 is None:
        saved_dirs = [d for d in os.listdir(pth_bht) if d.startswith('model_') and os.path.isdir(os.path.join(pth_bht, d))]
        if saved_dirs:
            saved_dirs.sort(reverse=True)
            model_tag_1 = saved_dirs[0].replace('model_', '')
            print(f'Auto-detected BHT model: {model_tag_1}')

    pth = os.path.join(pth_bht, 'model_' + model_tag_1)
    model_1, dt_p_1, return_state = fm.model_load_ext(pth, missed_stop=True)
    print(f'Loaded BHT model: {return_state}')

    dt_p_1 = pd.read_excel('data/data_processed/data_for_sample_p05_bht.xlsx', parse_dates=['date'])
    fm.transform(ds=dt_p_1, model=model_1, set_start=False, if_by_ref=True)

    model_rep_1 = fmr.ModelRep(dt_p_1, trans_dict=trans_dict_1, language='english')

    dt_p_1 = fmr.media_ssn_apply(
        dt_p_1, path=os.path.join('code_py', 'model_data', 'options', 'seasonality.xlsx'),
        sheet='data')

    # ------------------ load model_2 (Sales) ---------------------- #

    model_tag_2 = None
    pth_sales = os.path.join('code_py', 'model_data_sales')

    if model_tag_2 is None:
        saved_dirs = [d for d in os.listdir(pth_sales) if d.startswith('model_') and os.path.isdir(os.path.join(pth_sales, d))]
        if saved_dirs:
            saved_dirs.sort(reverse=True)
            model_tag_2 = saved_dirs[0].replace('model_', '')
            print(f'Auto-detected Sales model: {model_tag_2}')

    pth = os.path.join(pth_sales, 'model_' + model_tag_2)
    model_2, dt_p_2, return_state = fm.model_load_ext(pth, missed_stop=True)
    print(f'Loaded Sales model: {return_state}')

    dt_p_2 = pd.read_excel('data/data_processed/data_for_sample_p05_sales.xlsx', parse_dates=['date'])
    fm.transform(ds=dt_p_2, model=model_2, set_start=False, if_by_ref=True)

    model_rep_2 = fmr.ModelRep(dt_p_2, trans_dict=trans_dict_2, language='english')

    dt_p_2 = fmr.media_ssn_apply(
        dt_p_2, path=os.path.join('code_py', 'model_data', 'options', 'seasonality.xlsx'),
        sheet='data')

    # ------------------ run option ---------------------- #

    # 2025 scenario
    opt_set = fmr.OptionSettings(target=[['brand_1']]
                                 , date_start=datetime.strptime('2025-06-30', '%Y-%m-%d').date()
                                 , date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                 , ref_date_start=datetime.strptime('2024-01-01', '%Y-%m-%d').date()
                                 , ref_date_end=datetime.strptime('2024-12-31', '%Y-%m-%d').date()
                                 , apply_vars=["superbrand"]
                                 , plan_period='exact')

    dt_p_1, dt_pred_1, opt_sum_1 = fmr.option_report(model_1, dt_p_1, model_rep_1,
                                                      options_c['predefined'], opt_set, if_exact=False)
    dt_p_2 = pred_merge_bht(dt_p_2, dt_pred_1)
    dt_p_2, dt_pred_2, opt_sum_2 = fmr.option_report(model_2, dt_p_2, model_rep_2,
                                                      options_c['predefined'], opt_set, if_exact=False)

    # 2026 scenario
    opt_set = fmr.OptionSettings(target=[['brand_1']]
                                 , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                 , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                 , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                 , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                 , apply_vars=["superbrand"]
                                 , plan_period='exact')

    dt_p_1, dt_pred_1, opt_sum_1 = fmr.option_report(model_1, dt_p_1, model_rep_1,
                                                      options_m['predefined'], opt_set, if_exact=False)
    dt_p_2 = pred_merge_bht(dt_p_2, dt_pred_1)
    dt_p_2, dt_pred_2, opt_sum_2 = fmr.option_report(model_2, dt_p_2, model_rep_2,
                                                      options_m['predefined'], opt_set, if_exact=False)

    # --------------------- output ----------------------- #

    if if_visual:

        print('BHT R^2 by superbrand train volume')
        print(
            dt_pred_1[(dt_pred_1['listed'] == 2)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(
                lambda x: fm.metrics.r_squared(x['observed'], x['predicted']))
        )

        print('Sales R^2 by superbrand train volume')
        print(
            dt_pred_2[(dt_pred_2['listed'] == 2)].groupby(['superbrand', 'date']).sum().groupby(['superbrand']).apply(
                lambda x: fm.metrics.r_squared(x['observed'], x['predicted']))
        )

        # fit / predict — BHT
        fig = fmr.fit_main_plot_vol(model_1, dt_pred_1, model_rep_1, period='month', show_future=True,
                                    target_superbrand='brand_1')
        fig.show()

        # fit / predict — Sales
        fig = fmr.fit_main_plot_vol(model_2, dt_pred_2, model_rep_2, period='week', show_future=True,
                                    target_superbrand='brand_1')
        fig.show()

        # Decomposition — Sales
        split_m_m = fmr.extract_effect(model_2, dt_p_2, model_rep_2)
        split_m_m = split_m_m.sort_values(['superbrand', 'bs_key', 'date'])

        fig = fmr.decompose_main_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep_2
                                      , period='m', show_future=True, contour_line=True)
        fig.show()

        fig = fmr.waterfall_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep_2
                                 , date_start='2024-01-01', date_end='2024-12-31')
        fig.show()

    if if_curves:

        opt_set_crv = fmr.OptionSettings(target=['brand_1']
                                         , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                         , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                         , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                         , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                         , plan_period='exact')

        dt_p_1, dt_pred_1, opt_sum_1 = fmr.option_report(model_1, dt_p_1, model_rep_1,
                                                          options_m['zero'], opt_set_crv, if_exact=True)
        dt_p_2 = pred_merge_bht(dt_p_2, dt_pred_1)
        dt_p_2, dt_pred_2, opt_sum_2 = fmr.option_report(model_2, dt_p_2, model_rep_2,
                                                          options_m['zero'], opt_set_crv, if_exact=True)

        print(datetime.now())
        curves_full_data = fmr.option_report_multi_post(
            model=[model_1, model_2]
            , ds=[dt_p_1, dt_p_2]
            , model_rep=[model_rep_1, model_rep_2]
            , opt_set=opt_set_crv
            , budget_step=5
            , bdg_max=300
            , adhoc_curves_max_costs=None
            , if_exact=True
            , cores=14)
        print(datetime.now())

        # Plot curves for Sales model (index 1)
        curves_full_figs = fmr.curves_full_plot_short(curves_full_data[1], model_rep_2)
        for fig in curves_full_figs.values():
            fig.show()

        time.sleep(5)

        curves_full_figs = fmr.curves_full_plot_long(curves_full_data[1], model_rep_2)
        for fig in curves_full_figs.values():
            fig.show()

    if if_export:

        model_rep_1, prs = export_pptx.set_config(model_rep_1, template_name='blanc')
        del prs
        model_rep_2, prs = export_pptx.set_config(model_rep_2, template_name='blanc')

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

        prs = export_pptx.export_option_detail(prs, model_2, model_rep_2
                                               , dt_pred=dt_pred_2[dt_pred_2['listed'].isin([2, 3])]
                                               , ds=dt_p_2[dt_p_2['listed'].isin([2, 3])]
                                               , opt_set=opt_set_retro
                                               , sku_var=['superbrand', 'master', 'form']
                                               , options_m=None
                                               , period='month'
                                               , if_volume=True
                                               , if_sku_fit_predict=True
                                               , if_sku_decompose=True
                                               , if_sku_waterfall=True
                                                )

        # curves
        opt_set_crv = fmr.OptionSettings(target=[['brand_1']]
                                         , date_start=datetime.strptime('2026-01-01', '%Y-%m-%d').date()
                                         , date_end=datetime.strptime('2026-12-31', '%Y-%m-%d').date()
                                         , ref_date_start=datetime.strptime('2025-01-01', '%Y-%m-%d').date()
                                         , ref_date_end=datetime.strptime('2025-12-31', '%Y-%m-%d').date()
                                         , plan_period='exact'
                                         , apply_vars=['superbrand'])
        
        dt_p_1, dt_pred_1, opt_sum_1 = fmr.option_report(model_1, dt_p_1, model_rep_1,
                                                         options_m['zero'], opt_set_crv, if_exact=True)
        dt_p_2 = pred_merge_bht(dt_p_2, dt_pred_1)
        dt_p_2, dt_pred_2, opt_sum_2 = fmr.option_report(model_2, dt_p_2, model_rep_2,
                                                         options_m['zero'], opt_set_crv, if_exact=True)
        
        prs = export_pptx.export_curves(prs
                                        , [model_1, model_2]
                                        , [model_rep_1, model_rep_2]
                                        , ds=[dt_p_1, dt_p_2]
                                        , opt_set_crv=opt_set_crv
                                        , budget_step=5
                                        , bdg_max=300
                                        , fixed_vars=None
                                        , cores=14
                                        , if_exact=True
                                        )

        # Options table
        prs = export_pptx.export_option_table(prs, model_2, model_rep_2
                                              , dt_pred=dt_pred_2
                                              , ds=dt_p_2
                                              , opt_set=opt_set
                                              , options_m=options_m
                                              , if_exact=False
                                              )

        prs.save(
            "output/_Econometrics_" + model_2.conf.target_superbrand.capitalize() + "_sales_" + pd.Timestamp('today').strftime(
                '%Y%m%d') + ".pptx")

    if if_save:
        model_rep_2 = fmr.ModelRep(dt_p_2, trans_dict=trans_dict_2, language='english')
        model_2.save(dt_p_2, model_rep=model_rep_2
                     , path=os.path.join(os.getcwd(), 'code_py', 'model_data_sales')
                     , save_tag=model_tag_2 + '_dash')

    if if_debug:
        breakpoint()
        1
