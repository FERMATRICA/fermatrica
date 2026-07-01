import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
p05_root = os.path.join(_SCRIPT_DIR, '..', '..')


def make_notebook(cells):
    return {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.0'}
        },
        'cells': cells
    }


def md(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [source]}


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [source]}


def write_notebook(cells, filename):
    path = os.path.join(p05_root, 'code_py', 'scripts', filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(make_notebook(cells), f, indent=1)
    print(f'Created: {path}')


# ============================================================
# model_bht.ipynb — BHT model (brand-month-ABA)
# Follows p03 model.ipynb conventions exactly.
# ============================================================
bht_cells = [

    code('import copy\n'
         'import io\n'
         'import numpy as np\n'
         'import os\n'
         'import pandas as pd\n'
         'import pkgutil\n'
         'import re\n'
         'import time\n'
         'import winsound\n'
         '\n'
         "main_curr_dir = os.getcwd()\n"
         "if os.path.basename(os.getcwd()) == 'scripts':\n"
         "    os.chdir('../../')\n"
         '\n'
         'import fermatrica as fm\n'
         'import fermatrica_rep as fmr\n'
         '\n'
         'import code_py.adhoc.model\n'
         'import code_py.adhoc.reporting\n'
         '\n'
         'metrics_max = 0\n'
         'metrics_min = np.inf\n'
         '\n'
         'import warnings\n'
         "warnings.simplefilter(action='ignore', category=FutureWarning)"),

    md('## Set environment'),

    code("pd.set_option('display.max_columns', None)\n"
         "pd.set_option('display.expand_frame_repr', False)\n"
         "pd.set_option('max_colwidth', None)\n"
         '\n'
         '# script settings\n'
         '\n'
         "if_opt = False  # optimize / run w/o optimizing\n"
         "cur_algo = 'l_cma'  # 'l_cma', 'l_cobyla', 'g_de'\n"
         "if_visual = True  # switch on/off visuals\n"
         "if_save = True  # save model and for further analysis"),

    md('## How to set model\n'
       '\n'
       '1. Create or copy project with following structure:\n'
       '    ```-[root]\n'
       '        -code_py\n'
       '            -adhoc\n'
       '                -__init__.py\n'
       '                -model.py [optional]\n'
       '            -model_data\n'
       '                -model_conf.xlsx\n'
       '            -scripts\n'
       '                -model.ipynb\n'
       '                \n'
       '2. File "model_conf.xlsx"\n'
       '    1. Setup (model type, Y var, (product) price var, target brand, series frequency, conversion): sheet "setup"\n'
       '    2. RHS (linear model specs): sheet "RHS"\n'
       '    3. Transformations: sheet "params"\n'
       '    4. Scoring (mostly for optimising): sheet "scoring"\n'
       '    5. LHS (optional, Y transformations): sheet "LHS"\n'
       '3. Data file\n'
       '    1. Mandatory fields:\n'
       '        1. "superbrand" (string): brand name\n'
       '        2. "bs_key" (int): SKU / entity id\n'
       '        3. "date" (datetime): date\n'
       '        4. "listed" (int):\n'
       '            - 1: before train period\n'
       '            - 2: train period\n'
       '            - 3: test period\n'
       '            - 4: future\n'
       '        5. any kind of target variable (as set in "model_conf.xlsx")\n'
       '        6. any kind of produce price variable (as set in "model_conf.xlsx")'),

    md('## Load data and current model definition'),

    code("dt_p = pd.read_excel('data/data_processed/data_for_sample_p05_bht.xlsx', parse_dates=['date'])\n"
         '\n'
         "# 'superbrand' as brand name and 'bs_key' as SKU / entity id\n"
         '# Already set in data\n'
         '\n'
         "# 'Intercept' is not required\n"
         "if 'Intercept' in dt_p.columns:\n"
         "    del dt_p['Intercept']  # much faster than .drop()\n"
         '\n'
         "pth = os.path.join('code_py', 'model_data_bht', 'model_conf.xlsx')\n"
         '\n'
         'model = fm.Model(path=pth\n'
         '                , adhoc_code=[code_py.adhoc.model]\n'
         '                , ds=dt_p)\n'
         '\n'
         '\n'
         'start_time = time.time()\n'
         '\n'
         'model = fm.transform(ds=dt_p\n'
         '                    , model=model\n'
         '                    , set_start=True\n'
         '                    , if_by_ref=True)\n'
         '\n'
         "print('time elapsed: ' + str((time.time() - start_time)))"),

    code('dt_p.shape'),

    md('## Optimising (optional)'),

    code('if if_opt:\n'
         '    model_back = copy.deepcopy(model)\n'
         '\n'
         "    if cur_algo == 'l_cobyla':\n"
         '        model = fm.optimize_local_cobyla(dt_p\n'
         '                                        , model\n'
         '                                        , revert_score_sign=False\n'
         '                                        , verbose=True\n'
         '                                        , epochs=1\n'
         '                                        , iters_epoch=100\n'
         '                                        , error_score=1e+12\n'
         '                                        , ftol_abs=.1\n'
         '                                        , bounds_rel_distance=0.0)\n'
         "    elif cur_algo == 'l_cma':\n"
         '        model = fm.optimize_local_ncma(dt_p\n'
         '                                        , model\n'
         '                                        , revert_score_sign=False\n'
         '                                        , epochs=1\n'
         '                                        , verbose=True\n'
         '                                        , iters_epoch=2000\n'
         '                                        , error_score=1e+12\n'
         '                                        , cores=15\n'
         '                                        , approach=\'init_sequent\'\n'
         '                                        , save_epoch=False\n'
         '                                        , bounds_rel_distance=0.05\n'
         '                                        )\n'
         "    elif cur_algo == 'g_de':\n"
         '        model = fm.optimize_global_de(dt_p\n'
         '                                        , model\n'
         '                                        , revert_score_sign=False\n'
         '                                        , epochs=50\n'
         '                                        , verbose=True\n'
         '                                        , iters_epoch=90\n'
         '                                        , pop_size=45\n'
         '                                        , mutation=.8\n'
         '                                        , max_no_improvement=10\n'
         '                                        , crossover=.9\n'
         '                                        , pop_init_method=\'lhs\'\n'
         '                                        , mut_method=\'combined\'\n'
         '                                        , if_init=False\n'
         '                                        , ftol_abs=.01\n'
         '                                        , error_score=1e+12\n'
         '                                        , cores=15\n'
         "                                        , save_epoch=os.path.join(os.getcwd(), 'output', 'data', 'global_de_sales_2804'))\n"
         '    model = copy.deepcopy(model)\n'
         '\n'
         '    winsound.Beep(400, 1000)'),

    md('## Get results'),

    code("model = fm.transform(ds=dt_p\n"
         "                  , model=model\n"
         "                  , set_start=False\n"
         "                  , if_by_ref=True)\n"
         '\n'
         'pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)\n'
         'pred = fm.fit_predict(dt_p, model, if_fit=False, return_full=False)\n'
         '\n'
         'dt_pred = fm.predict_ext(model, dt_p)\n'
         '\n'
         '\n'
         'rsrc = pkgutil.get_data("fermatrica_rep", "/res/dict/vis_dict.xlsx")\n'
         'xlsxio = io.BytesIO(rsrc)\n'
         'vis_dict = pd.read_excel(xlsxio)\n'
         '\n'
         'model_rep = fmr.ModelRep(dt_p, vis_dict)'),

    code('dt_p.shape'),

    md('## Statistic and metrics'),

    md('### Consolidated statistics'),

    code("print('\\nScoring\\n')\n"
         '\n'
         "print('Full RMSE : ' + str(round(fm.metrics.rmse(dt_p.loc[dt_p['listed'].isin([2, 3]), model.conf.Y_var], pred[dt_p['listed'].isin([2, 3])]), 2)))\n"
         "print('Full Combined scoring : ' + str(\n"
         "    round(fm.scoring(dt_p.loc[dt_p['listed'].isin([2, 3]), :], pred[dt_p['listed'].isin([2, 3])], model), 5)))\n"
         '\n'
         "print('\\nFit\\n')\n"
         '\n'
         "print('R^2 train: ' + str(round(fm.metrics.r_squared(dt_p.loc[dt_p['listed'].isin([2]), model.conf.Y_var], pred[dt_p['listed'].isin([2])]), 4)))\n"
         "print('MAPE train: ' + str(round(fm.metrics.mapef(dt_p.loc[dt_p['listed'].isin([2]), model.conf.Y_var], pred[dt_p['listed'].isin([2])]), 4)))\n"
         "print('MAPE test: ' + str(round(fm.metrics.mapef(dt_p.loc[dt_p['listed'].isin([3]), model.conf.Y_var], pred[dt_p['listed'].isin([3])]), 4)))"),

    md('### Model summary'),

    code("print('\\nMain model\\n')\n"
         "print(model.obj.models['main'].summary())\n"
         '\n'
         "metrics_table = fmr.metrics_group_table(dt_pred, group_vars=['superbrand'])"),

    code("print('Grouped metrics\\n')\n"
         'print(metrics_table)'),

    code("if 'model_conf_back' in locals() or 'model_conf_back' in globals():\n"
         '\n'
         "    tmp = pd.merge(model.conf.params[['variable', 'fun', 'arg', 'value']]\n"
         "                    , model.conf_back.params[['variable', 'fun', 'arg', 'value']], on=['variable', 'fun', 'arg'])\n"
         "    tmp = tmp[(tmp['value_x'] != tmp['value_y']) & (tmp['value_x'].notna())]\n"
         '\n'
         "    tmp['diff'] = 0\n"
         "    mask = (tmp['value_y'] != 0) & (pd.to_numeric(tmp['value_y'], errors='coerce').notna())\n"
         "    tmp.loc[mask, 'diff'] = tmp.loc[mask, 'value_x'] / tmp.loc[mask, 'value_y'] - 1\n"
         '\n'
         "    print('Changed params\\n')\n"
         '    tmp'),

    md('### VIF (Multicollinearity)'),

    code("vif_tbl = fmr.stats.vif_table(model.obj.models['main'])\n"
         'vif_tbl'),

    md('## Observed and predicted'),

    code('if if_visual:\n'
         '    \n'
         "    fig = fmr.fit_mult_plot_vol(model, dt_pred, model_rep, period='week', group_var=['superbrand'], show_future=False)\n"
         '    fig.show()'),

    md('## Decomposition and Waterfall'),

    code('split_m_m = None\n'
         '\n'
         'if if_visual:\n'
         '\n'
         '    split_m_m = fmr.extract_effect(model, dt_p, model_rep)\n'
         "    split_m_m = split_m_m.sort_values(['superbrand', 'date'])\n"
         '\n'
         '    fig = fmr.decompose_sub_plot(split_m_m=split_m_m, brands=None, model_rep=model_rep\n'
         "                                    , sku_var=['superbrand'], period='week', show_future=False\n"
         '                                    , contour_line=True)\n'
         '    fig.show()\n'
         '\n'
         "    fig = fmr.waterfall_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep\n"
         "                                                    , date_start='2020-01-01', date_end='2023-12-31')\n"
         '    fig.show()'),

    md('## Save or export results'),

    code("pth = ''\n"
         '\n'
         'if if_save:\n'
         '\n'
         "    pth = model.save(dt_p, path=os.path.join(os.getcwd(), 'code_py', 'model_data_bht'))"),

]

write_notebook(bht_cells, 'model_bht.ipynb')


# ============================================================
# model_sales.ipynb — Sales model (SKU-week-units with ABA)
# Follows p02 model.ipynb conventions + BHT model loading step.
# ============================================================
sales_cells = [

    code('import copy\n'
         'import io\n'
         'import numpy as np\n'
         'import os\n'
         'import pandas as pd\n'
         'import pkgutil\n'
         'import re\n'
         'import time\n'
         'import winsound\n'
         '\n'
         "main_curr_dir = os.getcwd()\n"
         "if os.path.basename(os.getcwd()) == 'scripts':\n"
         "    os.chdir('../../')\n"
         '\n'
         'import fermatrica as fm\n'
         'import fermatrica_rep as fmr\n'
         '\n'
         'import code_py.adhoc.model\n'
         'import code_py.adhoc.reporting\n'
         'from code_py.adhoc.reporting import pred_merge_bht\n'
         '\n'
         'metrics_max = 0\n'
         'metrics_min = np.inf\n'
         '\n'
         'import warnings\n'
         "warnings.simplefilter(action='ignore', category=FutureWarning)"),

    md('## Set environment'),

    code("pd.set_option('display.max_columns', None)\n"
         "pd.set_option('display.expand_frame_repr', False)\n"
         "pd.set_option('max_colwidth', None)\n"
         '\n'
         '# script settings\n'
         '\n'
         "if_opt = False  # optimize / run w/o optimizing\n"
         "cur_algo = 'l_cma'  # 'l_cma', 'l_cobyla', 'g_de'\n"
         "if_visual = True  # switch on/off visuals\n"
         "if_save = True  # save model and for further analysis"),

    md('## Load BHT model and merge ABA predictions\n'
       '\n'
       'Auto-detect the latest saved BHT model from `model_data_bht/`.\n'
       'Override `model_tag` with a specific timestamp to use a saved model.'),

    code("pth_bht = os.path.join('code_py', 'model_data_bht')\n"
         "saved_dirs = [d for d in os.listdir(pth_bht)\n"
         "              if d.startswith('model_') and os.path.isdir(os.path.join(pth_bht, d))]\n"
         'saved_dirs.sort(reverse=True)\n'
         "model_tag = saved_dirs[0].replace('model_', '') if saved_dirs else None\n"
         "print('Using BHT model: ' + str(model_tag))\n"
         '\n'
         "pth = os.path.join(pth_bht, 'model_' + model_tag)\n"
         'model_1, dt_bht, return_state = fm.model_load_ext(pth, missed_stop=True)\n'
         'print(return_state)\n'
         '\n'
         "dt_bht = pd.read_excel('data/data_processed/data_for_sample_p05_bht.xlsx', parse_dates=['date'])\n"
         'fm.transform(ds=dt_bht, model=model_1, set_start=False, if_by_ref=True)\n'
         '\n'
         "print('BHT data: ' + str(dt_bht.shape) + ', brands=' + str(dt_bht['superbrand'].nunique()))"),

    code("dt_pred_1 = fm.predict_ext(model_1, dt_bht)\n"
         "print('BHT predictions: ' + str(dt_pred_1.shape))\n"
         '\n'
         "dt_p = pd.read_excel('data/data_processed/data_for_sample_p05_sales.xlsx', parse_dates=['date'])\n"
         '\n'
         "if 'Intercept' in dt_p.columns:\n"
         "    del dt_p['Intercept']\n"
         '\n'
         "print('Sales data: ' + str(dt_p.shape) + ', SKUs=' + str(dt_p['bs_key'].nunique()) + ', weeks=' + str(dt_p['date'].nunique()))\n"
         '\n'
         'dt_p = pred_merge_bht(dt_p, dt_pred_1)\n'
         "print('aba_predicted after merge: min=' + str(round(dt_p['aba_predicted'].min(), 2)) + ', max=' + str(round(dt_p['aba_predicted'].max(), 2)))"),

    md('## Load data and current model definition'),

    code("pth = os.path.join('code_py', 'model_data_sales', 'model_conf.xlsx')\n"
         '\n'
         'model = fm.Model(path=pth\n'
         '                , adhoc_code=[code_py.adhoc.model]\n'
         '                , ds=dt_p)\n'
         '\n'
         '\n'
         'start_time = time.time()\n'
         '\n'
         'model = fm.transform(ds=dt_p\n'
         '                    , model=model\n'
         '                    , set_start=True\n'
         '                    , if_by_ref=True)\n'
         '\n'
         "print('time elapsed: ' + str((time.time() - start_time)))"),

    code('dt_p.shape'),

    md('## Optimising (optional)'),

    code('if if_opt:\n'
         '    model_back = copy.deepcopy(model)\n'
         '\n'
         "    if cur_algo == 'l_cobyla':\n"
         '        model = fm.optimize_local_cobyla(dt_p\n'
         '                                        , model\n'
         '                                        , revert_score_sign=False\n'
         '                                        , verbose=True\n'
         '                                        , epochs=1\n'
         '                                        , iters_epoch=100\n'
         '                                        , error_score=1e+12\n'
         '                                        , ftol_abs=.1\n'
         '                                        , bounds_rel_distance=0.0)\n'
         "    elif cur_algo == 'l_cma':\n"
         '        model = fm.optimize_local_ncma(dt_p\n'
         '                                        , model\n'
         '                                        , revert_score_sign=False\n'
         '                                        , epochs=1\n'
         '                                        , verbose=True\n'
         '                                        , iters_epoch=1000\n'
         '                                        , error_score=1e+12\n'
         '                                        , cores=15\n'
         '                                        , approach=\'init_sequent\'\n'
         '                                        , save_epoch=False\n'
         '                                        , bounds_rel_distance=0.05\n'
         '                                        )\n'
         "    elif cur_algo == 'g_de':\n"
         '        model = fm.optimize_global_de(dt_p\n'
         '                                        , model\n'
         '                                        , revert_score_sign=False\n'
         '                                        , epochs=50\n'
         '                                        , verbose=True\n'
         '                                        , iters_epoch=90\n'
         '                                        , pop_size=45\n'
         '                                        , mutation=.8\n'
         '                                        , max_no_improvement=10\n'
         '                                        , crossover=.9\n'
         '                                        , pop_init_method=\'lhs\'\n'
         '                                        , mut_method=\'combined\'\n'
         '                                        , if_init=False\n'
         '                                        , ftol_abs=.01\n'
         '                                        , error_score=1e+12\n'
         '                                        , cores=15\n'
         "                                        , save_epoch=os.path.join(os.getcwd(), 'output', 'data', 'global_de_sales_2804'))\n"
         '    model = copy.deepcopy(model)\n'
         '\n'
         '    winsound.Beep(400, 1000)'),

    md('## Get results'),

    code("model = fm.transform(ds=dt_p\n"
         "                  , model=model\n"
         "                  , set_start=False\n"
         "                  , if_by_ref=True)\n"
         '\n'
         'pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)\n'
         'pred = fm.fit_predict(dt_p, model, if_fit=False, return_full=False)\n'
         '\n'
         'dt_pred = fm.predict_ext(model, dt_p)\n'
         '\n'
         '\n'
         'rsrc = pkgutil.get_data("fermatrica_rep", "/res/dict/vis_dict.xlsx")\n'
         'xlsxio = io.BytesIO(rsrc)\n'
         'vis_dict = pd.read_excel(xlsxio)\n'
         '\n'
         'model_rep = fmr.ModelRep(dt_p, vis_dict)'),

    code('dt_p.shape'),

    md('## Statistic and metrics'),

    md('### Consolidated statistics'),

    code("print('\\nScoring\\n')\n"
         '\n'
         "print('Full RMSE : ' + str(round(fm.metrics.rmse(dt_p.loc[dt_p['listed'].isin([2, 3]), model.conf.Y_var], pred[dt_p['listed'].isin([2, 3])]), 2)))\n"
         "print('Full Combined scoring : ' + str(\n"
         "    round(fm.scoring(dt_p.loc[dt_p['listed'].isin([2, 3]), :], pred[dt_p['listed'].isin([2, 3])], model), 5)))\n"
         '\n'
         "print('\\nFit\\n')\n"
         '\n'
         "print('R^2 train: ' + str(round(fm.metrics.r_squared(dt_p.loc[dt_p['listed'].isin([2]), model.conf.Y_var], pred[dt_p['listed'].isin([2])]), 4)))\n"
         "print('MAPE train: ' + str(round(fm.metrics.mapef(dt_p.loc[dt_p['listed'].isin([2]), model.conf.Y_var], pred[dt_p['listed'].isin([2])]), 4)))\n"
         "print('MAPE test: ' + str(round(fm.metrics.mapef(dt_p.loc[dt_p['listed'].isin([3]), model.conf.Y_var], pred[dt_p['listed'].isin([3])]), 4)))"),

    md('### Model summary'),

    code("print('\\nMain model\\n')\n"
         "print(model.obj.models['main'].summary())\n"
         '\n'
         "metrics_table = fmr.metrics_group_table(dt_pred, group_vars=['superbrand'])"),

    code("print('Grouped metrics\\n')\n"
         'print(metrics_table)'),

    code("if 'model_conf_back' in locals() or 'model_conf_back' in globals():\n"
         '\n'
         "    tmp = pd.merge(model.conf.params[['variable', 'fun', 'arg', 'value']]\n"
         "                    , model.conf_back.params[['variable', 'fun', 'arg', 'value']], on=['variable', 'fun', 'arg'])\n"
         "    tmp = tmp[(tmp['value_x'] != tmp['value_y']) & (tmp['value_x'].notna())]\n"
         '\n'
         "    tmp['diff'] = 0\n"
         "    mask = (tmp['value_y'] != 0) & (pd.to_numeric(tmp['value_y'], errors='coerce').notna())\n"
         "    tmp.loc[mask, 'diff'] = tmp.loc[mask, 'value_x'] / tmp.loc[mask, 'value_y'] - 1\n"
         '\n'
         "    print('Changed params\\n')\n"
         '    tmp'),

    md('### VIF (Multicollinearity)'),

    code("vif_tbl = fmr.stats.vif_table(model.obj.models['main'])\n"
         'vif_tbl'),

    md('## Observed and predicted'),

    code('if if_visual:\n'
         '    \n'
         "    fig = fmr.fit_mult_plot_vol(model, dt_pred, model_rep, period='week', group_var=['superbrand'], show_future=False)\n"
         '    fig.show()'),

    code('if if_visual:\n'
         '    \n'
         "    fig = fmr.fit_mult_plot_vol(model, dt_pred, model_rep, period='week', group_var=['superbrand', 'master', 'form']\n"
         "        , bs_key_filter=dt_p.loc[dt_p['superbrand'] == 'brand_1', 'bs_key'].unique(), show_future=False)\n"
         '    fig.show()'),

    md('## Decomposition and Waterfall'),

    code('split_m_m = None\n'
         '\n'
         'if if_visual:\n'
         '\n'
         '    split_m_m = fmr.extract_effect(model, dt_p, model_rep)\n'
         "    split_m_m = split_m_m.sort_values(['superbrand', 'date'])\n"
         '\n'
         '    fig = fmr.decompose_sub_plot(split_m_m=split_m_m, brands=None, model_rep=model_rep\n'
         "                                    , sku_var=['superbrand'], period='month', show_future=False\n"
         '                                    , contour_line=True)\n'
         '    fig.show()\n'
         '\n'
         "    fig = fmr.waterfall_plot(split_m_m=split_m_m, brands=['brand_1'], model_rep=model_rep\n"
         "                                                    , date_start='2020-01-01', date_end='2023-12-31')\n"
         '    fig.show()'),

    md('## Save or export results'),

    code("pth = ''\n"
         '\n'
         'if if_save:\n'
         '\n'
         "    pth = model.save(dt_p, path=os.path.join(os.getcwd(), 'code_py', 'model_data_sales'))"),


]

write_notebook(sales_cells, 'model_sales.ipynb')

print('\nBoth notebooks created.')
