# Содержание

1.  [Идеология моделирования маркетингового микса](#part_1)

    -   [Постановка задачи, согласование с клиентом основных моментов, специфики и необходимых деталей](#start)

    -   [Сбор и подгрузка данных](#data_collection) 
    
    -   [Генерация дополнительных переменных и предварительный анализ данных](#feature_engineering)
    
    -   [Первичная сборка модели и просмотр результатов](#model_start)
    
    -   [Доработка модели](#tuning)
    
        -   [Ручная доработка модели](#hand_tuning)
        
        -   [Доработка модели с использованием оптимизатора](#optimisation)
        
    -   [Выгрузка и представление результатов в удобном клиенту формате](#results)

2.  [Как пользоваться фреймворком FERMATRICA для моделирования - Пошаговая инструкция](#part_2)


---------------------------------------------------------------
---------------------------------------------------------------

# <a name="part_1"></a>Идеология моделирования маркетингового микса

## <a name="start"></a>Постановка задачи, согласование с клиентом основных моментов, специфики и необходимых деталей

Переписка с клиентской командой и с клиентом, установочные встречи, брифинг.

[К практике - Шаг 1](#step_1)

## <a name="data_collection"></a>Сбор и подгрузка данных

-   подумать, что нужно
-   запросить у клиента
-   запросить у коллег из других подразделений
-   найти самим
-   подгрузить
-   проверить корректность
-   склеить в блоки и далее в единый датасет

Данные могут подгружаться в отдельном скрипте в Python.

Важно, чтобы предварительно все переменные были приведены к единой гранулярности для корректной склейки в единый датасет.

> _Дополнительно:_  
> <small>Если модель панельная, то нужно аккуратно установить соответствие не только между периодами и грануяцией, но и между брендами и суббрендами, регионами или другими группами объектов. Это называется гармонизация</small>

[К практике - Шаг 2](#step_2)

## <a name="feature_engineering"></a>Генерация дополнительных переменных (feature engineering) и предварительный анализ данных

При необходимости

-   создать агрегированные переменные из имеющихся,
-   создать фиктивные (dummy) переменные,
-   создать взвешивающие коэффициенты, веса,
-   очистить одни независимые переменные от влияния других,
-   выделить сезонную составляющую (предварительно очистив целевую от важных факторов, которые могут помешать корректному выделению сезонности)

Построить графики, матрицы парных корреляций и другие необходимые статистики для более полного понимания динамики исследуемых показателей, выявления особенностей в данных и потенциальных ошибок (ещё один этап проверки).

## <a name="model_start"></a>Первичная сборка модели и просмотр результатов

Имея теоретическое представление о структуре будущей модели (гипотезы относительно включения и невключения различных переменных, потенциальных их преобразований) и изучив предварительно ряды данных, оцениваем первичную модель, которую будем считать базовой. Изучаем полученные результаты, чтобы понять, куда двигаться дальше.

Модель FERMATRICA состоит из двух контуров:

1. Внутренний контур - линейная модель, простая или панельная. Линейная модель быстро считается, легко интерпретируется и хорошо исследована статистически. Однако мало какие зависимости в жизни носят линейный характер.

2. Внешний контур - параметризованные нелинейные преобразования независимых переменных, совершаемые до отправки их в качестве фич во внутренний контур, а также преобразования целевой переменной, совершаемые до и после выполнения внутреннего контура.

Двухконтурная схема, в общем следуя стандарту моделирования маркетингового микса, расширяет его и позволяет моделировать сложные зависимости и явления.

[К практике - Шаг 3](#step_3)

Просмотр результатов предполагает детальное изучение модели как с точки зрения статистики и соответствия предпосылкам, так и с точки зрения содержания, смысла исследуемого процесса. Нужно обратить внимание на следующие моменты:

-   саммари по модели (знаки, значимость коэффициентов)
-   график наблюдаемых и прогнозных значений зависимой переменной, график остатков
-   результаты тестов на автокорреляцию, стационарность и нормальность остатков
-   качество прогноза (MAPE и тест на несмещенность прогноза)
-   уровень мультиколлинеарности по VIF
-   декомпозиция (на графике динамики и агрегированная по периодам, вклады переменных)
-   адекватность преобразований (график исхдной и преобразованной переменной, нужно отсмотреть все, чтобы никакие преобразования не искажали смысла, который в них заложен)
-   кривые предельной отдами от медиаинвестиций, кривые ROMI (соотношение кривых, уровень бюджета, на котором значения становятся отличными от нуля и на котором происходит насыщение, адекватность степени пологости)

[К практике - Шаг 4](#step_4)

## <a name="tuning"></a>Доработка модели

[К практике - Шаг 5](#step_5)

### <a name="hand_tuning"></a>Ручная доработка модели

Если есть необходимость, можно внести изменения нужно преобразования переменных или в сами переменные (например, иначе задать некоторый признак), мы можем внести ручные правки в модель (в настройки модели или в исходные данные).

Такие изменения можно вносить и для того, чтобы проверить какую-то гипотезу, например для дальнейшего изменения настроек оптимизатора (например, изменения границ значений гиперпараметров или целесообразности включения переменной, приобразования для переменной).

### <a name="optimisation"></a>Доработка модели с использованием оптимизатора

Параметры преобразований могет быть подобраны не только вручную, но и с помощью численной оптимизации - максимизируется скор - функция, которая отражает качество модели (подбирается с учетом особенностей модели и требований к ней)

## <a name="results"></a>Выгрузка и представление результатов в удобном клиенту формате

В зависимости от формата, предпочитаемого клиентом, могут быть выгружены детали по модели в excel или в презентацию.

Также может быть отдельно разработан дашборд для целей клиента.

[К практике - Шаг 6](#step_6)

------------------------------------------------------------------------
------------------------------------------------------------------------

# <a name="part_2"></a>Как пользоваться фреймворком FERMATRICA для моделирования - Пошаговая инструкция

## <a name="step_1"></a> ШАГ 1

Создаем проект - папку следующей структуры:

-   ***data*** - папка, в которой будут хранится исходные данные для проекта
-   ***code_py*** - папка, в которой будут хранится предобработанные данные, скрипты и результаты моделировани:
    -   **adhoc** - папка, в которой будут жить функции, написанные специально для этого проекта. Может быть пустой
    -   **scripts** - основная рабочая папка, в ней будут жить скрипты для работы над моделью
    -   <a name="data"></a>**data** - папка с подпапкой **data_processed**, там будут жить предобработанные данные, готовые для работы над моделью
    -   **model_data** - папка, в которой будут лежать
        -   файлы с настройками модели (например, [*model_conf.xlsx*](#config))
        -   папки с сохраненными моделями (могут быть в несколько уровней). Папка с моделью (она и есть объект модели) будет создана автоматически после сохранения модели вручную или после работы оптимизатора. Название папки model_YYY-MM-DD_HHmmss состоит из трех файлов:
            -   *dt_p.pkl.zstd* - файл с данными
            -   *model_conf.xlsx* - файл с настройками модели
            -   *model_objects.pkl.lzma* - файл с объектами модели и результатами оценки модели
    -   **results** - папка, в которую можно выгрузить результаты оценки модели в удобном для дальнейшего использования виде
    -   дополнительные файлы, если требуются (например, словари)
-   ***reports*** - папка, в которой будут жить финальные результаты (презентации, калькуляторы и тд для демонстрации и/или отправки заказчику)
-   ***.idea*** - папка, которая отражает, что просто папка стала Python-проектом (создается автоматически при первом открытии папки как проекта в PyCharm)

[Назад к теории - Постановка задачи, согласование с клиентом основных моментов, специфики и необходимых деталей](#start)

## <a name="step_2"></a> ШАГ 2

Начинаем заполнять папки:

-   Исходные данные кладем в папку [data](#data)
-   В папке code_py.scripts или в папке code зоздаем скрипт для подгрузки и проверки данных. Подгружаем и проверяем данные.
-   собираем их в единый <a name="dataset"></a>датасет, который сохраняем в code_py.data.data_processed

[Назад к теории - Сбор и подгрузка данных](#data_collection)

## <a name="step_3"></a>ШАГ 3

Переходим к построению модели. Начинаем с первичного моделирования.

Сборка модели может быть проведена по двум схемам: в Jupyter Notebook (Блок **Load data and current model definition**) или с помощью скрипта model.py.

Линейная модель как без преобразований, так и с ними может быть оценена с помощью FERMATRICA. 

Для оценки модели требуется подготовленный [датасет](#dataset), а также нужно заполнить Excel-файл с настройками (конфигурацией модели). Для этого создаем Excel-файл [model_config.xlsx](#config)

<details>

**<summary>Заполнение файла с <a name="config"></a> настройками модели - model_conf.xlsx (чтобы узнать подробности, нажмите на треугольник)</summary>**


Excel-файл, который содержит следующие листы:

-   ***setup*** - общие настройки модели:

    -   **Y_var** - имя зависимой переменной
    -   **model_type** - тип модели (OLS, LME, LMEA, FE, по умолчанию OLS)
    -   **period_var** - переманная, задающая период
    -   **price_var** - переменная цены, по умолчанию price_distr (если цены неизвестны, необходимо создать переменную price_distr со значением 1, название которой указать здесь)
    -   **bs_key** - название переменной групп объёктов, как правило, бренда - superbrand (если не панель и бренд один, необходимо создать переменную superbrand в датасете с единым значением - названием моделируемого бренда)

    Для панельных моделей и опционально:

    -   **summary_type** - для сводных данных по бюджетным опциям (доступные значения sum - сумма за период и fin - значение на конец периода, по умолчанию sum)
    -   **units** - единицы измерения зависимой переменной, по умолчанию units
    -   **conversion_var** - переменная конверсии (например, если зависимая переменная не продажи, а выводы хотим делать для продаж)
    -   **conversion_fun** - функция конверсии (имя кастомной функции, по которой булет произведен переход от зависимой переменной к интересующей, например, продажам)
    -   <a name="cs"></a>**custom_scoring** - прописанная вручную целевая функция, которая будет использована вместо стандартной взвеженной суммы скоров
    -   **target_audience** - целевая аудитория для ТВ, например, f25-34 (если несколько, то прописываются в несколько строк, тогда далее считываюия в list)
    -   **target_superbrand** - название основного для исследования бренда (если не панель и бренд один, необходимо создать переменную superbrand в датасете с единым значением - названием моделируемого бренда)
    -   **lme_method** - метод оптимизации в statsmodels.formula.api.mixedlm (lbfgs и powell; powell более точен в сложных моделях, не падает, работает относительно медленно; lbfgs - работает относительно быстро, предпочтителен для относительно стандартных, но больших панельных моделей, для сложных может падать)
    -   **fixed_effect_var** - какая переменная отвечает за наличие сущностей (обычно bs_key)
    -   **exclude_from_curves** - переменные, которые хотим исключить из отрисовки в дашборде
    -   **exclude_curve** - переменные, которые хотим исключить из отрисовки в дашборде

-   ***LHS*** - преобразования левой части (обычно нужно для сложных моделей; для простых моделей, как правило, не требуется)

-   ***params*** - преобразования с указанием трансформируемой переменной, функции преобразования, параметров, их границ и стартовых значений, флага включения и других дополнительных настроек:

    -   **variable** - имя переменной, которая будет преобразована
    -   **fun** - имя функции преобразования (базовые заданы в `fermatrica.model.transform`, можно задать свои adhoc функции, тогда имя нужно прописать с путем, например, code_py.adhoc.model.adhoc_tv)
    -   **arg** - аргументы функции преобразования - параметры для оценивания
    -   **lower**, **upper**, **value** - значения параметров: нижнее и верхнее (нужно при оптимизации), текущее (оно же стартовое при оптимизации)
    -   **type** - типа переменной параметра (например, float64, int, bool или str)
    -   **fixed** - флаг на необходимость оптимизировать параметр (1 - если да, 0 - если нет - тогда будет взято текущее значение)
    -   **if_active** - флаг на использование преобразования и параметра (1 - если да, 0 - если нет, тогда в объекте модели не будет этих параметров)
    -   **index_vars** - переменная группировки (по умолчанию группировки не будет)
    -   **index_aggr** - функция группировки (заданы в `fermatrica.model.transform`, возможны следующие: sum_kpi_coef, sum_kpi_coef_master, max, mfreq, по умолчанию группировки нет или sum, если в index_vars не пусто)
    -   **index_free_var** - переменная, по которой будет сделано разделение для применения преобразования (не одинаковое преобразование для всех сущностей, а разные по значениям этой переменной)
    -   **set_start** - если есть формула, то это флаг на её использование для расчета текущего значения по этой формуле
    -   **formula** - формула, по которой будет вычисляться текущее значение параметра (например, на основе переменных в рабочем датасете ds)

-   ***RHS*** - какие перемнные потенциально будут входить в модель с флагами на включение, с указанием ожидаемых знаков и весов, отражающих "важность" переменных в рамках работы оптимизатора

    -   **token** - имя переменной, которая будет включена в линейную модель
        -   если переменная должна быть преобразована, то к имени переменной через "\_" добавляется имя функции преобразования
        -   переменная в линейной модели модет представлять собой комбинацию, тогда она записывается через формулу как I(), например, I(X1 + X2 + X3), где X1, X2, X3 - переменные в датасете
    -   **fixed** - заготовка на будущее, пока везде ставим 1 (нужна для дальнейшего использования, когда будет добавлен функционал по подбору набора переменных, включаемых в модель, в процессе оптимизации)
    -   **if_active** - флаг на включение переменной в модель (1 - если включаем, 0 - иначе)
    -   **sign** - предпочитаемый знак переменной (1 - если переменная должна быть положительной, -1 - если переменная должна быть отрицательной)
    -   **sign_weight** - вес, отражающий относительную важность знака этой переменной (будет учтено в скорах при оптимизации)
    -   **signif_weight** - вес, отражающий относительную важность значимости этой переменной (будет учтено в скорах при оптимизации)
    -   **display_var** - как переменная должна быть названа в анализе и в репортинге, например, дашборже или при создании презентации (по умолчанию дублируется token; если у нескольких переменных одинаковые имена, они будут агрегированы)

-   ***scoring*** - что входит в оптимизируемую целевую функцию, какие скоры и с какими весами (сами скоры задаются в `fermatrica.evaluation.scoring`)

    При оптимизации максимизируем общий скор, для этого

    -   **rmse** - корень из суммы квадратов остатков - чем меньше, тем лучше - if_invert = 1
    -   **r_squared** - коэффициент детерминации, он же R-квадрат - чем больше тем лучше - if_invert = 0
    -   **r_squared_adjusted** - скорректированный коэффициент детерминации, он же R-квадрат-adjusted - чем больше тем лучше - if_invert = 0
    -   **t_value** - t-статистики тестов на значимость коэффициентов регрессии с учетом весов без учета желаемых знаков, в скор идут все случаи - чем больше, тем лучше - if_invert = 0
    -   **p_value** - p-значения тестов на значимость коэффициентов регрессии с учетом весов без учета желаемых знаков, в скор идут все случаи - чем меньше, тем лучше - if_invert = 1
    -   **sign_simple** - соответствие желаемым знакам с учетом весов, в скор идут случаи несоответствия - чем меньше, тем лучше - if_invert = 1
    -   **sign_t_value** - t-статистики тестов на значимость коэффициентов регрессии с учетом весов и знаков, в скор идут случаи несоответствия знаков желаемым - чем меньше, тем лучше - if_invert = 1
    -   **vif** - среднее значение мер мультиколлинеарности - чем меньше, тем лучше - if_invert = 1
    -   **durbin_watson_score** - статистика теста Дарбина-Уотсона на автокорреляцию в остатках за вычетом 2 (!!! важно, что не durbin_watson) - чем меньше, тем лучше - if_invert = 1

> _Дополнительно:_
> -    <small>Если задан кастомный скоринг в [**custom_scoring**](#cs), его компоненты должны быть заданы на этом листе и активны.</small>
> -    <small>Если хотим перевернуть весь скор (invert), прописываем это при вызове функции оптимизации.</small>

-   ***trans_path_df*** - этот лист будет создан автоматически после первого прогона оптимизатора или может быть исходно создан.

    На этом листе для каждой переменной задается

    -   **variable** - исходная переменная из датасета ds, которая используется в модели,
    -   **variable_fin** - преобразованная переменная, которая используется в модели
    -   **price** - цена за единицу медиа воздействия для каждого медийного канала для корректного отображений кривых (если переменная немедийная, то ячейка с ценой должна быть пустая),
    -   **display_var** - имя переменной, как она должна быть названа в дашборде или при создании презентации.

</details>
<br/>

<details>

Каждое преобразование, заданное на вкладке `params`, создаёт в датасете новую переменную, имя которой составляется из имени исходной переменной и имени функции преобразования, соединённых нижних подчёркиванием. Например, преобразование `softmax` над переменной `tv` создаёт переменную `tv_softmax`.

Преобразования выполняются последовательно.

На той же вкладке задаются коэффициенты для преобразований левой части, они отличаются пустым полем `variable` и значением `coef` в поле `fun`. Коэффициенты применяются в порядке выполнения преобразований левой части.

**<summary>Основные функции, которые используются для преобразований переменных в модели (чтобы узнать подробности, нажмите на треугольник)</summary>**

Функции задаются на листе `params` файла `model_conf.xlsx`.

#### Временные преобразования

***lag()***: лагирование (сдвиг во времени) переменной, запаздывание

-    **n** - число лагов (на сколько периодов сдвиг)
-    **cval** - значение, которым будут заполнены пропуски в начале, которые образуются при сдвиге. По умолчанию 0

***mar()***: правое сглаживание методом скользящих средних

-    **n** - число лагов (на сколько периодов сдвиг)
-    **if_scale** - нужно ли взвешивать сглаженную переменную на среднее значение в трейне + тесте

- ***adstock(), adstockp()***: геометрический адсток (запаздывание, помогает моделировать отложенный эффект). Первый вариант - взвешенный, с суффиксом 'p' - без взвешивания ("pure")

-    **a** - decay rate (какая доля воздействия в период t будет иметь эффект только в t+1)

***adstockpd()***: линейная комбинация двух адстоков с разной скоростью затухания.

-    **a1** - decay rate первого адстока
-    **a2** - decay rate второго адстока
-    **w2** - коэффициент для второго адстока при составлении линейной комбинации (от 0 до 1). Первый адсток взвешивается, соответственно, на `1 - w2`

***dwbl(), dwblp()***: затухание / временное преобразование на базе распределения Вейбулла (второй вариант - "чистый", устойчив к добавлению новых наблюдений в ряду)

-    **dur** - длительность эффекта в периодах (округляется до целого)
-    **shape** - параметр, определяющий форму кривой (shape < 1 — С-образная кривая, shape > 1 — S-образная)
-    **scale** - коэффициент масштаба
-    **norm** - флаг нормализации результата, по умолчанию False

#### Насыщение

***expm1(), expm1scaled()***: экспоненциальное преобразование с вычитанием 1 из результата и взвешиванием с медианой и стандартным отклонением (второй вариант)

-    **product** - множитель для исходной переменной перед взятием экспоненты, делает кривую более гладкой или более пологой

***softmax(), softmaxfull()***: разновидность логистического преобразования (сигмоиды), первый вариант - с вычетом нуля по иксу (чтобы при 0 по X не было значения, отличного от 0, по Y)

-    **lambda** - параметр крутизны преобразования, меньше - круче
-    **avg** - точка перегиба (скорость прироста начинает снижаться)
-    **std** - стандартное отклонение исходной переменной, используется для очищения лямбды от размерности исходной переменной

***logistic()***: разновидность логистического преобразования (сигмоида)

-    **steep** - параметр крутизны преобразования
-    **ec50** - параметр достижения половины уровня насыщения
-    **level** - уровень насыщения (используется для очищения остальных параметров от размерности исходной переменной)

- ***adbudg()***: в зависимости от значений аргументов, приближается к степенной или сигмоидальной форме

-    **steep** - параметр крутизны преобразования
-    **ec50** - параметр достижения половины уровня насыщения
-    **level** - уровень насыщения (используется для очищения остальных параметров от размерности исходной переменной)

...И ещё полдесятка реже используемых функций

#### Комплексные преобразования

***arl(), arlp()***: преобразование, совмещающее насыщение, адсток и запаздывание. В первом случае насыщение задаётся функцией `adbudg()`, во втором - `logistic()`.

-    **a** - decay rate
-    **steep** - параметр крутизны преобразования
-    **ec50** - параметр достижения половины уровня насыщения
-    **level** - уровень насыщения (используется для очищения остальных параметров от размерности исходной переменной)
-    **n_lag** - число лагов (на сколько периодов сдвиг)
-    **cval** - при наличии лага, значение, которым заполняются края смещенного вектора

!   В функциях `arl()` и `arlp()` сначала идёт насыщение, потом адсток, потом сдвиг (лаг)

***price()***: комплексное преобразование с расчётом цен относительно средних по группе в момент времени.

</details>
<br/>

Далее загружаем первичную модель и оцениваем её без использования оптимизатора. Это делается в три шага.

1. **Загрузка модели.** Создаём объект `fermatrica.Model`, считывая `model_conf.xlsx` при инициализации объекта. При более сложной структуре модели на этом же этапе аргументами передаётся adhoc-код и функция построения категориальной модели.
    ```python
    import os
    import fermatrica as fm
   
    pth = os.path.join("code_py", "data", "model_conf.xlsx")
    model = fm.Model(path=pth
                     , custom_predict_fn=None
                     , adhoc_code=[code_py.adhoc.model]
                     , ds=dt_p)
    ```

2. **Выполнение трансформаций внешнего контура.** Построение модели и получение прогноза делается в два шага при помощи одной и той же пары функций. Это необходимо для улучшения производительности у сложных моделей. Сначала выполняются преобразования внешнего контура. Аргумент `set_start` запускает расчёт начальных значений параметров (где таковой задан), относитесь к нему аккуратно.
    ```python
    model = fm.transform(ds=dt_p
                         , model=model
                         , set_start=True
                         , if_by_ref=True)
    ```

3. **Фит внутреннего контура и получение прогноза.** Когда преобразования выполнены, запускаем внутренний контур. На этом шаге строится внутренняя линейная модель, а также туда и обратно совершаются преобразования левой части. Можно запустит `fermatrica.fit_predict()` два раза с разными аргументами - сначала будет построена модель и получен прогноз на трейне, потом получим прогноз уже на всём датасете. Функция `fermatrica.predict_ext()` возвращает прогноз в виде не серии, а фрейма с дополнительными столбцами, удобными для анализа
    ```python
    pred, pred_raw, model = fm.fit_predict(dt_p, model, if_fit=True, return_full=True)

    dt_pred = fm.predict_ext(model, dt_p)
    ```

> _Дополнительно:_  <small>Пример можно посмотреть здесь: [примеры скоро будут готовы]</small>

[Назад к теории - Первичная сборка модели и просмотр результатов](#model_start)

## <a name="step_4"></a> ШАГ 4

Просматриваем результаты, сохраняем, если они нас удовлетворяют

Если хотим сохранить модель, используем метод `fermatrica.model.save` (для неё нужно указать путь к папке, в которую будут сохраняться результаты).

Просмотр результатов возможен

-   в Jupyter Notebook (Блоки **Get results**, **Statistic and metrics**, **Observed and predicted**, **Decomposition and Waterfall**), если модель строится в нём
-   в том же скрипте model.py с использованием написанных в FERMATRICA функций расчета нужных статистик и графиков (тесты при необходимости, график наблюдаемых и прогнозных значений зависимой переменной, график остатков, качество пронноза, декомпозиция). Тогда модель можно не сохранять и продолжать работу над ней, меняя настройки в конфиге и запуская скрипт с оценкой модели
-   в дашборде FERMATRICA_DASH, если модель сохранена

Каждый способ имеет свои плюсы и минусы, исследователь сам может выбрать предпочтительный для него путь.

[Назад к теории - Первичная сборка модели и просмотр результатов](#model_start)

## <a name="step_5"></a> ШАГ 5

Работаем над моделью

### Меняем параметры и состав переменных вручную

-   Вносим правки в список переменных на листе ***RHS*** в [конфиге](#config) (добавляем новые или "включаем"/"выключаем" уже приписанные)
-   Добавляем/убираем/меняем преобразования на листе ***params***  и в названии переменных, включенных модель на листе ***RHS*** в [конфиге](#config)
-   Меняем значения параметров преобразований в столбце **value** на листе ***params*** в [конфиге](#config)

Оцениваем модель заново, анализируем результаты.

Может проводиться в отдельном скрипте .py или в рамках доработки модели в Jupyter Notebook или в скрипте model.py.

> _Дополнительно:_  <small>Преобразований к одной переменной может быть применено несколько последовательно. Для этого нужно на листе params указать порядок: преобразование *a* накладывается на преобразованную уже переменную X с помощью *b*, тогда в params нужно прописать сначала *b* для X, а потом *a* для X_b, а в модель будет входить X_b_a, что и нужно прописать в **token**</small>

[Назад к теории - Доработка модели](#tuning)

### Используем оптимизатор для подбора значений параметров

Параметры преобразований могут быть подобраны с использованием численной оптимизации. 

По умолчанию целевая функция (скор) формируется как линейная комбирация метрик, заданных в [конфиге](#config) на листе ***scoring*** в столбце **metrics** и столбце **if_active** (нужные должны быть "включены"), с весами **weight**.
Также целевая функция может быть задана иначе вручную в том жа файле в [**custom_scoring**](#cs).

Поддерживаются локальные и глобальные алгоритмы, на данный момент это локальный алгоритм COBYLA, вызываемый функцией `fermatrica.optimize_local_cobyla()`, и генетический алгоритм PyGAD, вызываемый функцией `fermatrica.optimize_global_ga()`.

<details>

**<summary>Настройки оптимизатора (чтобы узнать подробности, нашмите на треугольник)</summary>**

Для локального алгоритма COBYLA (***local***):

-   *revert_score_sign* - изменять ли знак итогового скора? Меняем, если наш скор максимизируется, потому что COBYLA минимизирует
-   *verbose* - выводить ли диагностическую информацию
-   *epochs* - число эпох - количество раз, которое будет запускаться оптимизатор. В локальном оптимизаторе следующая эпоха наследует предыдущей, поэтому количество оптимальных моделей не зависит от количества эпох и всегда равно 1. Смысл нескольких эпох - "встряхивание" алгоритма (shuffle) для лучшего избегания локальных минимумов
-   *iters_epoch* - максимальное число итераций работы оптимизатора в рамках одной эпохи
-   *error_score* - скор в случае ошибки. Ставим какое-нибудь огромное число, в случае минимизирующего алгоритма - со знаком плюс
-   *ftol_abs* - порог прироста оптимизируемой функции, после которого алгоритм останавливается. COBYLA обращается с ним довольно своенравно, если видит потенциал дальнейшей оптимизации (склон), но не вовсе игнорирует.

Для генетического алгоритма PyGAD (***global***):

-   *revert_score_sign* - изменять ли знак итогового скора? Меняем, если наш скор минимизируется, потому что PyGAD максимизирует
-   *verbose* - выводить ли диагностическую информацию
-   *epochs* - число эпох - количество раз, которое будет запускаться оптимизатор, оно же - количество моделей, которое будет оценено на выходе. В глобальном оптимизаторе эпохи независимы, и в конце каждой сохраняется найденный оптимальный вариант
-   *iters_epoch* - максимальное число итераций работы оптимизатора в рамках одного запуска
-   *pop_size* - размер популяции для генетического алгоритма (исходное число особей-моделей на старте работы алгоритма)
-   *pmutation* - вероятность мутации (случайных изменений в особях-моделях) при генетического алгоритма
-   *max_no_improvement* - максимальное число итераций, при котором нет регистрируемого повышения скора
-   *error_score* - скор в случае ошибки. Ставим какое-нибудь огромное число, в случае максимизирующего алгоритма - со знаком минус
-   *cores* - число логических ядер компьютера, которые могут быть задействованы для расчетов
-   *save_epoch* - путь к папке, в которую сохранять результаты после каждой эпохи

</details>
<br/>

> _Дополнительно:_  <small>Пример можно посмотреть здесь: [coming soon]</small>

Повторяем ШАГ 4 и ШАГ 5, пока не получим результат, который полностью нас удовлетворит.


## <a name="step_6"></a> ШАГ 6

Выгрузка результатов для клиента возможна: 

-   в презентацию PowerPoint
-   в дашборд

> _Дополнительно:_  
> - <small>Дашборд существует в виде отдельного репозитория FERMATRICA_DASH</small>
> - <small>Генератор слайдов входит в состав репозитория FERMATRICA_REP</small>

[Назад к теории - Выгрузка и представление результатов в удобном клиенту формате](#results)

В папке output готовим отчет по модели.



------------------------------------------------------------------------
------------------------------------------------------------------------
