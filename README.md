# FERMATRICA Flexible Econometrics Framework. Optimizer

[_Russian version below_](#RU)

The core of the FERMATRICA econometrics framework - modeling.

### 1. Key Ideas

FERMATRICA (pronounced as _"Fermatritsa"_) allows the construction of time series models and panel models with a focus on marketing mix modeling. The core functionality can also be applied to tasks from other subject areas.

The approach used separates the model description into two contours:

1. Inner contour - generally a linear model.
2. Outer contour - transformations that occur before building/executing the linear model and/or after it.

By splitting the model into two parts, we avoid the issue of adjusting parameters/coefficients for a complex nonlinear model directly. Nonlinearity is moved to a separate contour with appropriate algorithms, effectively representing parameterized feature engineering. The inner contour retains the transformed variables with included nonlinearity, allowing for the use of simple and fast linear algorithms like OLS/LME.

The specificity of FERMATRICA is a broader interpretation of the outer contour. The classical approach allows building models with only one type of dependency of the target variable on factors - either additive or multiplicative. In FERMATRICA, you can add multiplicative factors to additive ones by specifying them in the LHS transformations. Such additional factors will be optimized using the outer contour alongside the transformation parameters.

In addition to multiplicative factors, other transformations can be added to the outer contour, including auxiliary models for correcting both Y and X.

Among other advantages of FERMATRICA are wide and flexible support for panel models, availability of global and local search algorithms for the outer contour, modularity (new algorithms can be included as development progresses), and a broad range of supported transformations.

From a technical standpoint, FERMATRICA employs a mixed approach with extensive use of standalone functions and limited use of OOP, allowing for a balance between code performance, data consistency, and usability.

### 2. Components

The repository includes the core of the FERMATRICA framework - modules responsible for building and executing the model.

- Model description and execution (`fermatrica.model`)
  - Model description (`fermatrica.model.model`, `fermatrica.model.model_conf`, `fermatrica.model.model_obj`)
  - Building and executing the inner contour (`fermatrica.model.predict`)
  - Left-hand side transformations (`fermatrica.model.lhs_fun`)
  - Right-hand side transformations (`fermatrica.model.transform`, `fermatrica.model.transform_fun`)
- Metrics and scoring (`fermatrica.evaluation`, `fermatrica.scoring`)
- Optimization of outer contour parameters (`fermatrica.optim`)

### 3. Installation

To facilitate work, it is recommended to install all components of the FERMATRICA framework. It is assumed that work will be conducted in PyCharm (VScode is OK also for sure).

1. Create a Python virtual environment of your choice (Anaconda, Poetry, etc.) or use a previously created one. It makes sense to establish a separate virtual environment for econometric tasks and for every new version of FERMATRICA.
    1. Mini-guide on virtual environments (external): https://blog.sedicomm.com/2021/06/29/chto-takoe-venv-i-virtualenv-v-python-i-kak-ih-ispolzovat/
    2. For framework version v010, let the virtual environment be named FERMATRICA_v010.
2. Clone the FERMATRICA repositories to a location of your choice.
    ```commandline
    cd [my_fermatrica_folder]
    git clone https://github.com/FERMATRICA/fermatrica_utils.git 
    git clone https://github.com/FERMATRICA/fermatrica.git
    git clone https://github.com/FERMATRICA/fermatrica_rep.git 
    ```
   1. To work with the interactive dashboard: _coming soon_
   2. For preliminary data work: _coming soon_
3. In each of the repositories, select the FERMATRICA_v010 environment (FERMATRICA_v020, FERMATRICA_v030, etc.) through Add interpreter in the PyCharm interface and switch to the corresponding git branch.
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    git checkout v010 [v020, v030...]
    ```
4. Install all cloned packages except FERMATRICA_DASH using pip install.
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    pip install .
    ```
   1. Instead of navigating to each project's folder, you can specify the path to it in pip install:
       ```commandline
       pip install [path_to_fermatrica_part]
       ```
5. If necessary, install third-party packages/libraries required for the functioning of FERMATRICA using `conda install` or `pip install`. To update versions of third-party packages, use `conda update` or `pip install -U`.

> FERMATRICA actively uses the XLSX format for storing settings and parameters as it is the most user-friendly option. No complex XLSX formatting is provided, so theoretically, the core should work on operating systems other than Windows, both technically and in terms of user comfort, but functionality on operating systems other than Windows has not been tested.

------------------------------------
<a name="RU"></a>
Ядро эконометрического фреймворка FERMATRICA - моделирование.

### 1. Ключевые идеи

FERMATRICA (произносится как _"Ферматрица"_) позволяет строить модели временного ряда и панельные модели с упором на моделирование маркетингового микса. Ядерный функционал также может быть применён к задачам из других предметных областей.

Используется подход, при котором описание модели разделяется на два контура:

1. Внутренний контур - как правило, линейная модель
2. Внешний контур - преобразования, совершающиеся перед построением / выполнением линейной модели и/или после неё

Разбивая модель на две части, мы избегаем проблемы с подбором параметров / коэффициентов для сложной нелинейной модели в лоб. Нелинейность выносится в отдельный контур с соответствующими алгоритмами, представляющий собой фактически параметризованный feature engineering. Во внутреннем контуре остаются преобразованные переменные с уже включённой нелинейностью, что позволяет использовать простые и быстрые линейные алгоритмы типа OLS / LME.

Специфика FERMATRICA - более широкая трактовка внешнего контура. Классический подход позволяет строить модели только с одним типом зависимости целевой переменной от факторов - либо аддитивной, либо мультипликативной. В FERMATRICA вы можете добавить к аддитивным факторам мультипликативные, указав его в преобразованиях LHS. Такие дополнительные факторы будут оптимизироваться при помощи внешнего контура, наряду с параметрами преобразований.

Помимо мультипликативных факторов, во внешний контур могут быть добавлены дополнительные преобразования, включая вспомогательные модели для коррекции как Y, так и X.

Среди других преимуществ FERMATRICA - широкая и гибкая поддержка панельных моделей, доступность алгоритмов глобального и локального поиска для внешнего контура, модульность (по мере развития можно будет подключать новые алгоритмы), широкий набор поддерживаемых преобразований.

С технической точки зрения FERMATRICA использует смешанный подход с широким использованием отдельных функций и ограниченным использованием ООП, что позволяет найти баланс между производительностью кода, согласованностью данных и удобством использования.

### 2. Состав

Репозиторий включает ядро фреймворка FERMATRICA - модули, отвечающие за построение и выполнение модели.

- Описание и выполнение модели (`fermatrica.model`)
  - описание модели (`fermatrica.model.model`, `fermatrica.model.model_conf`, `fermatrica.model.model_obj`)
  - построение и выполнение внутреннего контура (`fermatrica.model.predict`)
  - преобразования левой части (`fermatrica.model.lhs_fun`)
  - преобразования правой части (`fermatrica.model.transform`, `fermatrica.model.transform_fun`)
- Метрики и скоринг (`fermatrica.evaluation`, `fermatrica.scoring`)
- Оптимизация параметров внешнего контура (`fermatrica.optim`)

### 3. Установка

Для удобной работы рекомендуется установить все составляющие фреймворка FERMATRICA. Предполагается, что работа будет вестись в PyCharm или VsCode.

1. Создайте виртуальную среду Python удобного для вас типа (Anaconda, Poetry и т.п.) или воспользуйтесь ранее созданной. Имеет смысл завести отдельную виртуальную среду для эконометрических задач и для каждой новой версии FERMATRICA
   - Мини-гайд по виртуальным средам (внешний): https://blog.sedicomm.com/2021/06/29/chto-takoe-venv-i-virtualenv-v-python-i-kak-ih-ispolzovat/
   - Для версии фреймворка v010 пусть виртуальная среда называется FERMATRICA_v010
2. Клонируйте в удобное для вас место репозитории FERMATRICA
    ```commandline
    cd [my_fermatrica_folder]
    git clone https://github.com/FERMATRICA/fermatrica_utils.git 
    git clone https://github.com/FERMATRICA/fermatrica.git
    git clone https://github.com/FERMATRICA/fermatrica_rep.git 
    ```
    - Для работы с интерактивным дашбордом FERMATRICA_DASH: _coming soon_
    - Для предварительной работы с данными FERMATRICA_DATA: _coming soon_
3. В каждом из репозиториев выберите среду FERMATRICA_v010 (FERMATRICA_v020, FERMATRICA_v030 и т.д.) через `Add interpreter` (в интерфейсе PyCharm) и переключитесь в соответствующую ветку гита
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    git checkout v010 [v020, v030...]
    ```
4. Установите все склонированные пакеты, кроме FERMATRICA_DASH, используя `pip install`
    ```commandline
    cd [my_fermatrica_folder]/[fermatrica_part_folder]
    pip install .
    ```
   - Вместо перехода в папку каждого проекта можно указать путь к нему в `pip install`
   ```commandline
   pip install [path_to_fermatrica_part]
   ```
5. При необходимости, поставьте сторонние пакеты / библиотеки Python, которые требуются для функционирования FERMATRICA, используя `conda install` или `pip install`. Для обновления версий сторонних пакетов используйте `conda update` или `pip install -U`

> FERMATRICA активно использует формат XLSX для хранения настроек и параметров как наиболее удобный для работы человека. Какого-либо сложного форматирования XLSX не предусматривается, поэтому в теории ядро должно работать в операционных системах, отличных от Windows, как технически, так и в плане удобства работы пользователя, но работосопособность в операционных системах, отличных от Windows, не тестировалась. 

