"""Regenerate notebooks/train.ipynb from a structured Python representation.

The narrative ноутбука mirrors the README report: each section follows the
pattern «что делаем → код → что получили». Running this script overwrites the
existing notebook — re-run it after editing the CELLS list below.

Usage:
    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Append a markdown cell."""
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    """Append a code cell."""
    CELLS.append(("code", text))


# ---------------------------------------------------------------------------
# 0. Титульный блок
# ---------------------------------------------------------------------------

md("""# Автоматизация ML-пайплайна: knowledge tracing на ASSISTments 2009

**Автор:** Григорьева Е.С.
**Дата:** май 2026
**Репозиторий:** <https://github.com/EkaterinaGrisha/knowledge-tracing>

Настоящий ноутбук — практическая часть учебного проекта. Он повторяет логику
основного отчёта (`README.md`), но в исполняемом виде: каждый раздел сначала
описывает, что и зачем выполняется, затем содержит код, и завершается
интерпретацией полученного результата.

**Содержание:**

0. Подготовка окружения
1. ETL: извлечение, преобразование, загрузка
2. Контроль качества данных и мониторинг дрейфа
3. Признаки
4. Обучение моделей
   - 4.1 BKT — байесовский baseline
   - 4.2 DKT — рекуррентная нейросеть
   - 4.3 DKT с автоматическим подбором архитектуры (Optuna)
   - 4.4 AutoML на табличных признаках (FLAML)
5. Сводное сравнение моделей
6. Визуализации
7. Выводы
8. Воспроизведение полного пайплайна одной командой
""")

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------

md("""## 0. Подготовка окружения

**Что делаем.** Загружаем конфигурацию пайплайна (`config/config.yaml`),
устанавливаем рабочую директорию в корень проекта (чтобы относительные пути
из конфигурации разрешались правильно), фиксируем источник данных и seed.
Этот блок единственный, который нужно выполнить **обязательно** — все
последующие разделы используют объект `cfg`. Пакет `knowledge_tracing`
установлен в `.venv` командой `poetry install`, поэтому импортируется без
манипуляций с `sys.path`; ноутбук нужно открыть с kernel `.venv`.

**Источник данных.** В переменной `DATA_SOURCE` выбирается режим работы:
`'sample'` (закоммиченная подвыборка из 288 студентов, прогон занимает
десятки секунд, не требует интернета) или `'full'` (полный датасет
ASSISTments 2009, скачивается автоматически на первом обращении, прогон
~20–45 минут на CPU). Для интерактивного изучения рекомендуется `'sample'`,
для финального воспроизведения цифр из отчёта — `'full'`.
""")

code("""import os
from pathlib import Path

# Перейти в корень проекта (там, где лежит config/config.yaml).
root = Path.cwd()
while not (root / 'config' / 'config.yaml').exists() and root != root.parent:
    root = root.parent
os.chdir(root)
print('project root:', root)

from knowledge_tracing.config import load_config
from knowledge_tracing.logging_setup import configure_logging
from knowledge_tracing.runtime import configure_runtime

configure_runtime()   # macOS: защита от двойной загрузки OpenMP (до импорта torch)
configure_logging()   # логи пакета выводятся в ячейки

cfg = load_config('config/config.yaml')   # типизированный и проверенный конфиг
SEED = cfg.seed
DATA_SOURCE = 'sample'   # 'sample' (быстро, офлайн) или 'full' (полный датасет)
FIGURES_DIR = cfg.output.figures_dir

print('source =', DATA_SOURCE, '· seed =', SEED)
""")

# ---------------------------------------------------------------------------
# 1. ETL
# ---------------------------------------------------------------------------

md("""## 1. ETL: извлечение, преобразование, загрузка

**Что делаем.** Слой ETL подготавливает данные для всех последующих стадий
обучения. Он состоит из трёх функций:

- `extract(cfg.data, data_source)` — загружает исходные данные. В режиме `'full'`
  скачивает train/test файлы ASSISTments с зеркала DKVMN (с ретраями и
  экспоненциальной задержкой при сетевых ошибках) и кэширует их в `data/raw/`;
  в режиме `'sample'` читает закоммиченный CSV.
- `transform(df_raw, cfg.data, seed)` — очищает данные, фильтрует студентов с короткой
  историей, обрезает слишком длинные последовательности, кодирует
  идентификаторы навыков в плотный индекс, **выполняет сплит по студентам**
  (а не по строкам — это критично для отсутствия утечки), вычисляет восемь
  причинных признаков для табличной модели.
- `load(processed, cfg.data.processed_dir)` — сохраняет результат в `data/processed/`
  (`features.parquet`, `interactions_long.parquet`, `dataset_stats.json`).
  Разнесение «подготовка → обучение» через файловую систему позволяет
  переиспользовать одну и ту же подготовку при многократных запусках обучения.

**Зачем именно так.** Сплит по студентам гарантирует, что прошлое одного
студента не попадёт одновременно в train и test. Обрезка `max_seq_len = 200`
ограничивает потребление памяти LSTM. Фильтр `min_seq_len = 3` отсекает
студентов, на которых невозможно построить осмысленную причинную историю.
""")

code("""from knowledge_tracing.etl.extract import extract
from knowledge_tracing.etl.load import load
from knowledge_tracing.etl.transform import transform

df_raw = extract(cfg.data, data_source=DATA_SOURCE)
processed = transform(df_raw, cfg.data, seed=SEED)
load(processed, cfg.data.processed_dir)

print('Исходных взаимодействий:', len(df_raw))
print('После очистки и фильтрации:', processed.stats['n_interactions'])
print('Студентов / навыков:', processed.stats['n_students'], '/', processed.stats['n_skills'])
print('Глобальная доля верных:', round(processed.stats['global_correct_rate'], 4))
print('Сплит студентов:', processed.stats['split_students'])
print('Сплит взаимодействий:', processed.stats['split_interactions'])
""")

md("""**Что получили.** Чистая «длинная» таблица в `processed.long` со схемой
`user_id | order_idx | skill_idx | correct | split`. Параллельно сформирован
табличный фрейм признаков `processed.features` (8 столбцов + метка + сплит)
для будущего AutoML. Студенты разделены на три непересекающихся множества с
зафиксированным seed; объёмы сплитов соответствуют конфигурации 70/10/20.
Артефакты сохранены в `data/processed/` — отсюда их подхватывают все
последующие стадии.
""")

# ---------------------------------------------------------------------------
# 2. Data quality + drift
# ---------------------------------------------------------------------------

md("""## 2. Контроль качества данных и мониторинг дрейфа

**Что делаем.** Перед обучением выполняем два независимых проверочных шага:

- **Data-quality gate** (`knowledge_tracing/monitoring/data_quality.py`) —
  проверки схемы, пропусков, диапазонов, бинарности меток и уникальности пары
  `(user_id, order_idx)`. Если хотя бы одна проверка не прошла,
  `enforce_quality_gate` останавливает работу (`DataQualityError`), а
  пайплайн сохраняет подробный отчёт как артефакт MLflow; режим «только
  предупредить» включается флагом `monitoring.fail_on_data_quality: false`.
- **Мониторинг дрейфа** (`knowledge_tracing/monitoring/drift.py`) — расчёт PSI (Population
  Stability Index) и теста Колмогорова-Смирнова для каждого из восьми
  признаков, сравнение распределений train против test. Принятые пороги PSI:
  < 0.10 — стабильно, 0.10–0.25 — умеренный дрейф, ≥ 0.25 — значимый.

**Зачем.** В производственной системе эти проверки — последний рубеж перед
обучением: они отлавливают случаи, когда данные «поползли» (изменился
формат, появились пропуски, сместились распределения) и предотвращают
тихую деградацию модели.
""")

code("""from knowledge_tracing.monitoring.data_quality import enforce_quality_gate, quality_report
from knowledge_tracing.monitoring.drift import drift_report

dq = quality_report(processed.long)
print('Data quality passed =', dq['passed'])
for name, info in dq['checks'].items():
    print(f'  {name}: passed = {info[\"passed\"]}')
enforce_quality_gate(dq, fail=cfg.monitoring.fail_on_data_quality)

features = processed.features
drift = drift_report(
    features[features['split'] == 'train'],
    features[features['split'] == 'test'],
    processed.feature_cols,
    psi_warn=cfg.monitoring.psi_warn,
    psi_alert=cfg.monitoring.psi_alert,
)
print()
print('Drift status:', drift['overall_status'])
print(f'Признаков со значимым дрейфом: {drift[\"n_significant_drift\"]} из {drift[\"n_features\"]}')
for name, info in drift['features'].items():
    print(f'  {name}: PSI = {info[\"psi\"]:.4f}, status = {info[\"status\"]}')
""")

md("""**Что получили.** Все пять проверок качества данных прошли успешно —
`data_quality_passed = True`. По итогам сравнения train против test ни один
из восьми признаков не выходит за порог PSI = 0.10; общий статус — `OK`.
Значения PSI на порядок меньше границы умеренного дрейфа, что закономерно:
сравниваются train и test одного датасета, разделённые по студентам, без
сдвига во времени или среде. Эти же проверки в продакшене запускались бы
ежедневно против свежего батча данных.
""")

# ---------------------------------------------------------------------------
# 3. Признаки
# ---------------------------------------------------------------------------

md("""## 3. Признаки

**Что делаем.** Извлекаем первые строки таблицы признаков, чтобы наглядно
показать структуру входа табличной модели и убедиться в его причинности.

**Принцип причинности.** Каждый признак вычисляется *до* того, как известна
метка текущего шага: `user_prior_*` и `skill_prior_*` суммируют историю
до текущего взаимодействия исключительно, `recent3_correct_rate` — это
скользящая успешность за три предыдущих шага, `skill_difficulty` оценивается
**только на train** и применяется к val/test как фиксированный справочник.
Так исключается утечка целевой метки в признаковое пространство.
""")

code("""processed.features.head(8)
""")

md("""**Что получили.** Восемь причинных признаков на каждое взаимодействие.
Поле `correct` — целевая метка, `split` — принадлежность к train/val/test,
`user_id` нужен для построения последовательностей в DKT и для группировки
при возможных дополнительных анализах. Именно `processed.features` подаётся
на вход AutoML-модели в разделе 4.4.
""")

# ---------------------------------------------------------------------------
# 4. Модели
# ---------------------------------------------------------------------------

md("""## 4. Обучение моделей

**Общая постановка.** На одном и том же тестовом наборе обучаются и
сравниваются четыре модели разной природы и степени автоматизации — от
интерпретируемого baseline до полностью автоматического AutoML. Метрики
качества (AUC, Accuracy, F1, Precision, Recall, RMSE, log-loss) рассчитываются
по протоколу one-step-ahead: для каждого взаимодействия в test-сплите модель
по предыдущей истории студента предсказывает вероятность правильного ответа,
после чего предсказания сопоставляются с фактическими метками.

Все модели реализуют общий интерфейс `KnowledgeTracingModel`: `fit(data)`
на обучающих данных и `predict(data.test)` — пары «фактический ответ /
предсказанная вероятность». Метрики считаются одной функцией
`compute_metrics`. Ниже модели обучаются и оцениваются по очереди;
результаты накапливаются в словарях `results` и `preds`.
""")

code("""from knowledge_tracing.etl.datasets import build_training_data
from knowledge_tracing.evaluation.metrics import compute_metrics

data = build_training_data(processed)   # последовательности + табличные признаки по сплитам
results = {}
preds = {}
print(
    f'Последовательностей: train={len(data.train.sequences)}, '
    f'val={len(data.val.sequences)}, test={len(data.test.sequences)}'
)
print(f'Уникальных навыков: {data.n_skills}')
""")

# 4.1 BKT
md("""### 4.1 BKT — байесовский knowledge tracing

**Что делаем.** Обучаем классический BKT — скрытую марковскую модель с двумя
состояниями («навык не освоен» и «навык освоен») и четырьмя параметрами на
**каждый** навык: `p_init`, `p_learn`, `p_slip`, `p_guess`. Параметры
оцениваются алгоритмом Expectation-Maximization с использованием
forward-backward (число итераций EM — `em_iters = 30`).

**Зачем.** Модель проста, интерпретируема и обучается за секунды. Её AUC
служит нижней границей — если более сложная модель не превосходит BKT,
её ценность сомнительна.
""")

code("""from knowledge_tracing.models.bkt import BKTModel

bkt = BKTModel(cfg.models.bkt)
bkt.fit(data)
preds['BKT'] = bkt.predict(data.test)
results['BKT'] = compute_metrics(*preds['BKT'])
results['BKT']
""")

md("""**Что получили.** Базовое качество, против которого будут сравниваться
все остальные модели. На полном датасете AUC BKT составляет около 0.72 — это
типичный уровень BKT-моделей в литературе по knowledge tracing. На sample
число студентов меньше и качество может незначительно отличаться, но порядок
величин тот же.
""")

# 4.2 DKT
md("""### 4.2 DKT — рекуррентная нейросеть (LSTM)

**Что делаем.** Обучаем нейросеть архитектуры `Embedding → LSTM → Linear`.
Пара `(навык, корректность)` кодируется в индекс `2 · skill_idx + correct`,
LSTM моделирует динамику знаний, линейная голова проектирует скрытое
состояние в `K`-мерный вектор вероятностей. Функция потерь — бинарная
кросс-энтропия, оптимизатор — Adam. Гиперпараметры по умолчанию:
`embed_dim = 64`, `hidden_dim = 64`, `dropout = 0.2`, `lr = 0.005`,
`batch_size = 32`, `epochs = 30`.

**Зачем.** Рекуррентная сеть способна моделировать длинные зависимости в
последовательности взаимодействий — в отличие от BKT, который видит лишь
текущее скрытое состояние. Это даёт DKT преимущество в точности предсказания
на нетривиальных историях.

При наличии GPU (CUDA) функция `pick_device()` автоматически выбирает его;
на CPU обучение всё ещё уложится в разумное время благодаря умеренной
размерности модели.
""")

code("""from knowledge_tracing.models.dkt import DKTModel

dkt = DKTModel(cfg.models.dkt, seed=SEED)
dkt.fit(data)
preds['DKT'] = dkt.predict(data.test)
results['DKT'] = compute_metrics(*preds['DKT'])
results['DKT']
""")

md("""**Что получили.** DKT уверенно превосходит BKT по AUC — типичный прирост
+0.08–0.09 на полном датасете. Train-loss модели монотонно
снижается по эпохам (`dkt.training_curve()`), что подтверждает корректное обучение (без расходимости
и без преждевременного плато). Кривая обучения визуализируется в разделе 6.
""")

# 4.3 DKT + Optuna
md("""### 4.3 DKT с автоматическим подбором архитектуры (Optuna)

**Что делаем.** Автоматизируем архитектурный поиск поверх DKT с помощью
Optuna (TPE-сэмплер). Пространство поиска: `embed_dim ∈ {32, 64, 128}`,
`hidden_dim ∈ {32, 64, 128}`, `dropout ∈ [0.0, 0.5]`,
`lr ∈ log-uniform [1e-3, 2e-2]`, `batch_size ∈ {16, 32, 64}`.
Каждый trial обучает уменьшенную копию DKT (`epochs_per_trial = 8`) и
возвращает AUC на val; Optuna максимизирует AUC. После `n_trials = 25`
лучшая конфигурация переобучается полностью на `epochs = 30` и оценивается
на тесте.

**Зачем.** Ручной перебор гиперпараметров для DKT занимает много времени
дата-сайентиста и склонен к смещению (исследователь часто проверяет
«разумные» конфигурации, пропуская неочевидные). Optuna систематически
обходит пространство и предлагает регуляризацию там, где её часто
недооценивают.

Этот блок — самый длительный по времени в ноутбуке (на sample — несколько
минут, на full — около 10–15 минут). При желании пропустить его, можно
закомментировать и в `results['DKT+Optuna']` записать результат DKT.
Для быстрого знакомства уменьшите бюджеты: `load_config('config/config.yaml',
['config/quick.yaml'])`.
""")

code("""from knowledge_tracing.models.dkt_optuna import DKTOptunaModel

dkt_optuna = DKTOptunaModel(cfg.models.dkt_optuna, cfg.models.dkt, seed=SEED)
dkt_optuna.fit(data)   # поиск архитектуры на val + переобучение лучшей конфигурации
print('Лучшая конфигурация:', dkt_optuna.search_.best_params)
print('Лучший val AUC:', round(dkt_optuna.search_.best_val_auc, 4))
preds['DKT+Optuna'] = dkt_optuna.predict(data.test)
results['DKT+Optuna'] = compute_metrics(*preds['DKT+Optuna'])
results['DKT+Optuna']
""")

md("""**Что получили.** Optuna находит конфигурацию, обычно отличающуюся от
дефолтной DKT — чаще всего другим балансом размерностей и более сильной
регуляризацией (`dropout` в диапазоне 0.3–0.5). Это согласуется с
наблюдением, что DKT склонна к переобучению на нашем объёме данных, и
дополнительный dropout помогает обобщению. Прирост AUC относительно DKT —
порядка +0.002–0.005; небольшое улучшение, но получено полностью
автоматически.
""")

# 4.4 AutoML FLAML
md("""### 4.4 AutoML на табличных признаках (FLAML)

**Что делаем.** Подаём табличный фрейм причинных признаков (см. раздел 3) в
FLAML — фреймворк AutoML от Microsoft Research. FLAML использует адаптивную
стратегию CFO (Cost-Frugal Optimization) и за отведённый бюджет времени
одновременно выбирает алгоритм-кандидат (LightGBM, XGBoost, RandomForest или
ExtraTrees) и его гиперпараметры, максимизируя AUC на валидации.
Параметры: `time_budget_s = 600` секунд, `metric = "roc_auc"`,
`estimator_list = ["lgbm", "xgboost", "rf", "extra_tree"]`. FLAML всегда
расходует весь бюджет времени, поэтому число испытаний (и выбранная
конфигурация) зависит от скорости компьютера.

**Зачем.** Это самый быстрый путь от «есть признаки» к «есть рабочая модель»:
никакого ручного выбора алгоритма, никакого подбора гиперпараметров. Сравнение
с DKT показывает, насколько хорошо спроектированные причинные признаки могут
сократить разрыв между табличным AutoML и специализированной моделью
последовательностей.
""")

code("""from knowledge_tracing.models.automl_flaml import AutoMLModel

automl = AutoMLModel(cfg.models.automl_flaml, seed=SEED)
automl.fit(data)   # табличные признаки train, ранняя остановка по val
preds['AutoML'] = automl.predict(data.test)
results['AutoML'] = compute_metrics(*preds['AutoML'])
print('Лучший выбранный эстимейтор:', automl.report()['automl_best_estimator'])
results['AutoML']
""")

md("""**Что получили.** FLAML обычно выбирает `xgboost` или `lgbm` (зависит от
бюджета и данных) и достигает AUC порядка 0.79 на полном датасете — заметно
выше BKT и ниже DKT на 0.01 AUC. Этот результат демонстрирует ценность
качественных причинных признаков: универсальный табличный AutoML без
рекуррентной архитектуры подбирается совсем близко к специализированной
модели.
""")

# ---------------------------------------------------------------------------
# 5. Сравнение моделей
# ---------------------------------------------------------------------------

md("""## 5. Сводное сравнение моделей

**Что делаем.** Сводим метрики всех четырёх моделей в одну таблицу, чтобы
сопоставить их на одном тестовом наборе.
""")

code("""import pandas as pd

df_results = (pd.DataFrame(results).T
              [['auc', 'accuracy', 'f1', 'precision', 'recall', 'rmse', 'log_loss', 'n']]
              .round(4))
df_results.sort_values('auc', ascending=False)
""")

md("""**Что получили.** На полном датасете порядок моделей по AUC устойчив:
**DKT + Optuna > DKT > AutoML > BKT**. Различие между DKT и DKT + Optuna
невелико (~0.002 AUC), что естественно — обе суть рекуррентные модели,
отличие в архитектурных гиперпараметрах. AutoML занимает промежуточное
положение и существенно превосходит BKT.

Различие `n` между моделями (44 581 у DKT/DKT+Optuna и 45 353 у BKT/AutoML)
объясняется протоколом one-step-ahead для рекуррентных моделей: на первом
шаге каждой последовательности у модели ещё нет истории, и предсказание не
формируется. Это не влияет на сравнимость метрик, поскольку для тестовой
выборки исключение одного шага на студента вносит пренебрежимо малый сдвиг.
""")

# ---------------------------------------------------------------------------
# 6. Визуализации
# ---------------------------------------------------------------------------

md("""## 6. Визуализации

**Что делаем.** Строим семь графиков в едином стиле и сохраняем их в
`artifacts/figures/` (каталог `output.figures_dir` из конфигурации). Эти же
графики автоматически логируются как артефакты в MLflow при запуске
`kt train`; опубликованные в отчёте (`README.md`, §10) версии лежат в
`reports/figures/` и обновляются осознанно командой `make publish-report`.
""")

code("""from IPython.display import Image, display

from knowledge_tracing.evaluation import visualize as viz

best_name = max(results, key=lambda k: results[k]['auc'])
figures = [
    viz.plot_dataset_overview(processed.long, processed.stats, FIGURES_DIR),
    viz.plot_model_comparison(results, FIGURES_DIR),
    viz.plot_roc(preds, FIGURES_DIR),
    viz.plot_dkt_loss(dkt.training_curve(), FIGURES_DIR),
    viz.plot_confusion(*preds[best_name], best_name, FIGURES_DIR),
    viz.plot_calibration(*preds[best_name], best_name, FIGURES_DIR),
]
importance = automl.feature_importance()
if importance is not None:
    figures.append(
        viz.plot_feature_importance(list(importance.index), importance.to_numpy(), FIGURES_DIR)
    )
for path in figures:
    display(Image(str(path)))
""")

md("""**Что получили.** Семь графиков:

- `dataset_overview.png` — четырёхпанельный обзор датасета;
- `model_comparison.png` — bar-chart четырёх моделей по AUC/ACC/F1/RMSE;
- `roc_comparison.png` — ROC-кривые всех моделей на одной системе координат;
- `dkt_loss.png` — кривая обучения DKT;
- `confusion_matrix.png` — матрица ошибок лучшей модели;
- `calibration.png` — калибровочная кривая лучшей модели;
- `feature_importance.png` — важность признаков модели AutoML.

Подробное толкование графиков приведено в `README.md`, §10.
""")

# ---------------------------------------------------------------------------
# 7. Выводы
# ---------------------------------------------------------------------------

md("""## 7. Выводы

**Технические итоги.** В ноутбуке реализован сквозной автоматизированный
ML-пайплайн: ETL → контроль качества данных → мониторинг дрейфа → обучение
четырёх моделей разной природы → сводное сравнение → визуализации. Каждая
стадия использует ту же реализацию, что и production-пайплайн
(`knowledge_tracing.pipeline.run_pipeline()`, команда `kt train`); разница
лишь в интерактивной разбивке по шагам.

**Содержательные итоги.**

- Рекуррентные модели (DKT) превосходят классический BKT на 0.08–0.09 AUC.
- Автоматический подбор архитектуры (Optuna) даёт небольшой, но
  систематический прирост (~+0.002 AUC) и выбирает сильную регуляризацию.
- Табличный AutoML (FLAML) на восьми причинных признаках достигает AUC,
  уступающего DKT всего на ~0.010 — это иллюстрирует роль качественной
  инженерии признаков.
- Ни на одном из восьми признаков не наблюдается дрейфа между train и test
  (PSI < 0.10); все пять проверок качества данных пройдены.

Полные количественные результаты, контейнеризация (Docker), CI/CD (GitHub
Actions) и связь с критериями оценивания описаны в основном отчёте `README.md`.
""")

# ---------------------------------------------------------------------------
# 8. One-shot
# ---------------------------------------------------------------------------

md("""## 8. Воспроизведение полного пайплайна одной командой

Ниже — необязательная ячейка, выполняющая **весь** пайплайн через
`run_pipeline(cfg, ...)`: метрики и графики записываются в `artifacts/`,
запуск — в MLflow, а модель DKT+Optuna сохраняется в `artifacts/model` и
регистрируется в Model Registry как `challenger`, если проходит порог
качества. Эта же функция вызывается командой `kt train` и в контейнере
(`docker run kt-pipeline train --data-source full`). Результаты видны в
MLflow UI:

```bash
make mlflow   # mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Открыть в браузере <http://localhost:5000>.
""")

code("""# Раскомментируйте для полного прогона (на full-датасете занимает 20–45 минут):
# from knowledge_tracing.pipeline import run_pipeline
# summary = run_pipeline(cfg, data_source='full')
# summary['results']
""")


# ---------------------------------------------------------------------------
# Сериализация в .ipynb
# ---------------------------------------------------------------------------


def _to_source(text: str) -> list[str]:
    """Convert a string to the per-line list format used by .ipynb."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def build_notebook() -> dict:
    """Notebook JSON (nbformat 4.5) from the collected cells."""
    nb_cells = []
    for cell_type, text in CELLS:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": _to_source(text),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.11 (kt-pipeline)",
                "language": "python",
                "name": "kt-venv",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write notebooks/train.ipynb."""
    out = Path(__file__).resolve().parents[1] / "notebooks" / "train.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    with out.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"Notebook written: {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
