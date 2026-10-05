# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версии —
[Semantic Versioning](https://semver.org/lang/ru/).

## [2.0.0] — подготовка проекта к production

Мажорная версия: изменились имя пакета, способ установки и команды запуска.

### Добавлено

- Управление зависимостями через **Poetry**: `pyproject.toml` с группами
  (`dev`, `notebook`, `presentation`), `poetry.lock` с точными версиями,
  `poetry.toml` (окружение `.venv/` внутри проекта), `.python-version`.
- **pre-commit**: гигиена файлов, ruff (линтер и форматтер), mypy, nbstripout,
  `poetry check --lock`, запрет коммитов в `main`, быстрые тесты на `pre-push`.
- Команда **`kt`**: `etl`, `train` (`--quick`, `--override`), `predict`,
  `promote`; `python -m knowledge_tracing`.
- Типизированная **конфигурация** (pydantic) с проверкой при загрузке;
  оверлей `config/quick.yaml`.
- Общий интерфейс моделей `KnowledgeTracingModel` и реестр моделей.
- **Упаковка и регистрация модели**: DKT+Optuna сохраняется в
  `artifacts/model`, логируется в MLflow как pyfunc-модель и при test AUC не
  ниже `serving.min_test_auc` регистрируется как `challenger`;
  `kt promote` переводит её в `champion`, не допуская ухудшения. Модель
  логируется по схеме MLflow «models from code» (без pickle-объекта).
- **MODEL_CARD.md** с оценкой по подгруппам (`scripts/sliced_metrics.py`).
- **BPMN-процесс** жизненного цикла и согласования модели
  (`docs/model_lifecycle.md`, `docs/bpmn/`), шаблон Pull Request, `CODEOWNERS`.
- Регрессионная проверка `scripts/compare_metrics.py` и эталон
  `docs/baseline/`.
- Тесты конфигурации, инференса, реестра, CLI (включая сквозной прогон
  train → регистрация → predict → promote); порог покрытия 90 % в CI.

### Изменено

- Код перенесён из пакета `src` в устанавливаемый пакет
  `knowledge_tracing` (src-layout).
- `pipeline.run()` разбит на стадии; модели обрабатываются единым циклом.
- Модели возвращают только предсказания; метрики считаются в одном месте.
- Прогоны пишут результаты в `artifacts/`; опубликованные результаты в
  `reports/` обновляются командой `make publish-report`.
- Логирование через логгеры модулей; импорт пакета не меняет глобальных
  настроек matplotlib, seaborn и optuna.
- MLflow хранит запуски и реестр в SQLite (`mlruns/mlflow.db`).
- Dockerfile — многоэтапная сборка на Poetry, точка входа `kt`; CI запускает
  pre-commit, тесты с покрытием и smoke-прогон образа; actions обновлены.
- Ноутбук использует API пакета и хранится без выводов ячеек.

### Исправлено

- `mlflow.tracking_uri` из конфигурации игнорировался.
- Data-quality gate не останавливал пайплайн при провале проверок.
- Лучшая модель не сохранялась, и ею нельзя было воспользоваться.
- Каждый прогон перезаписывал закоммиченные результаты в `reports/`.
- Недостижимая функция `write_sample` и неверная подсказка при отсутствии
  sample-датасета.
- Неточности в документации: обрезка истории оставляет первые 200 попыток;
  диапазон learning rate в поиске Optuna — [1e-3, 2e-2].

### Удалено

- `requirements.txt`, `pytest.ini`, `scripts/install_torch_cpu.sh`,
  `scripts/run_pipeline*.sh` (заменены Poetry, `pyproject.toml` и `kt`).

## [1.0.0] — исходная версия

Автоматизированный ML-пайплайн knowledge tracing на ASSISTments 2009:
ETL, BKT, DKT, DKT+Optuna, FLAML AutoML, мониторинг, MLflow, Docker, CI.
Состояние зафиксировано тегом `v1.0.0-baseline`.
