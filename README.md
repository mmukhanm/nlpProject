# Анализ тональности отзывов о товарах маркетплейса

Учебный Python-проект на **реальных отзывах Amazon о телефонах и аксессуарах**. Программа: **недели 1–6, затем 8–13**, каждый этап отдельно. Неделя 7 намеренно пропущена.

**Текущий статус:** классический блок выполнен, приложение работает, отчёт и презентация собраны. Код Attention, frozen BERT и fine-tuning реализован, но реальные BERT-эксперименты пока **не выполнены**: облачная сеть запрещает `huggingface.co` (HTTP 403). Случайная модель и вымышленные метрики не используются. См. [статус выполнения](reports/execution_status.json).

## Что получилось

- 1000 исходных реальных отзывов UCI Amazon Cells → 982 после нормализованной дедупликации.
- 589 train / 196 validation / 197 test; seed=42, стратификация, одинаковые split для всех методов.
- BOW, TF-IDF, Word2Vec, GloVe, fastText; 6 классических классификаторов.
- По validation выбран **BOW + Multinomial Naive Bayes**: test accuracy **0.802**, macro-F1 **0.802**.
- Streamlit: текст → тональность, вероятности, диаграмма, JSON. Пять настоящих [скриншотов](reports/screenshots) и [результаты UI-проверок](reports/ui_checks.json).
- [Итоговый отчёт](reports/FINAL_REPORT.md), [таблица всех выполненных опытов](reports/all_metrics.csv), [разбор 10 ошибок](reports/error_analysis.csv), [презентация на 10 слайдов](reports/presentation.pptx).

Язык отзывов **английский**; документация, notebooks и UI — на русском. Корпус содержит два класса (positive/negative), без neutral. Это фрагменты отзывов о мобильной технике, не полный корпус современных маркетплейсов. [Источник, лицензия и ограничения](data/DATA_CARD.md).

## Быстрый запуск

Нужен **Python 3.12**. Из корня репозитория:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
python -m pip install --no-deps -e .
streamlit run app.py --browser.gatherUsageStats false
```

Классические модели уже сохранены в `models/`, их можно использовать без повторного обучения. Загружайте joblib/Gensim-модели только из доверенного источника: эти форматы содержат исполняемые Python-объекты. Зависимости для них зафиксированы в `requirements.lock`; несовместимые версии scikit-learn могут не прочитать сериализованные модели.

В подготовленной облачной среде:

```bash
cd /workspace/nlpProject
source /workspace/venvs/nlp/bin/activate
streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

## Воспроизведение исследования

Все команды выполняются **из корня репозитория**. Набор данных включён в Git; скачивание повторно проверяет SHA-256.

```bash
export PYTHONHASHSEED=42
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
python scripts/download_data.py
python -m nlp_project.classical
python -m scripts.build_report
python -m pytest -q
python -m scripts.run_notebooks
```

Последняя команда выполняет notebooks 1–6, 11–13; notebook 11 показывает baseline-ошибки и явно отмечает отсутствие BERT. Notebooks 8–10 до загрузки весов сохраняются **без outputs**. Их статус фиксируется в `reports/notebook_execution.json`.

### BERT: оставшийся блок

Нужен доступ к Hugging Face и хранилищам весов. Необходимо разрешить `huggingface.co`, `cdn-lfs.huggingface.co`, `cdn-lfs-us-1.huggingface.co`, `cas-bridge.xethub.hf.co`. В облаке требования сохранены в черновике настроек, но сохранение черновика само по себе не применяет сетевую политику. После применения доступа:

```bash
python -m nlp_project.transformer
python -m scripts.build_report
python -m pytest -q
python -m scripts.run_notebooks --all
```

Используется **настоящий pretrained `prajjwal1/bert-tiny`**, а не случайно инициализированный encoder: 2 слоя, hidden=128, CPU. Это компактный вариант BERT, а не полный BERT-base. Конкретная загруженная ревизия будет записана в `reports/bert_source.json`. Замороженный encoder + обученный LR — **feature extraction**, не zero-shot sentiment: у базового BERT нет sentiment-головы.

Fine-tuning: AdamW, batch=16, max_length=96, 4 эпохи, clipping=1, weight_decay=0.01. План: LR 2e-5 / 3e-5 / 5e-5 при 100% train; затем 50% / 75% / 100% train при LR=3e-5. Валидация выбирает checkpoint и лучшую модель. Test не используется для подбора. Не ожидайте гарантированного улучшения после fine-tuning — это проверяемая гипотеза.

После BERT перезапустите приложение, чтобы сбросить кэш выбранной модели, и заново снимите UI-скриншоты, если выбор модели изменился.

### Проверки UI и скриншоты

Сначала запустите Streamlit. В другом терминале:

```bash
python scripts/capture_ui.py --browser /usr/bin/chromium
```

Скрипт использует системный Chromium через Playwright. Если браузер не установлен, установите Chromium средствами своей ОС и передайте путь через `--browser`. Скриншоты не создаются искусственно. Пять контрольных входов — написанные для демонстрации тексты, **не часть реального корпуса** и не независимая оценка accuracy. Пятый смешанный отзыв baseline классифицирует неверно; это сохранено без исправления результата.

## Этапы и материалы

| Неделя | Реализация | Notebook | Результат / статус |
|---|---|---|---|
| 1 | Окружение, загрузка, статистика | [01](notebooks/week_01.ipynb) | `corpus_stats.json`, реальные данные; выполнено |
| 2 | Токены, стоп-слова, нормализация, слитный текст | [02](notebooks/week_02.ipynb) | `preprocessing.csv`, `glued_text.csv`; выполнено |
| 3 | Леммы, POS, Morph, Porter/Snowball | [03](notebooks/week_03.ipynb) | `morphology.csv`, top-50, сравнение; выполнено |
| 4 | BOW / TF-IDF, топ-слова, облако | [04](notebooks/week_04.ipynb) | CSV и графики; выполнено |
| 5 | Word2Vec / GloVe / fastText | [05](notebooks/week_05.ipynb) | Векторы, соседи, арифметика; выполнено |
| 6 | Классификация и PCA | [06](notebooks/week_06.ipynb) | 6 baseline-моделей, метрики; выполнено |
| 8 | 16 текстов, attention и контекстные embeddings | [08](notebooks/week_08.ipynb) | Код готов; выполнение заблокировано загрузкой весов |
| 9 | Frozen pretrained BERT + LR | [09](notebooks/week_09.ipynb) | Код готов; выполнение заблокировано |
| 10 | Fine-tuning BERT | [10](notebooks/week_10.ipynb) | Код и параметры готовы; обучение заблокировано |
| 11 | LR, объём train, полный набор метрик, 10 ошибок | [11](notebooks/week_11.ipynb) | Baseline-ошибки выполнены; BERT-опыты ожидают доступ |
| 12 | Streamlit, 5 контрольных отзывов | [12](notebooks/week_12.ipynb) | Работает с лучшим выполненным baseline; 5 скриншотов |
| 13 | Итоговый notebook, отчёт, 10 слайдов | [13](notebooks/week_13.ipynb) | Материалы отражают фактический статус; личная защита — автором |

Исходный код — `nlp_project/`, orchestration — `scripts/`, тесты — `tests/`. Notebooks создаются воспроизводимо через `python -m scripts.make_notebooks`; эта команда **пересоздаёт файлы и очищает outputs**, не запускайте её поверх собственных правок.

## Как предотвращается утечка

Нормализованные точные повторы удаляются **до** разбиения; предложения с конфликтующими метками исключаются. Словарь/IDF, собственные embeddings и классификаторы обучаются только на train. Validation используется для выбора; test — для отчёта. Разбор ошибок не применялся для переобучения. Если использовать эти ошибки для улучшения, потребуется новый независимый test. Поскольку UCI не даёт ID товара/пользователя, разделение по товарам невозможно. Семантические дубликаты не исключены.

Стоп-слова удаляются с сохранением отрицаний. spaCy выполняет морфологию **в исходном контексте**. Slit-text исследуется отдельно и не применяется ко всем словам. Word2Vec/fastText обучены на малом train-корпусе; GloVe/BERT используют внешнее предобучение, поэтому бюджеты обучения неравны. fastText: bucket=20 000 для умеренной памяти на маленьком корпусе.

## Подготовка к защите

1. Впишите ФИО и группу на первом слайде презентации.
2. Покажите карточку данных и объясните разбиение train/validation/test.
3. Сравните частотные и векторные методы по фактической таблице.
4. Для BERT сначала проверьте статус выполнения: не объявляйте код завершённым экспериментом.
5. Продемонстрируйте позитивный, негативный и смешанный отзыв, затем разберите ошибку и ограничения вероятностей.

Код — MIT. Корпус — отдельные условия UCI CC BY 4.0, с обязательной атрибуцией. Внешние pretrained-модели имеют собственные лицензии; ссылки в [карточке данных](data/DATA_CARD.md).
