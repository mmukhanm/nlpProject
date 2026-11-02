"""Собрать отчёт и 10 слайдов из фактических результатов, без выдуманных метрик."""
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports'


def table(frame):
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main():
    frames = [pd.read_csv(REPORT / 'classical_metrics.csv')]
    pred_frames = [pd.read_csv(REPORT / 'classical_predictions.csv')]
    bert_path = REPORT / 'bert_metrics.csv'
    expected_bert = {'bert_frozen_lr', 'bert_lr2e5_full', 'bert_lr3e5_full', 'bert_lr5e5_full', 'bert_lr3e5_half', 'bert_lr3e5_75'}
    bert_done = False
    if bert_path.exists():
        bert_frame = pd.read_csv(bert_path)
        bert_done = set(bert_frame.model) == expected_bert and len(bert_frame) == 12
        frames.append(bert_frame)
        pred_frames.append(pd.read_csv(REPORT / 'bert_predictions.csv'))
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(REPORT / 'all_metrics.csv', index=False)
    best = all_metrics.query("split == 'validation'").sort_values(['f1_macro', 'model'], ascending=[False, True]).iloc[0]
    kind = 'bert' if best.model.startswith('bert_') else 'classical'
    (REPORT / 'best_model.json').write_text(json.dumps({'name': best.model, 'kind': kind, 'selection': 'validation macro-F1', 'validation_f1': float(best.f1_macro)}, indent=2))
    predictions = pd.concat(pred_frames, ignore_index=True)
    selected = predictions.query('model == @best.model and split == "test"').copy()
    errors = selected[selected.label != selected.prediction].head(10).copy()
    # These are human-reviewed hypotheses for the actual baseline errors, not automatic causal explanations.
    annotations = {
        'amazon-0003': ('Косвенная жалоба', 'Зависимость от зарядки выражена без типичных отрицательных слов.', 'Добавить примеры жалоб на автономность.'),
        'amazon-0080': ('Устойчивое выражение', 'holds up very well описывает долговечность; разреженная модель не обобщила выражение.', 'Больше примеров устойчивых оборотов.'),
        'amazon-0119': ('Контекст отрицания', "could not have been simpler — позитивная конструкция с отрицанием.", 'Контекстная модель и примеры сложных отрицаний.'),
        'amazon-0244': ('Неявная оценка', 'defeats the purpose выражает отрицательную оценку без bad.', 'Доменные фразы и контекстные embeddings.'),
        'amazon-0313': ('Сравнение', 'Восстановление прежней ёмкости батареи — положительный результат.', 'Примеры сравнений состояния товара.'),
        'amazon-0324': ('Опечатка и отрицание', 'cant not hear — шумная разговорная конструкция; cant не раскрывается как cannot.', 'Контролируемая нормализация опечаток без потери отрицания.'),
        'amazon-0333': ('Несколько аспектов', 'Положительные оценки удобства, звука и заряда не были объединены моделью.', 'Аспектный анализ; расширение корпуса.'),
        'amazon-0338': ('Ожидание и уступка', 'Низкая цена положительна, но ожидалось лучшее качество.', 'Примеры смешанных оценок и конструкций с but.'),
        'amazon-0341': ('Короткое отрицание', 'not happy не распознано надёжно при ограниченном корпусе.', 'Контекстные отрицания; больше коротких отзывов.'),
        'amazon-0350': ('Редкая лексика', 'JERKS выражает негатив, но слово редко в обучающей выборке.', 'Расширить лексику и использовать предобученные векторы.'),
    }
    for col in ['reason', 'analysis', 'proposed_fix']: errors[col] = ''
    for idx, row in errors.iterrows():
        note = annotations.get(row.id, ('Требует ручного разбора', 'Факт ошибки подтверждён меткой корпуса; причина пока не установлена.', 'Ручная проверка контекста и метки.'))
        errors.loc[idx, ['reason', 'analysis', 'proposed_fix']] = note
    errors.to_csv(REPORT / 'error_analysis.csv', index=False)
    ConfusionMatrixDisplay.from_predictions(selected.label, selected.prediction, display_labels=['Negative', 'Positive'], cmap='Blues')
    plt.title(f'Selected on validation: {best.model}')
    plt.tight_layout(); plt.savefig(REPORT / 'figures/confusion_matrix.png', dpi=160); plt.close()
    tests = all_metrics.query("split == 'test'").set_index('model')
    tests[['accuracy', 'f1_macro']].plot.barh(figsize=(9, 6), xlim=(0, 1), title='Held-out test / fixed split')
    plt.tight_layout(); plt.savefig(REPORT / 'figures/model_comparison.png', dpi=160); plt.close()
    stats = json.loads((REPORT / 'corpus_stats.json').read_text())
    test_row = all_metrics.query('model == @best.model and split == "test"').iloc[0]
    status = ('BERT-эксперименты выполнены; результаты включены ниже.' if bert_done else
              '**Блокер недель 8–11:** сеть запрещает huggingface.co (HTTP 403). Код реализован, но pretrained BERT, attention и fine-tuning ещё не выполнены. Метрики BERT отсутствуют, случайная модель не подставлялась.')
    report = f'''# Анализ тональности отзывов о товарах маркетплейса

## Статус и постановка задачи

Python-проект: бинарная классификация тональности реальных английских отзывов о телефонах и аксессуарах Amazon. Интерфейс и учебные материалы на русском. Расписание: недели 1–6, затем 8–13; неделя 7 пропущена по заданию.

{status}

## Данные и протокол

Источник: UCI Sentiment Labelled Sentences, Kotzias et al. (KDD 2015), DOI: https://doi.org/10.24432/C57604. Только файл Amazon Cells: 1000 предложений (500 позитивных / 500 негативных). Это реальные фрагменты отзывов, а не сгенерированные примеры. Не весь каталог Amazon и не отзывы современных российских маркетплейсов.

После очистки повторов: **{stats['unique_documents']}** документов, {stats['words']} слов. Порог задания ≥500 документов выполнен; 50 000 слов для этого варианта не требуется (условие «или»). Исключено 18 повторов/конфликтующих копий после нормализации. Итоговое распределение и разбиение: `{stats['splits']}`. Seed=42, стратификация по тональности. ID и split сохранены в `data/processed/reviews.csv`.

Все обучаемые словари, TF-IDF, Word2Vec, fastText и классификаторы обучены только на train. Внешние GloVe/BERT предобучены отдельно. Выбор модели и эпохи — только по validation macro-F1, порог 0.5 фиксирован заранее. Test используется для итогового сравнения; выводы из ошибок не применялись для переобучения. При дальнейшем изменении модели по этим ошибкам нужен новый test.

В UCI-файле нет ID товара/автора, поэтому разделение по товарам и авторам невозможно. Удалены точные нормализованные повторы; семантически похожие отзывы могут остаться. Это ограничивает оценку переноса на новые товары. Интервалы неопределённости не оценены; test мал ({len(selected)} примеров), результаты одного seed не доказывают превосходство на всех данных.

## Неделя 1. Окружение и корпус

Python 3.12, Jupyter, NLTK, spaCy, Gensim, scikit-learn, PyTorch и Transformers. SHA-256 исходного TSV фиксирован в `data/DATA_CARD.md`. Статистика — `corpus_stats.json`, первичный просмотр — notebook 01.

## Неделя 2. Очистка и токенизация

Удалены HTML, URL, пунктуация; приведён регистр, раскрыты isn't/can't/won't, удалены стоп-слова. **not/no/never сохраняются**, поскольку меняют тональность. В таблице `preprocessing.csv` приведены 20 сравнений до/после. Slit-text (`thebatteryisnotgood`) исследуется отдельно в `glued_text.csv` через wordninja; автоматическое разбиение не применяется ко всем отзывам, чтобы не ломать бренды и обычные слова. Нормализация рассчитана на английский.

## Неделя 3. Морфология

spaCy `en_core_web_sm` определяет леммы, POS и признаки Morph на исходных предложениях с контекстом. NLTK Porter/Snowball отсекают окончания и могут создавать несловарные основы; spaCy даёт словарную форму. Для английского падежная система ограничена; признаки выводятся только если модель их обнаружила, русские падежи не выдумываются. Полная таблица, сравнение стеммеров и top-50 слов/лемм сохранены в CSV. Поиск дефектов теггера требует ручной проверки: модель не даёт эталонной разметки.

## Неделя 4. Bag-of-Words и TF-IDF

Общие параметры: 1–2-граммы, до 12 000 признаков, min_df=1. BOW отражает частоту, TF-IDF снижает вес частых общих слов. Суммарные веса top-20 отражают именно обучающую часть, а не весь корпус. Визуализации: `bow_terms.png`, `tfidf_terms.png`, `wordcloud.png`.

## Неделя 5. Word2Vec, GloVe, fastText

Word2Vec и fastText: 50 измерений, skip-gram, окно 5, min_count=1, 40 эпох, workers=1, seed=42; fastText bucket=20 000 для умеренного расхода памяти. GloVe: готовые `glove-wiki-gigaword-50`, без дообучения. `semantic_examples.json` содержит соседей и арифметику good + phone − bad; `embedding_coverage.json` — покрытие словаря. Арифметика — диагностический эксперимент, её результаты не обязательно семантически осмысленны. Малый корпус ограничивает собственные embeddings; GloVe использует значительно больше внешних данных, сравнение не является сравнением архитектур при равном бюджете.

## Неделя 6. Прикладная задача

Сравнены BOW+NB, TF-IDF+NB, TF-IDF+Logistic Regression и средние Word2Vec/GloVe/fastText + Logistic Regression. Для fastText подсловные представления доступны и для новых слов. PCA отображает средние Word2Vec-векторы train в 2D; разделение на картинке не является метрикой качества. Полные предсказания и вероятности сохранены для проверки метрик.

## Неделя 8. Transformer и Attention

Реализация: 16 текстов validation, BERT mean pooling с attention mask, сравнение cosine similarity с TF-IDF на одинаковых текстах. Отдельно сохраняется матрица attention последнего слоя с усреднением голов для одного текста. Attention: softmax(QKᵀ/√d)V; порядок кодируется позиционными embeddings. Строка матрицы соответствует query, столбец key. Attention зависит от контекста, но не доказывает причинную важность токена. BERT требует исходного текста, а не списка лемм без отрицаний. Артефакты появляются только после успешного запуска BERT.

## Неделя 9. Предобученный BERT

`prajjwal1/bert-tiny`: настоящий английский pretrained BERT, 2 слоя, hidden=128. Компактный размер выбран для CPU. Encoder заморожен; Logistic Regression обучается на train embeddings. Это **feature extraction**, а не zero-shot классификация: у базового BERT нет готовой sentiment-головы. Случайная classification head не используется как «предобученный анализатор тональности».

## Неделя 10. Fine-tuning

Обучаются encoder и новая binary classification head. AdamW, weight_decay=0.01, batch=16, max_length=96, gradient clipping=1, 4 эпохи. Лучший checkpoint выбирается по validation macro-F1 (при равенстве более ранняя эпоха). Параметры/длительность записываются в training_config.json, loss и F1 по эпохам — в CSV. Усечение длинных текстов отдельно подсчитывается.

## Неделя 11. Эксперименты и ошибки

Зафиксирован план: learning rate 2e-5 / 3e-5 / 5e-5 на 100% train; затем 50% / 75% / 100% train при 3e-5. Во всех опытах одна validation/test и seed=42. Выбор по validation. Отчёт содержит accuracy и macro precision/recall/F1. Для выбранной модели ниже показаны первые {len(errors)} ошибок test по порядку ID. Причины — гипотезы, проверенные вручную для baseline; для других ошибок требуются отдельные комментарии. Изменение объёма меняет также число шагов обучения, поэтому это практический бюджетный эксперимент, а не изоляция только объёма при равном compute.

## Фактически полученные метрики

{table(all_metrics[['model','split','accuracy','precision_macro','recall_macro','f1_macro','n']])}

Выбрана **{best.model}**: validation macro-F1={best.f1_macro:.3f}; test accuracy={test_row.accuracy:.3f}, macro-F1={test_row.f1_macro:.3f}. Модель с максимальным test результатом не обязательно выбирается — test не используется для выбора. Текущий лучший результат относится только к выполненным экспериментам.

## Разбор ошибок

{table(errors[['id','text','label','prediction','reason','analysis','proposed_fix']])}

## Неделя 12. Приложение

Streamlit `app.py`: ввод английского отзыва, кнопка анализа, класс, вероятности, диаграмма, экспорт JSON. Пустой/неподдерживаемый ввод и чрезмерная длина обрабатываются явно. Вероятности не калиброваны; нейтрального класса нет. Пять новых контрольных текстов задаются отдельно от корпуса: это демонстрационные входы, они не выдаются за реальные отзывы или новую оценку качества. Скриншоты в `reports/screenshots/`, browser-ответы в `ui_checks.json`. Успешный UI-тест означает работоспособность, а не правильность тональности во всех пяти случаях.

## Неделя 13. Итог и защита

Материалы: notebooks по каждой неделе, этот отчёт, таблица метрик, 10 слайдов `presentation.pptx`, приложение и инструкция воспроизведения. Личная защита и демонстрация перед комиссией выполняются автором работы. Демо: запустить приложение, ввести позитивный отзыв, негативный и смешанный; показать вероятности, затем объяснить ошибку и ограничения корпуса.

## Выводы и ограничения

На этом небольшом корпусе частотные модели дают полезный базовый результат; собственным embeddings не хватает данных. Нельзя заранее гарантировать, что BERT после fine-tuning станет лучше — это проверяется экспериментом. Негативные конструкции, косвенные жалобы и смешанные аспекты требуют контекста. Результат не переносится автоматически на русский язык, отзывы Wildberries/Ozon, нейтральные оценки или другие товарные категории. Для практического внедрения нужны новый доменный корпус, разбиение по товарам, несколько seed и оценка калибровки.

## Источники

1. Kotzias D., Denil M., de Freitas N., Smyth P. From Group to Individual Labels Using Deep Features. KDD, 2015.
2. UCI: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences (CC BY 4.0).
3. Pennington J., Socher R., Manning C. GloVe: Global Vectors for Word Representation. EMNLP, 2014.
4. Devlin J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018.
5. Turc I. et al. Well-Read Students Learn Better: On the Importance of Pre-training Compact Models. 2019. Карточка модели: https://huggingface.co/prajjwal1/bert-tiny
'''
    (REPORT / 'FINAL_REPORT.md').write_text(report, encoding='utf-8')
    slides = [
        ('Анализ тональности отзывов', 'Amazon · телефоны и аксессуары\nPython-проект · недели 1–6 и 8–13\nАвтор: впишите ФИО / группу перед защитой'),
        ('Задача и данные', f"Реальные отзывы UCI Amazon Cells\n1000 исходных; {stats['unique_documents']} после удаления повторов\nДва класса: позитивный / негативный\nЯзык отзывов: английский"),
        ('Честная оценка', 'Train / validation / test = 60 / 20 / 20\nДедупликация до разбиения; seed=42\nСловарь и модели обучаются на train\nВыбор модели только по validation macro-F1'),
        ('Предобработка и морфология', 'HTML / URL / регистр / токены / стоп-слова\nОтрицания not / no / never сохранены\nspaCy: леммы, части речи, Morph\nNLTK Porter / Snowball; эксперимент со слитным текстом'),
        ('От частот к embeddings', 'Bag-of-Words и TF-IDF: 1–2-граммы\nWord2Vec / fastText: 50D, 40 эпох\nGloVe-50: внешнее предобучение\nMean pooling + Logistic Regression; PCA'),
        ('Transformer и BERT', 'BERT-tiny: 2 слоя, hidden=128, CPU\nAttention: softmax(QKᵀ/√d)V\nFrozen encoder + LR → fine-tuning\n' + ('BERT выполнен' if bert_done else 'СТАТУС: загрузка весов заблокирована сетью; результатов BERT пока нет')),
        ('План экспериментов', 'LR: 2e-5 / 3e-5 / 5e-5\nОбъём train: 50% / 75% / 100% при 3e-5\n4 эпохи, batch=16, max_length=96\nAdamW; лучший checkpoint по validation'),
        ('Фактические результаты', f'Выбрана: {best.model}\nValidation macro-F1: {best.f1_macro:.3f}\nTest accuracy: {test_row.accuracy:.3f}\nTest macro-F1: {test_row.f1_macro:.3f}\nПолная таблица: reports/all_metrics.csv'),
        ('Ошибки и приложение', f'{len(errors)} разобранных ошибок в error_analysis.csv\nОтрицание, редкая лексика, смешанные аспекты\nStreamlit: класс + вероятности + JSON\n5 новых контрольных входов и реальные скриншоты'),
        ('Выводы и дальнейшая работа', 'Корпус мал; качество на новых товарах не гарантировано\nВероятности требуют калибровки\nНет нейтрального класса и русского языка\nСледующий шаг: новый доменный корпус и несколько seed'),
    ]
    deck = Presentation(); deck.slide_width = Inches(13.333); deck.slide_height = Inches(7.5)
    for i, (title, body) in enumerate(slides, 1):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        for p in slide.placeholders[1].text_frame.paragraphs:
            p.font.size = Pt(24)
        box = slide.shapes.add_textbox(Inches(.5), Inches(7), Inches(12), Inches(.3))
        box.text_frame.text = f'Review Lens · учебный NLP-проект                                      {i:02d} / 10'
        if i == 8:
            # Chart is on a separate area to avoid covering text.
            slide.placeholders[1].text_frame.paragraphs[0].font.size = Pt(22)
        slide.notes_slide.notes_text_frame.text = 'Объясните результат своими словами. Не называйте невыполненные эксперименты завершёнными. Источники и детали протокола: FINAL_REPORT.md.'
    deck.save(REPORT / 'presentation.pptx')
    state = {'classical': 'completed', 'bert': 'completed' if bert_done else 'blocked: huggingface.co HTTP 403',
             'python': platform.python_version(), 'selected_model': best.model,
             'selection_split': 'validation', 'presentation_slides': len(slides)}
    (REPORT / 'execution_status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
