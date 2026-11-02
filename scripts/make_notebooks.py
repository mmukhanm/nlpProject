"""Создать учебные notebooks. Выполнение: python -m scripts.run_notebooks."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
BOOT = '''from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / 'nlp_project').exists() else Path.cwd().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import json
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown
from nlp_project.core import corpus, clean, tokens, metrics, SEED
df = corpus()
train, val, test = [df[df.split == s].copy() for s in ('train', 'validation', 'test')]
'''

SPECS = [
('01', 'Введение в NLP и корпус',
 'Цель: установить окружение и исследовать реальные отзывы. Используется только Amazon Cells из UCI, не IMDb/Yelp. Порог задания — минимум 500 документов ИЛИ 50 000 слов. Подробные источники и лицензия: `data/DATA_CARD.md`.',
 '''print('Документов после удаления повторов:', len(df))
print('Слов:', sum(len(tokens(t, False)) for t in df.text))
display(df[['id','text','label','split']].head(10))
display(pd.crosstab(df.split, df.label))
assert len(df) >= 500
assert df.normalized.is_unique
''',
 'В корпусе 1000 исходных фрагментов; после нормализованной дедупликации остаётся 982. Отзывы английские, два класса. Нет ID товара и автора — нельзя проверить перенос на новые товары через group split.'),
('02', 'Токенизация, стоп-слова и слитный текст',
 'Цель: построить полный pipeline и сравнить до/после. Для sentiment нельзя удалять not/no/never. «Слитный» текст исследуется отдельно: сегментация может ломать бренды.',
 '''from nlp_project.core import normalize, split_glued
examples = train.head(12).text
display(pd.DataFrame({'исходный': examples, 'нормализация': examples.map(normalize), 'токены': examples.map(tokens), 'готовый': examples.map(clean)}))
display(pd.DataFrame([{'до': t, 'после': split_glued(t)} for t in ['thebatteryisnotgood', 'soundqualityisexcellent', 'thisphonedoesnotwork']]))
assert 'not' in tokens("This phone isn't good")
''',
 'Стоп-слова уменьшают размер словаря, но удаление отрицаний меняет смысл. Slit-text не включён автоматически в основной классификатор. Для Transformer сохраняется исходный текст с контекстом.'),
('03', 'Лемматизация и морфоанализ spaCy',
 'Цель: леммы, POS, Morph и сравнение с Porter/Snowball. Английская морфология не эквивалентна русской: падеж/число указываются только когда они реально присутствуют в разборе.',
 '''from nlp_project.core import morphology
comparison = morphology(train.head(20).text.tolist())
display(comparison[['word','lemma','pos','morph','porter','snowball']].drop_duplicates().head(40))
full = pd.read_csv(ROOT / 'reports/morphology.csv')
display(full.word.value_counts().head(50).to_frame('частота слова'))
display(full.lemma.value_counts().head(50).to_frame('частота леммы'))
display(Image(filename=str(ROOT / 'reports/figures/pos.png')))
''',
 'Стемминг отсекает окончания без контекста, лемматизация учитывает словарную форму и POS. Полные топ-50 сохранены в отчётах. spaCy может ошибаться на разговорных фрагментах; это автоматическая, а не эталонная морфология.'),
('04', 'Bag-of-Words и TF-IDF',
 'Цель: сравнить матрицы и значимые слова. Обучение словаря и IDF — строго на train; validation/test только transform.',
 '''from nlp_project.classical import vectorizer
for method in ['bow', 'tfidf']:
    vec = vectorizer(method)
    x_train = vec.fit_transform(train.text)
    x_val = vec.transform(val.text)
    print(method, 'train:', x_train.shape, 'validation:', x_val.shape)
    weights = np.asarray(x_train.sum(axis=0)).ravel()
    top = np.argsort(weights)[-20:][::-1]
    display(pd.DataFrame({'term': vec.get_feature_names_out()[top], 'weight': weights[top]}))
display(Image(filename=str(ROOT / 'reports/figures/wordcloud.png')))
''',
 'BOW отражает повторяемость, TF-IDF понижает вес общих слов. Суммы весов не показывают причинную важность для классификатора. Облако слов — описание train, не оценка качества.'),
('05', 'Word2Vec, GloVe и fastText',
 'Цель: обучить собственные векторы, найти соседей, выполнить арифметику и сравнить покрытие. GloVe предобучен на внешнем корпусе; равенства объёма обучения здесь нет.',
 '''from gensim.models import Word2Vec, FastText
sentences = [tokens(t) for t in train.text]
local = Word2Vec(sentences, vector_size=50, min_count=1, window=5, sg=1, workers=1, epochs=40, seed=SEED)
print('Word2Vec / phone:', local.wv.most_similar('phone', topn=5))
print('good + phone - bad:', local.wv.most_similar(positive=['good','phone'], negative=['bad'], topn=5))
ft = FastText.load(str(ROOT / 'models/fasttext.model'))
print('fastText, OOV-вектор:', ft.wv['batteryzzzz'].shape)
examples = json.loads((ROOT / 'reports/semantic_examples.json').read_text())
display(pd.DataFrame({k: str(v.get('phone')) for k,v in examples.items()}, index=['соседи phone']).T)
display(pd.read_json(ROOT / 'reports/embedding_coverage.json').T)
''',
 'На 589 train-документах Word2Vec/fastText ограничены объёмом и могут давать неинтуитивных соседей. fastText использует n-граммы символов; GloVe покрывает только известные слова. Арифметика не гарантирует осмысленного ответа.'),
('06', 'Прототип sentiment-классификатора и PCA',
 'Цель: применить векторы к тональности и сравнить с классическим baseline. Test не используется для выбора модели.',
 '''from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nlp_project.classical import vectorizer
model = Pipeline([('vec', vectorizer('bow')), ('clf', MultinomialNB())])
model.fit(train.text, train.label)
print('BOW+NB / validation:', metrics(val.label, model.predict(val.text)))
display(pd.read_csv(ROOT / 'reports/classical_metrics.csv'))
display(Image(filename=str(ROOT / 'reports/figures/embedding_pca.png')))
''',
 'Готовый прототип работает на реальных отзывах. PCA отображает векторы в 2D, но не заменяет accuracy/F1. Лучший baseline выбирается по validation; полный test сохраняется для сравнения.'),
('08', 'Transformer и механизм Attention',
 'Цель: 16 текстов, контекстные представления и TF-IDF. Требуются настоящие pretrained-веса: сначала `python -m nlp_project.transformer`. Если сеть блокирует Hugging Face, notebook остаётся невыполненным — случайная модель не подставляется.',
 '''from transformers import AutoTokenizer, BertModel
from nlp_project.transformer import download_model, attention_experiment, encode, seed_all
seed_all()
path = download_model()
tokenizer = AutoTokenizer.from_pretrained(path)
encoder = BertModel.from_pretrained(path, attn_implementation='eager')
attention_experiment(encoder, tokenizer, val, train)
display(pd.read_csv(ROOT / 'reports/attention_texts.csv'))
display(Image(filename=str(ROOT / 'reports/figures/attention.png')))
display(Image(filename=str(ROOT / 'reports/figures/representation_comparison.png')))
''',
 'Attention = softmax(QKᵀ/√d)V. Для каждого слова контекст меняет выходной вектор. Рассмотрим «Was not happy»: TF-IDF хранит частоты not/happy, BERT обновляет представление happy с учётом not и позиции. Само наличие attention не гарантирует правильную классификацию, а величина веса не доказывает причинность.'),
('09', 'Предобученный BERT: feature extraction',
 'Цель: применить pretrained encoder без изменения его весов. В базовом BERT нет sentiment-головы. Здесь используется разрешённый вариантом задания feature extraction + LR, это не zero-shot классификация.',
 '''import joblib
from transformers import AutoTokenizer, BertModel
from nlp_project.transformer import encode, seed_all
seed_all()
path = ROOT / 'models/pretrained_bert'
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
encoder = BertModel.from_pretrained(path, local_files_only=True)
head = joblib.load(ROOT / 'models/bert_frozen_lr.joblib')
z = encode(encoder, tokenizer, test.text.tolist())
prob = head.predict_proba(z)[:,1]
display(pd.DataFrame({'text': test.text.values, 'label': test.label.values, 'prediction': (prob >= .5).astype(int), 'p_positive': prob}).head(15))
print(metrics(test.label, prob >= .5))
display(pd.read_csv(ROOT / 'reports/all_metrics.csv').query("model in ['tfidf_lr','bow_nb','bert_frozen_lr']"))
''',
 'Замороженный BERT использует внешнее предобучение, но LR обучается на train. Это отделяет пользу представлений от fine-tuning. Сравнение нужно читать вместе с размером корпуса, параметрами pooling и неопределённостью малого test.'),
('10', 'Fine-tuning BERT',
 'Цель: адаптировать encoder к тональности. Код обучения доступен прямо ниже. Основной воспроизводимый запуск пяти опытов — `python -m nlp_project.transformer`. RUN_TRAINING можно включить для отдельного запуска, он не перезаписывает итоговую таблицу.',
 '''from nlp_project.transformer import fine_tune, EPOCHS, BATCH_SIZE, MAX_LENGTH
from transformers import AutoTokenizer
print({'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'max_length': MAX_LENGTH, 'optimizer': 'AdamW', 'weight_decay': .01, 'seed': SEED})
RUN_TRAINING = False
if RUN_TRAINING:
    path = ROOT / 'models/pretrained_bert'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model, history = fine_tune(path, tokenizer, train, val, 3e-5, 1., 'bert_notebook_demo')
    display(pd.DataFrame(history))
history_path = ROOT / 'reports/bert_training_history.csv'
if not history_path.exists():
    raise FileNotFoundError('Fine-tuning ещё не выполнен. Запустите nlp_project.transformer после разрешения huggingface.co.')
display(pd.read_csv(history_path))
display(pd.read_csv(ROOT / 'reports/all_metrics.csv').query("model in ['bert_frozen_lr', 'bert_lr3e5_full']"))
''',
 'Выбор checkpoint только по validation macro-F1. До/после сравниваются frozen encoder + LR и fine-tuned BERT. Рост качества не гарантирован; отрицательный результат тоже нужно честно отразить.'),
('11', 'Эксперименты и анализ ошибок',
 'Цель: исследовать LR, объём train и минимум 10 ошибок. При отсутствии bert_metrics здесь видны только выполненные baseline-опыты; это не завершённое BERT-исследование.',
 '''scores = pd.read_csv(ROOT / 'reports/all_metrics.csv')
display(scores)
if (ROOT / 'reports/bert_training_history.csv').exists():
    history = pd.read_csv(ROOT / 'reports/bert_training_history.csv')
    display(history.groupby(['model','lr','fraction']).f1_macro.max().to_frame('best_validation_f1'))
    display(Image(filename=str(ROOT / 'reports/figures/bert_learning_curves.png')))
else:
    print('BERT-эксперименты пока не выполнены: сетевой блокер. Результаты не выдумываются.')
errors = pd.read_csv(ROOT / 'reports/error_analysis.csv')
display(errors[['id','text','label','prediction','reason','analysis','proposed_fix']])
assert len(errors) >= 10
display(Image(filename=str(ROOT / 'reports/figures/confusion_matrix.png')))
''',
 'Ошибки: сложное отрицание, косвенная жалоба, редкая лексика, смешанные аспекты. Это гипотезы для будущей работы, не доказанные причины. Нельзя улучшать модель по этим test-ошибкам и снова выдавать оценку на том же test за независимую.'),
('12', 'Пользовательское NLP-приложение',
 'Цель: Streamlit-интерфейс лучшей по validation модели. Запуск: `streamlit run app.py`. Пять новых контрольных текстов — демонстрация UI, не отдельный размеченный тестовый корпус.',
 '''from nlp_project.inference import Predictor
p = Predictor()
checks = json.loads((ROOT / 'reports/ui_checks.json').read_text())
display(pd.DataFrame(checks))
display(p.predict('The sound is excellent and the battery lasts all day.'))
display(Image(filename=str(ROOT / 'reports/screenshots/control_05.png')))
''',
 'UI возвращает класс и вероятность; уверенность не гарантирует правильность. Пятый контрольный текст смешанный и может быть классифицирован неверно. Пустой ввод, длина и язык проверяются. Скриншоты получены настоящим браузером, а не нарисованы.'),
('13', 'Итоговый notebook и защита',
 'Цель: объединить материалы, таблицу прогресса и презентацию на 10 слайдов. Личную защиту проводит автор проекта; данный notebook содержит демонстрацию и структуру доклада.',
 '''display(Markdown((ROOT / 'reports/FINAL_REPORT.md').read_text()))
display(pd.read_csv(ROOT / 'reports/all_metrics.csv'))
display(Image(filename=str(ROOT / 'reports/figures/model_comparison.png')))
from pptx import Presentation
deck = Presentation(ROOT / 'reports/presentation.pptx')
print('Слайдов:', len(deck.slides))
print('Статус выполнения:', json.loads((ROOT / 'reports/execution_status.json').read_text()))
''',
 'Перед защитой проверьте execution_status.json: невыполненные BERT-этапы нельзя объявлять завершёнными. Покажите источник корпуса, честное разбиение, сравнение подходов, ошибку модели и работающий UI. Объясните отсутствие нейтрального класса и ограничения английского корпуса.'),
]


def main():
    for week, title, goal, code, conclusion in SPECS:
        notebook = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(f'# Неделя {int(week)} — {title}\n\n{goal}'),
            nb.v4.new_code_cell(BOOT), nb.v4.new_code_cell(code), nb.v4.new_markdown_cell('## Выводы\n\n' + conclusion)])
        notebook.metadata['kernelspec'] = {'display_name': 'Python 3 (NLP)', 'language': 'python', 'name': 'python3'}
        notebook.metadata['language_info'] = {'name': 'python', 'version': '3.12'}
        nb.write(notebook, ROOT / 'notebooks' / f'week_{week}.ipynb')
    print(f'Created {len(SPECS)} notebooks')


if __name__ == '__main__': main()
