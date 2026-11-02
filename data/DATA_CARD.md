# Карточка данных

**Корпус:** Sentiment Labelled Sentences, UCI ML Repository, Amazon Cells subset.

- Первоисточник: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- DOI: https://doi.org/10.24432/C57604
- Работа: Dimitrios Kotzias, Misha Denil, Nando de Freitas, Padhraic Smyth. *From Group to Individual Labels Using Deep Features*, KDD 2015.
- Условия UCI: **CC BY 4.0**, https://creativecommons.org/licenses/by/4.0/. Условия корпуса не заменяются лицензией кода.
- Исходный README авторов: [raw/uci_readme.txt](raw/uci_readme.txt).
- Получено через публичное зеркало: https://github.com/ctc316/uci-sentiment_labelled_sentences/tree/b8411a53bfaf71dffbcbf3d5e1549d5c0c60ed3a/data
- Файл: `amazon_cells_labelled.txt`, UTF-8, TSV без заголовка, sentence + binary label.
- SHA-256: `47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3`.
- 1000 реальных английских фрагментов отзывов Amazon, 500 positive (1), 500 negative (0). Телефоны, гарнитуры, зарядки и другие аксессуары. IMDb и Yelp не используются.

## Подготовка

Исходный файл сохранён без изменения. Для производной таблицы `processed/reviews.csv` выполняются HTML-unescape, раскрытие распространённых отрицаний, lower-case, удаление URL/пунктуации и нормализованная дедупликация. Конфликтующие метки одинакового нормализованного текста исключаются. После обработки 982 документа; train=589, validation=196, test=197. Seed=42, стратификация. Поля: `text`, `label`, `id` (номер исходной строки), `normalized`, `split`.

## Ограничения и допустимые выводы

Данные — размеченные предложения, не полные отзывы со всеми метаданными. Нет ID товара, автора, даты, рейтинга; невозможно группировать по товару или автору, изучать временной дрейф или считать реальный баланс тональности Amazon. Классы искусственно сбалансированы составителями. Нет neutral; смешанные отзывы принудительно относятся к одному из двух классов. Не гарантируется отсутствие сарказма, ошибок разметки и семантически близких текстов между split. 197 test-примеров — небольшая оценка; один seed не даёт статистического доказательства превосходства.

Интерфейс на русском не означает поддержку русских отзывов. Модель не валидирована на Ozon/Wildberries или других категориях. Для переноса нужен отдельный корпус и независимая оценка. Контрольные тексты UI написаны для демонстрации и не выдаются за исходные отзывы.

## Внешние ресурсы

- GloVe Wiki/Gigaword 50D: https://nlp.stanford.edu/projects/glove/ ; загрузка через официальный https://github.com/piskvorky/gensim-data с проверкой checksum. Веса 400k слов не включаются в Git; сохранён только используемый поднабор. Атрибуция: Pennington, Socher, Manning, EMNLP 2014. Условия pretrained GloVe — Public Domain Dedication and License (PDDL), см. страницу проекта.
- spaCy `en_core_web_sm` 3.8.0: https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0 ; MIT.
- BERT-tiny: https://huggingface.co/prajjwal1/bert-tiny ; следуйте лицензии/карточке модели перед перераспространением. Веса BERT не включаются в Git; загрузчик записывает точную ревизию после успешной загрузки. В текущем запуске они недоступны из-за сетевого HTTP 403.
