import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from nlp_project.core import ROOT, clean, corpus, load_corpus, metrics, normalize, tokens


def test_real_data_and_labels():
    df = load_corpus()
    assert len(df) >= 500
    assert set(df.label) == {0, 1}
    assert df.normalized.is_unique
    assert df.text.str.len().min() > 0


def test_no_leakage_between_partitions():
    df = corpus()
    assert set(df.split) == {"train", "validation", "test"}
    assert df.id.is_unique and df.normalized.is_unique
    parts = {s: set(df[df.split == s].normalized) for s in df.split.unique()}
    assert not parts['train'] & parts['test']
    assert not parts['train'] & parts['validation']
    assert not parts['validation'] & parts['test']
    for split in parts:
        assert set(df[df.split == split].label) == {0, 1}


@pytest.mark.parametrize("text", ["It isn't good!", "I can't recommend it.", "This won't work.", "Not good", "never works"])
def test_negation_survives(text):
    assert {"not", "never"} & set(tokens(text))


def test_html_urls_and_empty():
    assert normalize('<b>Great</b> https://example.com !') == 'great'
    assert tokens('') == []
    assert clean('THE battery is not good!') == 'battery not good'


def test_metrics_macro():
    y, p = [0, 0, 1, 1], [0, 1, 1, 1]
    m = metrics(y, p)
    assert m['accuracy'] == .75
    assert m['f1_macro'] == pytest.approx(f1_score(y, p, average='macro'))


def test_predictions_match_metrics():
    path = ROOT / 'reports/all_metrics.csv'
    if not path.exists(): pytest.skip('Run report generation first')
    all_metrics = pd.read_csv(path)
    for name in ['classical', 'bert']:
        pp = ROOT / f'reports/{name}_predictions.csv'
        if not pp.exists(): continue
        predictions = pd.read_csv(pp)
        for (model, split), frame in predictions.groupby(['model', 'split']):
            assert frame.id.is_unique
            assert frame.positive_probability.between(0, 1).all()
            saved = all_metrics.query('model == @model and split == @split').iloc[0]
            assert metrics(frame.label, frame.prediction)['f1_macro'] == pytest.approx(saved.f1_macro)


def test_selection_uses_validation():
    path = ROOT / 'reports/all_metrics.csv'
    if not path.exists(): pytest.skip('Run report generation first')
    frame = pd.read_csv(path).query("split == 'validation'")
    expected = frame.sort_values(['f1_macro', 'model'], ascending=[False, True]).iloc[0].model
    actual = json.loads((ROOT / 'reports/best_model.json').read_text())['name']
    assert actual == expected


def test_predictor_input_and_probabilities():
    from nlp_project.inference import Predictor
    if not (ROOT / 'reports/best_model.json').exists(): pytest.skip('Train models first')
    p = Predictor()
    for text in ['', '   ', '1234!!!', 'ужасные наушники', 'x' * 5001]:
        with pytest.raises(ValueError): p.predict(text)
    result = p.predict('The sound is excellent and the battery is great.')
    assert result['label'] in [0, 1]
    assert .5 <= result['confidence'] <= 1
    assert 0 <= result['positive_probability'] <= 1
