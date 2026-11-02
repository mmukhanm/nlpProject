import pytest
from streamlit.testing.v1 import AppTest
from nlp_project.core import ROOT


def test_empty_review_error():
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    at.button[0].click().run(timeout=60)
    assert not at.exception
    assert at.error and 'Введите' in at.error[0].value


@pytest.mark.parametrize('text', [
    'The sound quality is excellent and I love these headphones.',
    'The battery is terrible. This phone does not work.',
    'Great price and easy to use. Highly recommended.',
    'The charger broke after one day. Waste of money.',
    'The design is nice but the sound is poor and I cannot recommend it.',
])
def test_five_control_reviews(text):
    if not (ROOT / 'reports/best_model.json').exists(): pytest.skip('Train models first')
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
    at.text_area[0].set_value(text)
    at.button[0].click().run(timeout=60)
    assert not at.exception and not at.error
    assert at.subheader[0].value in ['Позитивный', 'Негативный']
    assert '%' in at.metric[0].value
