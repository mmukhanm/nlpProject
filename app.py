"""Неделя 12: интерфейс для английских отзывов Amazon Cells."""
import json
import pandas as pd
import streamlit as st
from nlp_project.core import ROOT
from nlp_project.inference import Predictor

st.set_page_config(page_title="Review Lens · Анализ отзывов", page_icon="🔎", layout="centered")
st.markdown("<style>.stApp {background:#f5f7fc} h1 {color:#18345d} .stButton button {border-radius:12px; background:#264b89; color:white}</style>", unsafe_allow_html=True)
st.caption("REVIEW LENS  /  NLP LAB")
st.title("Что говорят покупатели?")
st.write("Анализ тональности отзывов о телефонах, наушниках и аксессуарах маркетплейса Amazon.")
st.info("Модель обучена на английских отзывах. Введите текст на английском. Классы: позитивный и негативный; отдельного нейтрального класса нет.")


@st.cache_resource
def predictor(): return Predictor()


with st.form("review_form"):
    text = st.text_area("Отзыв покупателя", placeholder="The battery lasts all day and the sound is excellent.", height=150, max_chars=5000)
    submitted = st.form_submit_button("Анализировать", use_container_width=True)

if submitted:
    try:
        with st.spinner("Анализируем отзыв…"):
            result = predictor().predict(text)
        st.subheader(result["sentiment"])
        st.metric("Вероятность выбранного класса", f"{result['confidence']:.1%}")
        st.progress(result["confidence"])
        st.bar_chart(pd.DataFrame({"Вероятность": [1-result["positive_probability"], result["positive_probability"]]}, index=["Негативный", "Позитивный"]))
        st.caption(f"Модель: {result['model']}. Вероятности не проходили отдельную калибровку и не гарантируют правильность ответа.")
        if result["truncated"]: st.warning("Для BERT текст сокращён до 96 токенов.")
        st.download_button("Скачать результат JSON", json.dumps({"text": text, **result}, ensure_ascii=False, indent=2), "prediction.json", "application/json")
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else "Модели ещё не подготовлены. Выполните команды обучения из README.")

with st.expander("О проекте и результатах"):
    st.write("Реальный корпус UCI / Amazon Cells: 1000 размеченных предложений. Авторы: Kotzias et al., 2015. Разбиение 60/20/20 после удаления повторов; лучшая модель выбирается по validation macro-F1.")
    p = ROOT / "reports/all_metrics.csv"
    if p.exists(): st.dataframe(pd.read_csv(p).query("split == 'test'"), hide_index=True)
    st.caption("Учебный прототип. Сарказм, смешанные отзывы, новые товары и ошибки в тексте могут приводить к неверным ответам.")
