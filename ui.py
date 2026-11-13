"""페이지에서 함께 쓰는 데이터 로딩과 그래프 표시 설정."""
import streamlit as st
from analysis import load_raw, prepare_data, complete_rows, evaluate

COLORS = {"정상": "#157F83", "파지 손실": "#DD6B3E"}

@st.cache_data(show_spinner=False)
def get_data():
    raw = load_raw()
    prepared = prepare_data(raw)
    return raw, prepared, complete_rows(prepared)

@st.cache_data(show_spinner="두 분류 모델을 학습하고 평가하고 있습니다.")
def get_evaluation(data):
    return evaluate(data)


def chart(figure, key=None):
    figure.update_layout(
        font={"family": "sans-serif", "size": 13},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": -0.2, "title_text": ""},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch", key=key, config={"displaylogo": False, "scrollZoom": False})


def csv_bytes(table):
    return table.to_csv(index=False).encode("utf-8-sig")
