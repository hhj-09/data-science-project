import streamlit as st
from analysis import PROJECT, QUESTION, QUESTIONS, SOURCE, TITLE
from ui import get_data

st.set_page_config(page_title=TITLE, page_icon="🦾", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1220px; padding-top: 2.3rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.04em;}
[data-testid="stMetric"] {background: #F0F7F6; border-radius: 12px; padding: 18px;}
[data-testid="stSidebar"] {border-right: 1px solid #E4ECEB;}
</style>
""", unsafe_allow_html=True)
try:
    raw, prepared, clean = get_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"데이터를 읽을 수 없습니다. {error}")
    st.info("data/dataset_02052023.xlsx가 main.py와 함께 업로드되어 있는지 확인하세요.")
    st.stop()


def home():
    st.write(PROJECT)
    st.subheader("센서에서 파지 손실의 단서를 찾습니다")
    st.write(QUESTION)
    a, b, c = st.columns(3)
    a.metric("원본 관측", f"{len(raw):,}행")
    b.metric("분석에 사용", f"{len(clean):,}행")
    c.metric("파지 손실 비율", f"{clean.grip_lost.mean():.2%}")
    st.caption("한 행은 한 시점의 관측입니다. 연속된 여러 행이 같은 손실 사건을 기록할 수 있습니다.")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("데이터 탐색")
            st.write("결측값을 확인하고 센서 분포와 관련성을 비교합니다.")
            st.page_link("pages/1_탐색.py", label="탐색 페이지 열기", icon="📊")
    with right:
        with st.container(border=True):
            st.subheader("분류 모델")
            st.write("같은 평가 자료로 두 모델의 결과를 비교합니다.")
            st.page_link("pages/2_분류_모델.py", label="모델 평가 보기", icon="🌳")
    with st.expander("탐구할 소질문 5개"):
        for number, question in enumerate(QUESTIONS, start=1):
            st.write(f"{number}. {question}")
    st.caption("grip_lost=False는 정상, True는 파지 손실입니다. 목표는 센서가 기록된 시점의 상태를 분류하는 것입니다.")
    st.markdown(f"[UCI UR3 CobotOps 원자료]({SOURCE})")


page = st.navigation([
    st.Page(home, title="프로젝트", icon="🦾", default=True),
    st.Page("pages/1_탐색.py", title="탐색", icon="📊", url_path="explore"),
    st.Page("pages/2_분류_모델.py", title="분류 모델", icon="🌳", url_path="models"),
])
st.set_page_config(page_title=TITLE, page_icon="🦾")
with st.sidebar:
    st.divider()
    st.caption("UR3 CobotOps · 제공된 원본 데이터")
    st.caption("입력 센서 19개 · 정상 / 파지 손실")
st.title("🦾 " + TITLE)
page.run()
