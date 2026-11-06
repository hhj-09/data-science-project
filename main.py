import streamlit as st
from analysis import TITLE, SOURCE, load_data, complete_rows

st.set_page_config(page_title=TITLE, page_icon="🦾", layout="wide")

def home():
    st.title("🦾 " + TITLE)
    st.write("UR3 협동로봇팔의 센서 데이터를 활용한 그리퍼 파지 손실 분류 모델 개발")
    st.markdown("### 로봇팔이 물체를 놓친 순간을 센서로 구분할 수 있을까?")
    st.write("각 측정 시점의 관절 전류·온도·회전 속도와 그리퍼 전류로 같은 시점의 파지 손실 여부를 분류합니다.")
    data = st.cache_data(load_data)()
    clean = complete_rows(data)
    a, b, c = st.columns(3)
    a.metric("원본 관측 행", f"{len(data):,}")
    b.metric("분석에 사용한 행", f"{len(clean):,}")
    c.metric("분석 자료 중 손실 표시 행", f"{clean.grip_lost.sum():,}")
    st.caption("연속된 여러 행이 같은 손실 사건을 기록할 수 있습니다.")
    st.page_link("pages/1_탐색.py", label="센서 분포와 결측값 탐색", icon="📊")
    st.page_link("pages/2_분류_모델.py", label="분류 모델 평가 결과", icon="🌳")
    st.markdown(f"[데이터 출처 · UCI UR3 CobotOps]({SOURCE})")

page = st.navigation([
    st.Page(home, title="프로젝트", icon="🦾", default=True),
    st.Page("pages/1_탐색.py", title="탐색", icon="📊"),
    st.Page("pages/2_분류_모델.py", title="분류 모델", icon="🌳"),
])
st.set_page_config(page_title=TITLE, page_icon="🦾")
page.run()
