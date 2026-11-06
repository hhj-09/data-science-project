
1_탐색.py
100%
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from analysis import FEATURES, load_data, complete_rows

st.title("📊 센서 데이터 탐색")
raw = st.cache_data(load_data)()
data = complete_rows(raw)
miss = raw[FEATURES].isna()
st.subheader("결측값 처리")
st.write(f"입력값이 일부라도 비어 있는 {miss.any(axis=1).sum()}행을 제외했습니다. 그중 {miss.all(axis=1).sum()}행은 모든 센서가 비어 있습니다.")
with st.expander("컬럼별 결측값 확인"):
    st.dataframe(raw.isna().sum().rename("결측 개수"))
    st.caption("원본 Temperature_T0는 Temperature_J0로 표시합니다.")
st.subheader("정상·파지 손실 비율")
counts = pd.DataFrame({"원본 행": raw.grip_lost.value_counts(), "처리 후 행": data.grip_lost.value_counts()}).sort_index()
counts.index = ["정상", "파지 손실"]
counts["처리 후 비율(%)"] = counts["처리 후 행"] / len(data) * 100
st.dataframe(counts.style.format({"처리 후 비율(%)": "{:.2f}"}))
st.subheader("정상과 파지 손실의 센서 분포")
sensor = st.selectbox("비교할 센서", FEATURES, index=18)
bins = np.histogram_bin_edges(data[sensor], bins=35)
records = []
for lost, group in data.groupby("grip_lost"):
    hist, _ = np.histogram(group[sensor], bins=bins)
    for i, n in enumerate(hist):
        records.append({"상태": "파지 손실" if lost else "정상", "시작": bins[i], "끝": bins[i+1], "비율": n/len(group), "행 수": int(n)})
chart = alt.Chart(pd.DataFrame(records)).mark_bar(opacity=0.48).encode(
    x=alt.X("시작:Q", title=sensor, bin="binned"), x2="끝:Q",
    y=alt.Y("비율:Q", stack=None, title="각 상태 안에서의 비율", axis=alt.Axis(format="%")),
    color="상태:N", tooltip=["상태", "시작", "끝", "행 수", alt.Tooltip("비율:Q", format=".1%")])
st.altair_chart(chart, width="stretch")
stats = data.groupby("grip_lost")[sensor].describe().rename(index={False:"정상", True:"파지 손실"})
st.dataframe(stats.rename(columns={"mean":"평균","50%":"중앙값","std":"표준편차","count":"행 수","min":"최솟값","max":"최댓값"}))
st.subheader("센서 간 관련성")
st.write("선택한 센서와 다른 센서의 피어슨 상관계수입니다. 부호는 변화 방향을, 절댓값은 선형 관계의 강도를 나타냅니다.")
corr = data[FEATURES].corr()[sensor].drop(sensor).sort_values(key=abs, ascending=False)
st.bar_chart(corr.rename("상관계수"), horizontal=True)
1_탐색.py 표시 중입니다.
