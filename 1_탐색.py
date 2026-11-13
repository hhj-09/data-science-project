import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analysis import FEATURES, PAPER, TARGET, LABELS, class_table, column_dictionary
from ui import COLORS, chart, csv_bytes, get_data

st.subheader("센서 데이터 탐색")
raw, prepared, data = get_data()
raw_tab, missing_tab, distribution_tab, correlation_tab = st.tabs(["원본·컬럼", "결측값", "상태·분포", "변수 간 관련성"])

with raw_tab:
    a, b, c = st.columns(3)
    a.metric("원본 행", f"{len(raw):,}")
    b.metric("원본 열", raw.shape[1])
    c.metric("입력 센서", len(FEATURES))
    st.write("원본 데이터 미리보기")
    limit = st.selectbox("표시할 행 수", [20, 100, 500, len(raw)])
    st.dataframe(raw.head(limit), hide_index=True, height=340)
    st.caption("원본 파일은 data/dataset_02052023.xlsx의 data 시트입니다.")
    st.write("컬럼별 의미")
    dictionary = column_dictionary(raw)
    st.dataframe(dictionary, hide_index=True, height=500)
    st.download_button("컬럼 설명표 다운로드", csv_bytes(dictionary), "column_dictionary.csv", "text/csv")
    st.info("Temperature_T0를 Temperature_J0로 대응시키고, 원본 'cycle '의 끝 공백을 제거했습니다. False 또는 0은 정상, True 또는 1은 파지 손실입니다.")
    st.caption("전류 A, 온도 °C, 속도 °/s는 소개 논문 표 1의 표기입니다. 원본 Excel에는 단위 메타데이터가 없어, 속도는 단위 변환 없이 원자료 값으로 분석합니다.")
    st.markdown(f"[단위와 상태 정의의 출처 · 소개 논문 표 1]({PAPER})")

with missing_tab:
    absent = prepared[FEATURES].isna()
    count = absent.sum(axis=1)
    st.write("센서값이 하나라도 비어 있는 행을 제외하는 완전 사례 분석을 사용합니다.")
    a, b, c = st.columns(3)
    a.metric("처리 전", f"{len(prepared):,}행")
    b.metric("제외", f"{len(prepared)-len(data):,}행")
    c.metric("처리 후", f"{len(data):,}행")
    st.write(f"입력 19개가 모두 비어 있는 행은 {(count == 19).sum()}개, 18개가 비어 있는 행은 {(count == 18).sum()}개입니다. 대부분의 센서를 임의의 값으로 채우면 실제 동작과 다른 조합이 만들어질 수 있어 이 행들을 제외했습니다.")
    missing = pd.DataFrame({"컬럼": prepared.columns, "결측 행 수": prepared.isna().sum().to_numpy(), "결측 비율(%)": prepared.isna().mean().to_numpy()*100})
    st.dataframe(missing.style.format({"결측 비율(%)": "{:.2f}"}), hide_index=True, height=400)
    fig = px.bar(missing.loc[missing["결측 행 수"] > 0], x="컬럼", y="결측 행 수", title="결측값이 있는 컬럼", color_discrete_sequence=[COLORS["정상"]])
    fig.update_traces(name="결측 행 수", showlegend=True)
    chart(fig, "missing")
    st.caption(f"종속변수 결측은 {prepared[TARGET].isna().sum()}개입니다. 제외된 행의 상태와 아래 처리 전후 비율을 함께 확인하세요.")
    st.dataframe(class_table(prepared.loc[~prepared.index.isin(data.index)]).rename(columns={"행 수":"제외된 행 수"}), hide_index=True)
    st.download_button("결측값 표 다운로드", csv_bytes(missing), "missing_values.csv", "text/csv")

with distribution_tab:
    before = class_table(prepared).assign(구간="처리 전")
    after = class_table(data).assign(구간="처리 후")
    counts = pd.concat([before, after], ignore_index=True)
    st.dataframe(counts[["구간", "상태", "행 수", "비율(%)"]].style.format({"비율(%)":"{:.2f}"}), hide_index=True)
    fig = px.bar(counts, x="구간", y="비율(%)", color="상태", text="행 수", title="정상·파지 손실 구성비", color_discrete_map=COLORS, category_orders={"상태":["정상","파지 손실"]})
    fig.update_yaxes(range=[0,100])
    chart(fig, "class_ratio")
    sensor = st.selectbox("분포를 비교할 센서", FEATURES, index=18)
    kind = st.radio("그래프 종류", ["히스토그램", "상자그림"], horizontal=True)
    chart_data = data[[sensor, TARGET]].assign(상태=data[TARGET].map(LABELS))
    if kind == "히스토그램":
        bins = st.slider("히스토그램 구간 수", 10, 60, 35, step=5)
        edges = np.histogram_bin_edges(data[sensor], bins=bins)
        centers = (edges[:-1] + edges[1:])/2
        fig = go.Figure()
        for status in ["정상", "파지 손실"]:
            values = chart_data.loc[chart_data["상태"] == status, sensor]
            frequencies, _ = np.histogram(values, bins=edges)
            fig.add_bar(x=centers, y=frequencies / len(values)*100, width=np.diff(edges), name=status, marker_color=COLORS[status], opacity=0.58)
        fig.update_layout(title=f"{sensor} 상태별 분포", barmode="overlay", xaxis_title=f"{sensor} 원자료 값", yaxis_title="각 상태 안에서의 비율(%)")
        st.caption("두 상태에 같은 구간 경계를 적용하고 각각의 전체 행 수로 나눴습니다.")
    else:
        fig = px.box(chart_data, x="상태", y=sensor, color="상태", points="outliers", title=f"{sensor} 상태별 중앙값과 산포", color_discrete_map=COLORS, labels={sensor:f"{sensor} 원자료 값"})
        st.caption("상자는 가운데 50%의 범위, 내부 선은 중앙값입니다. 수염은 사분위 범위의 1.5배 경계 안에 있는 최솟값·최댓값까지이며, 밖의 값은 점으로 표시됩니다.")
    chart(fig, "distribution")
    summary = data.groupby(TARGET)[sensor].describe().rename(index=LABELS).rename(columns={"count":"행 수", "mean":"평균", "std":"표준편차", "min":"최솟값", "25%":"1사분위수", "50%":"중앙값", "75%":"3사분위수", "max":"최댓값"})
    st.dataframe(summary.style.format("{:.6g}"))

with correlation_tab:
    st.write("피어슨 상관계수는 두 센서가 함께 변하는 선형 관계를 나타냅니다. 값의 범위는 -1부터 1까지이며 부호는 변화 방향을 나타냅니다.")
    sensor = st.selectbox("관련성을 확인할 센서", FEATURES, index=0)
    correlations = data[FEATURES].corr()
    related = correlations[sensor].drop(sensor).rename("상관계수").sort_values(key=abs, ascending=False).rename_axis("센서").reset_index()
    fig = px.bar(related, x="상관계수", y="센서", orientation="h", title=f"{sensor}와 다른 센서의 상관계수", color="상관계수", color_continuous_scale="Tealrose", range_color=[-1,1])
    fig.update_xaxes(range=[-1,1])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=570)
    chart(fig, "correlations")
    joint_pairs = pd.DataFrame({"관절":[f"J{i}" for i in range(6)], "전류·속도 상관계수":[correlations.loc[f"Current_J{i}",f"Speed_J{i}"] for i in range(6)]})
    st.write("같은 관절의 전류와 회전 속도 비교")
    st.dataframe(joint_pairs.style.format({"전류·속도 상관계수":"{:.4f}"}), hide_index=True)
    with st.expander("전체 센서 상관계수 표"):
        st.dataframe(correlations.style.format("{:.3f}"))
    st.caption("그래프는 결측 처리 후 전체 관측을 사용합니다. 상관관계를 파지 손실의 원인으로 해석하려면 별도의 검증이 필요합니다.")
