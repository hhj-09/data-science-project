
import pandas as pd
import streamlit as st
from analysis import LR, TREE, load_data, complete_rows, evaluate, split_cycles

st.title("🌳 분류 모델")
data = complete_rows(st.cache_data(load_data)())
train, test = split_cycles(data)
st.write("측정 시점의 grip_lost는 정상과 파지 손실 두 값 중 하나이므로 이진 분류를 사용합니다.")
st.markdown(f"**{LR}**는 표준화한 센서값을 가중합해 손실 점수를 계산합니다. **{TREE}**는 센서값을 기준으로 질문을 나누어 비선형적인 조건을 표현합니다.")
st.caption("두 모델 모두 적은 손실 행에 더 큰 학습 가중치를 줍니다. 트리의 최대 깊이는 5, 말단 노드의 최소 표본 수는 10입니다.")
st.subheader("학습과 평가 구간")
st.write("작업 주기를 첫 관측 시각 순으로 정렬해 앞 80%로 학습하고 뒤 20%에서 한 번 평가합니다. 시각과 주기 번호는 입력에서 제외합니다.")
parts = {"학습": train, "평가": test}
st.dataframe(pd.DataFrame({name: {"작업 주기": p.cycle.nunique(), "관측 행": len(p), "정상": int((~p.grip_lost).sum()), "파지 손실": int(p.grip_lost.sum())} for name,p in parts.items()}).T)
scores, matrices = st.cache_data(evaluate)(data)
st.subheader("모델별 평가 지표")
st.dataframe(scores.style.format("{:.4f}"))
st.caption(f"양성은 파지 손실입니다. 평가 행을 모두 정상으로 분류했을 때 기준 정확도는 {(~test.grip_lost).mean():.4f}, 손실 재현율은 0입니다.")
with st.expander("평가 지표 읽기"):
    st.write("정확도는 전체에서 맞힌 비율, 정밀도는 손실이라고 분류한 행 중 실제 손실 비율, 재현율은 실제 손실 행 중 찾아낸 비율입니다. F1은 정밀도와 재현율의 조화평균입니다.")
st.subheader("혼동 행렬")
for name, matrix in matrices.items():
    st.markdown(f"**{name}**")
    st.dataframe(pd.DataFrame(matrix, index=["실제 정상","실제 손실"], columns=["예측 정상","예측 손실"]))
st.caption("각 칸의 단위는 관측 행입니다. 이 평가는 같은 시점의 상태 분류를 다루며, 손실 발생 전 경보의 성능은 별도 검증이 필요합니다.")
축소
 1/4페이지
