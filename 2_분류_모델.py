import pandas as pd
import plotly.express as px
import streamlit as st
from analysis import LR, TREE, split_cycles, split_table
from ui import chart, csv_bytes, get_data, get_evaluation

st.subheader("분류 모델과 평가")
_, _, data = get_data()
train, test = split_cycles(data)
st.write("grip_lost의 정상과 파지 손실 중 하나를 구분하므로 이진 분류를 사용합니다. 19개 센서의 측정값을 입력하고 같은 시점의 상태를 정답으로 학습합니다.")

with st.expander("두 모델의 학습 방식과 설정", expanded=True):
    st.markdown(f"**{LR}**")
    st.write("센서값을 표준화한 뒤 가중합을 만들고, 이를 0~1 사이의 손실 확률 점수로 바꿉니다. 센서값의 가중합으로 두 상태를 구분하는 기본 모델로 사용합니다.")
    st.caption("StandardScaler → LogisticRegression · C=1.0 · solver=lbfgs · max_iter=2000")
    st.caption("표준화는 단위가 다른 센서의 크기를 맞춥니다. C는 가중치가 지나치게 커지는 것을 억제하는 정도를 조절하며 기본값 1을 사용합니다. max_iter는 학습 반복 횟수의 상한입니다.")
    st.markdown(f"**{TREE}**")
    st.write("‘특정 센서값이 기준 이하인가?’라는 질문으로 자료를 나눕니다. 여러 센서의 조건 조합과 비선형 관계를 표현할 수 있어 함께 비교합니다.")
    st.caption("DecisionTreeClassifier · max_depth=5 · min_samples_leaf=10 · random_state=42")
    st.caption("한 경로에서 질문을 최대 5단계까지만 나누고, 마지막 집단에는 최소 10행을 남깁니다. 난수 시드 42를 고정해 같은 자료로 결과를 재현할 수 있게 합니다.")
    st.write("두 모델 모두 class_weight='balanced'로 학습해 적게 나타나는 손실 행에 더 큰 가중치를 줍니다. 트리 깊이와 말단 표본 수를 제한해 지나치게 세부적인 규칙이 만들어지는 것을 줄입니다.")
    st.caption("예측은 각 모델의 predict를 사용합니다. 가중 학습의 확률 점수와 실제 현장 발생 확률의 일치 정도는 추가 검증이 필요합니다. 설정은 비교 전에 고정했습니다.")

st.write("학습·평가 자료 분할")
st.write("작업 주기를 첫 관측 시각 순으로 정렬해 앞 80%로 학습하고 뒤 20%에서 평가합니다. 같은 작업 주기가 두 집합에 함께 포함되지 않으며, 학습 구간의 마지막 시각이 평가 구간의 첫 시각보다 빠른지 확인합니다.")
st.dataframe(split_table(train, test), hide_index=True)
st.caption("80%와 20%는 작업 주기 수 기준입니다. 표준화의 평균과 표준편차는 학습 자료로만 계산합니다. Num, Timestamp, cycle, Robot_ProtectiveStop은 모델 입력에서 제외합니다.")

scores, matrices, predictions = get_evaluation(data)
st.subheader("평가 지표 비교")
st.dataframe(scores.style.format("{:.4f}"))
st.download_button("평가 결과 다운로드", csv_bytes(scores.reset_index()), "model_metrics.csv", "text/csv")
st.caption(f"파지 손실(True=1)이 양성입니다. 평가 구간은 정상 {(~test.grip_lost).sum():,}행, 손실 {test.grip_lost.sum():,}행입니다.")
st.write(f"모든 평가 행을 정상으로 분류하는 기준 모델의 정확도는 {(~test.grip_lost).mean():.4f}, 정밀도·재현율·F1은 0.0000입니다. 예측 양성이 없을 때 정밀도는 0으로 기록합니다.")
with st.expander("지표의 의미와 계산식"):
    st.markdown("""
- 정확도(Accuracy)는 전체 관측 중 상태를 맞힌 비율입니다. (TP + TN) / 전체 행 수
- 정밀도(Precision)는 손실로 분류한 행 중 실제 손실 비율입니다. TP / (TP + FP)
- 재현율(Recall)은 실제 손실 행 중 찾아낸 비율입니다. TP / (TP + FN)
- F1 점수는 정밀도와 재현율의 조화평균입니다. 2TP / (2TP + FP + FN)

TP는 손실을 찾아낸 행, TN은 정상을 맞힌 행, FP는 정상에서 낸 오경보, FN은 놓친 손실 행입니다.
""")
st.subheader("혼동 행렬(Confusion Matrix)")
for name, matrix in matrices.items():
    st.markdown(f"**{name}**")
    left, right = st.columns([1.25, 1])
    with left:
        fig = px.imshow(matrix, x=["정상", "파지 손실"], y=["정상", "파지 손실"], text_auto=True, color_continuous_scale="Teal", labels={"x":"예측 상태", "y":"실제 상태", "color":"관측 행 수"}, title="실제 상태와 예측 상태의 교차표", aspect="auto")
        fig.update_layout(height=330)
        chart(fig, name)
    with right:
        st.dataframe(pd.DataFrame(matrix, index=["실제 정상", "실제 파지 손실"], columns=["예측 정상", "예측 파지 손실"]))
        st.caption(f"찾아낸 손실 {matrix[1,1]}행 · 놓친 손실 {matrix[1,0]}행 · 오경보 {matrix[0,1]}행")

with st.expander("평가 관측별 실제값과 예측값"):
    st.dataframe(predictions, hide_index=True, height=320)
    st.download_button("평가 행별 예측 다운로드", csv_bytes(predictions), "test_predictions.csv", "text/csv")
st.caption("이 평가는 센서가 기록된 시점의 상태를 다룹니다. 미래의 손실을 미리 알리는 성능과 다른 로봇·작업 환경에서의 성능은 추가 자료로 검증해야 합니다.")
with st.expander("모델 구현 참고 자료"):
    st.markdown("[로지스틱 회귀 공식 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)")
    st.markdown("[의사결정트리 공식 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeClassifier.html)")
