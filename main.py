import streamlit as st
from analysis import (
    load_dataset,
    FEATURES,
    TARGET,
    validate_columns,
    prepare_target,
    prepare_features
)

st.set_page_config(
    page_title="UR3 그리퍼 파지 손실 분석",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 UR3 협동로봇팔 그리퍼 파지 손실 분석")
st.subheader("센서 데이터를 활용한 파지 손실 분류 프로젝트")

st.write(
    "UR3 협동로봇팔의 관절별 전류, 온도, 회전 속도 및 "
    "그리퍼 전류 데이터를 분석하여 파지 손실 여부를 분류합니다."
)

st.divider()

st.header("프로젝트 목표")

st.write(
    "로봇팔이 물체를 제대로 잡지 못하고 놓치는 현상을 "
    "센서 데이터를 통해 분석하는 것이 목표입니다. "
    "정상 파지와 파지 손실 상태의 센서 데이터 차이를 탐색하고, "
    "분류 모델을 활용하여 파지 손실 여부를 구분합니다."
)

col1, col2, col3 = st.columns(3)

col1.metric("1단계", "데이터 탐색")
col2.metric("2단계", "분류 모델")
col3.metric("3단계", "모델 평가")

st.info(
    "왼쪽 페이지 메뉴에서 '1_탐색'과 '2_분류_모델'을 선택하여 "
    "각 단계의 분석을 진행할 수 있습니다."
)

with st.expander("분석에 사용하는 변수 확인"):
    st.write("**독립변수**")
    st.write(", ".join(FEATURES))

    st.write("**종속변수**")
    st.write(TARGET)

    st.caption(
        "실제 데이터에는 Temperature_J0 대신 Temperature_T0가 "
        "포함되어 있어 해당 열을 사용합니다."
    )

df = load_dataset()

if df is not None:

    missing_columns = validate_columns(df)

    if missing_columns:
        st.error(
            "데이터에 다음 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    else:
        y, label_info = prepare_target(df[TARGET])
        X = prepare_features(df)

        st.header("데이터 기본 정보")

        col1, col2, col3 = st.columns(3)

        col1.metric("전체 데이터 수", f"{len(df):,}개")
        col2.metric("독립변수 수", f"{len(FEATURES)}개")
        col3.metric(
            "입력 변수 결측 셀",
            f"{int(X.isna().sum().sum()):,}개"
        )

        st.caption(f"종속변수 변환 기준: {label_info}")

        st.subheader("데이터 미리보기")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )
