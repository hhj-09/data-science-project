
import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------

st.set_page_config(
    page_title="UR3 CobotOps 데이터 분석",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------
# 2. 제목 및 소개
# -----------------------------------------

st.title("🤖 UR3 CobotOps 데이터 분석실")

st.subheader(
    "협동로봇팔의 센서 데이터를 활용한 그리퍼 파지 손실 분석"
)

st.write("""
이 웹앱은 UR3 협동로봇팔에서 수집한 센서 데이터를 분석하기 위한 프로젝트입니다.

로봇팔은 여러 개의 관절과 그리퍼를 이용하여 물체를 잡고 이동시킵니다.
이때 그리퍼가 물체를 제대로 잡지 못하거나 작업 도중 물체를 놓치는 문제가 발생할 수 있습니다.

이 프로젝트에서는 로봇팔의 관절별 전류, 온도, 회전 속도와 그리퍼 전류 등의 데이터를 활용하여
정상적인 파지 상태와 파지 손실 상태의 차이를 분석하고자 합니다.
""")

st.divider()

# -----------------------------------------
# 3. 데이터 소개
# -----------------------------------------

st.header("📊 1. 데이터 소개")

st.write("""
**UR3 CobotOps 데이터**는 UR3 협동로봇팔의 작동 과정에서 기록된
시간 정보, 관절별 센서값, 로봇의 보호 정지 상태 및 그리퍼의 파지 손실 여부 등을
포함하는 데이터입니다.

각 행은 기록된 로봇의 작동 관측값을 나타내며,
각 열은 해당 관측 시점의 센서값이나 작동 상태를 나타냅니다.
""")

st.info("""
**이 프로젝트에서 해결하고자 하는 문제**

로봇팔의 그리퍼가 물체를 제대로 잡지 못하고 놓치는 현상을
센서 데이터로 분석할 수 있을까?
""")

# -----------------------------------------
# 4. 데이터 분석 목표
# -----------------------------------------

st.header("🎯 2. 데이터 분석 목표")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("센서 데이터 탐색")
    st.write("""
    관절별 전류, 온도, 회전 속도 및 그리퍼 전류의 분포를 확인합니다.
    """)

with col2:
    st.subheader("파지 상태 비교")
    st.write("""
    정상 파지와 파지 손실 상태에서 센서값이 어떤 차이를 보이는지 분석합니다.
    """)

with col3:
    st.subheader("파지 손실 분류")
    st.write("""
    센서 데이터를 활용하여 그리퍼의 파지 손실 여부를 분류하는 모델을 만듭니다.
    """)

st.divider()

# -----------------------------------------
# 5. 데이터 컬럼 설명
# -----------------------------------------

st.header("📋 3. 데이터 컬럼 설명")

column_info = {
    "Timestamp": "센서 데이터가 기록된 시간 정보",
    "Current_J0": "J0 관절의 전류 데이터",
    "Temperature_T0": "T0로 표시된 온도 데이터",
    "Current_J1": "J1 관절의 전류 데이터",
    "Temperature_J1": "J1 관절의 온도 데이터",
    "Current_J2": "J2 관절의 전류 데이터",
    "Temperature_J2": "J2 관절의 온도 데이터",
    "Current_J3": "J3 관절의 전류 데이터",
    "Temperature_J3": "J3 관절의 온도 데이터",
    "Current_J4": "J4 관절의 전류 데이터",
    "Temperature_J4": "J4 관절의 온도 데이터",
    "Current_J5": "J5 관절의 전류 데이터",
    "Temperature_J5": "J5 관절의 온도 데이터",
    "Speed_J0": "J0 관절의 회전 속도 데이터",
    "Speed_J1": "J1 관절의 회전 속도 데이터",
    "Speed_J2": "J2 관절의 회전 속도 데이터",
    "Speed_J3": "J3 관절의 회전 속도 데이터",
    "Speed_J4": "J4 관절의 회전 속도 데이터",
    "Speed_J5": "J5 관절의 회전 속도 데이터",
    "Tool_current": "로봇팔 도구부의 전류 데이터",
    "cycle": "로봇의 작업 사이클 관련 정보",
    "Robot_ProtectiveStop": "로봇의 보호 정지 발생 여부",
    "grip_lost": "그리퍼의 파지 손실 발생 여부"
}

column_df = pd.DataFrame(
    list(column_info.items()),
    columns=["컬럼명", "설명"]
)

st.dataframe(
    column_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "컬럼 설명은 제공된 변수 이름을 바탕으로 작성했습니다. "
    "정확한 측정 단위와 세부 정의는 원본 데이터 설명서를 통해 확인해야 합니다."
)

st.divider()

# -----------------------------------------
# 6. 분석에 사용할 변수
# -----------------------------------------

st.header("🔍 4. 분석에 사용할 변수")

st.subheader("독립변수")

st.write("""
- Current_J0 ~ Current_J5
- Temperature_T0, Temperature_J1 ~ Temperature_J5
- Speed_J0 ~ Speed_J5
- Tool_current
""")

st.subheader("종속변수")

st.write("""
- grip_lost

그리퍼가 물체를 놓쳤는지 여부를 나타내는 변수입니다.
이 값을 예측하는 것이 아니라, 센서값을 바탕으로 정상 파지와 파지 손실을 분류하는 것이 목표입니다.
""")

st.divider()

# -----------------------------------------
# 7. 데이터 파일 불러오기
# -----------------------------------------

st.header("📂 5. 실제 데이터 확인")

st.write("""
아래에서 제공된 CSV 파일을 업로드하거나,
GitHub 저장소에 CSV 파일을 올려 자동으로 불러올 수 있습니다.
""")

file_path = Path(__file__).parent / "dataset_02052023.xlsx - data (1).csv"

uploaded_file = st.file_uploader(
    "UR3 CobotOps CSV 파일 업로드",
    type=["csv"],
    help="Google Drive에서 다운로드한 dataset_02052023.xlsx - data (1).csv 파일을 선택하세요."
)

df = None

# GitHub 저장소에 파일이 있는 경우
if file_path.exists():
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949")
    except Exception as e:
        st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}")

# 업로드한 파일이 있으면 업로드 파일 우선 사용
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="cp949")
    except Exception as e:
        st.error(f"업로드 파일을 읽는 중 오류가 발생했습니다: {e}")

# -----------------------------------------
# 8. 데이터 미리보기 및 기본 정보
# -----------------------------------------

if df is not None:

    st.success("UR3 CobotOps 데이터가 성공적으로 불러와졌습니다.")

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터 수", f"{len(df):,}행")
    col2.metric("전체 컬럼 수", f"{len(df.columns)}개")
    col3.metric(
        "전체 결측값 수",
        f"{int(df.isna().sum().sum()):,}개"
    )

    st.subheader("데이터 미리보기")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("컬럼별 결측값 확인")

    missing_df = pd.DataFrame({
        "컬럼명": df.columns,
        "결측값 수": df.isna().sum().values,
        "결측 비율(%)": (
            df.isna().mean().values * 100
        ).round(2)
    })

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "아직 데이터가 불러와지지 않았습니다. "
        "CSV 파일을 업로드하거나 GitHub 저장소에 데이터 파일을 추가하세요."
    )

# -----------------------------------------
# 9. 마무리
# -----------------------------------------

st.divider()

st.header("🚀 프로젝트 진행 과정")

st.write("""
1. UR3 CobotOps 데이터 수집
2. 데이터의 결측값 및 분포 탐색
3. 정상 파지와 파지 손실 상태 비교
4. 분류 모델 학습
5. 모델의 성능 평가 및 활용 가능성 분석
""")

st.caption(
    "UR3 CobotOps | Robot Gripper Grip Loss Classification Project"
)
