# -*- coding: utf-8 -*-
"""
스마트폰 이용행태 조사 2025 - 데이터 분석 웹앱 (Streamlit)

[실제 CSV 구조 확인 결과 - 코드는 이 구조를 기준으로 작성됨]
- 인코딩: CP949 (UTF-8로는 읽히지 않음)
- 54행 x 45열, 한 행 = 개인이 아니라 '집단 하나'의 집계 통계
- 구분1 = 집계 기준(총인원, 성별, 연령, 지역, 직업, 가구소득, 학력, 주거형태,
         가족구성, 방송이용 유형, 지역유형), 구분2 = 그 기준 안의 집단
- '총인원' 행 1개 = 전체 응답 집계 행 -> 집단 비교에서 제외
- 구분1마다 같은 약 7,969명을 서로 다른 기준으로 나눈 것이므로 집단끼리 겹침
- 이용시간대 5개 열의 합 = 100%, 콘텐츠 이용 빈도 7개 열의 합 = 100%

실행 방법:  같은 폴더에 데이터 CSV를 두고  streamlit run main.py
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 상수 (실제 CSV의 열 이름)
# ----------------------------------------------------------------------------
DATA_FILE = "스마트폰_이용행태_조사_2025_전처리완료.csv"  # main.py와 같은 폴더에 둘 것
ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-16"]

COL_G1 = "구분1"
COL_G2 = "구분2"
COL_N = "사례수 (명)"
COL_USE = "스마트폰 이용률 (퍼센트)"
COL_AVG = "평균이용시간 (분)"

# (열 이름, 보기 좋은 이름, 구간 대표값(분))  ※ '4시간 이상'은 임의로 300분으로 가정
TIME_BINS = [
    ("이용시간 1시간 미만 (퍼센트)", "1시간 미만", 30),
    ("1-2시간 미만 (퍼센트)", "1~2시간 미만", 90),
    ("2-3시간 미만 (퍼센트)", "2~3시간 미만", 150),
    ("3-4시간 미만 (퍼센트)", "3~4시간 미만", 210),
    ("4시간이상 (퍼센트)", "4시간 이상", 300),
]

CONTENTS = ["OTT시청", "뉴스정보검색", "SNS및메신저"]
CONTENT_NAME = {"OTT시청": "OTT 시청", "뉴스정보검색": "뉴스·정보 검색", "SNS및메신저": "SNS·메신저"}
# 실제 의미에 맞는 순서 (많이 이용 -> 적게 이용)
FREQ_ORDER = ["매일", "주 5-6일", "주 3-4일", "주 1-2일", "월 1-3일", "2-3달에 2일 이하", "전혀안함"]
# 빈도 점수: 일주일에 대략 며칠 이용하는지 (근사값)
FREQ_DAYS = {"매일": 7.0, "주 5-6일": 5.5, "주 3-4일": 3.5, "주 1-2일": 1.5,
             "월 1-3일": 0.5, "2-3달에 2일 이하": 0.1, "전혀안함": 0.0}

TOTAL_WORDS = {"총인원", "전체", "합계", "총계", "계", "전체응답", "전체 응답"}
CORE_DIMS = ["성별", "연령", "직업", "학력"]
ORDINAL_KEYWORDS = ["연령", "소득"]  # 순서가 분명한 변수만 순위상관 허용
MIN_ROWS_FOR_MODEL = 30  # 이보다 집단 수가 적으면 회귀 모델을 만들지 않음


# ----------------------------------------------------------------------------
# 공통 도우미
# ----------------------------------------------------------------------------
def show_fig(fig):
    """Streamlit 버전에 상관없이 그래프를 화면 너비에 맞게 표시"""
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def interpret(lines):
    """그래프 아래 '핵심 해석' 상자"""
    if isinstance(lines, str):
        lines = [lines]
    st.info("**핵심 해석**\n\n" + "\n\n".join(lines))


def find_col(columns, exact=None, keywords=()):
    if exact is not None and exact in columns:
        return exact
    if keywords:
        for c in columns:
            if all(k in c for k in keywords):
                return c
    return None


def fmt_min(m):
    """분 -> '196.9분(약 3시간 17분)'"""
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "-"
    h = int(m // 60)
    mm = int(round(m - h * 60))
    if mm == 60:
        h, mm = h + 1, 0
    return f"{m:.1f}분(약 {h}시간 {mm}분)" if h > 0 else f"{m:.1f}분"


def corr_strength(r):
    a = abs(r)
    if a >= 0.7:
        s = "강한"
    elif a >= 0.4:
        s = "어느 정도 뚜렷한"
    elif a >= 0.2:
        s = "약한"
    else:
        return "거의 없는"
    return f"{s} {'양(+)' if r > 0 else '음(-)'}의"


def safe_corr(x, y):
    """피어슨·스피어만 상관계수. 계산 불가능하면 None"""
    d = pd.concat([pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    if len(d) < 4 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return None
    try:
        pr, pp = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
        sr, sp = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    except Exception:
        return None
    return {"n": len(d), "pearson": float(pr), "p_pearson": float(pp),
            "spearman": float(sr), "p_spearman": float(sp)}


def add_fit_line(fig, x, y):
    d = pd.concat([pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    if len(d) >= 3 and d.iloc[:, 0].nunique() >= 2:
        m, b = np.polyfit(d.iloc[:, 0], d.iloc[:, 1], 1)
        xs = np.linspace(d.iloc[:, 0].min(), d.iloc[:, 0].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="추세선",
                                 line=dict(dash="dash", color="gray")))
    return fig


def weighted_mean_std(values, weights=None):
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v = v[ok]
    if len(v) == 0:
        return np.nan, np.nan
    if weights is not None:
        w = np.asarray(weights, dtype=float)[ok]
        if np.any(np.isnan(w)) or w.sum() <= 0:
            w = None
    else:
        w = None
    mean = np.average(v, weights=w)
    std = float(np.sqrt(np.average((v - mean) ** 2, weights=w)))
    return float(mean), std


# ----------------------------------------------------------------------------
# 데이터 불러오기 및 전처리
# ----------------------------------------------------------------------------
def read_csv_auto(file_bytes):
    """한글 인코딩을 자동으로 찾아서 읽기"""
    last_err = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            if df.shape[1] >= 2 and len(df) > 0:
                return df, enc
        except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            last_err = e
    raise ValueError(f"CSV 파일을 읽을 수 없습니다. (마지막 오류: {last_err})")


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    raw, enc = read_csv_auto(file_bytes)
    log = {"encoding": enc, "raw_rows": int(len(raw)), "raw_cols": int(raw.shape[1])}

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    raw_view = df.copy()
    log["raw_missing"] = int(df.isna().sum().sum())
    log["raw_duplicates"] = int(df.duplicated().sum())

    cols = list(df.columns)
    g1 = find_col(cols, COL_G1, ("구분", "1"))
    g2 = find_col(cols, COL_G2, ("구분", "2"))
    if g1 is None or g2 is None:
        raise ValueError("집단을 나타내는 '구분1', '구분2' 열을 찾을 수 없습니다.")
    id_cols = [g1, g2]

    # 1) 문자형 정리(공백 제거) / 숫자형 변환
    converted, stripped = [], 0
    for c in cols:
        if c in id_cols:
            df[c] = df[c].map(lambda v: " ".join(v.replace("\u00a0", " ").split()) if isinstance(v, str) else v)
            df[c] = df[c].replace("", np.nan)
            stripped += 1
        elif not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].map(lambda v: v.replace("\u00a0", " ").strip() if isinstance(v, str) else v)
            s = s.map(lambda v: v.replace(",", "").replace("%", "") if isinstance(v, str) else v)
            df[c] = pd.to_numeric(s, errors="coerce")
            converted.append(c)
    log["converted_cols"] = converted
    log["stripped_cols"] = stripped

    # 2) 퍼센트 열의 범위(0~100) 검증
    pct_cols = [c for c in df.columns if "퍼센트" in c and pd.api.types.is_numeric_dtype(df[c])]
    bad = 0
    for c in pct_cols:
        m = (df[c] < 0) | (df[c] > 100)
        bad += int(m.sum())
        df.loc[m, c] = np.nan
    log["out_of_range"] = bad

    # 3) 중복 제거 -> 4) 핵심 열 결측 행 제거
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["dup_removed"] = n0 - len(df)
    avg = find_col(list(df.columns), COL_AVG, ("평균", "이용시간"))
    key_cols = id_cols + ([avg] if avg else [])
    n1 = len(df)
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    log["missing_removed"] = n1 - len(df)
    log["after_clean_rows"] = int(len(df))
    log["remaining_missing"] = int(df.isna().sum().sum())
    return raw_view, df, log


def split_total(df, g1, g2):
    is_total = df[g1].isin(TOTAL_WORDS) | df[g2].isin(TOTAL_WORDS)
    return df[is_total].copy(), df[~is_total].copy()


# ----------------------------------------------------------------------------
# 화면 시작
# ----------------------------------------------------------------------------
st.set_page_config(page_title="스마트폰 이용행태 분석", layout="wide")
st.title("스마트폰 이용행태 조사 2025 - 데이터 분석")
st.caption("조사 자료를 자동으로 정리하고, 5가지 탐구 질문에 답하는 그래프와 해석을 보여 줍니다.")

def find_data_file():
    """main.py와 같은 폴더(없으면 현재 실행 폴더)에서 데이터 파일 찾기"""
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for folder in (here, os.getcwd()):
        path = os.path.join(folder, DATA_FILE)
        if os.path.exists(path):
            return path
    return None


data_path = find_data_file()
if data_path is None:
    st.error(f"데이터 파일 '{DATA_FILE}'을(를) 찾을 수 없습니다. main.py와 같은 폴더에 이 파일을 넣고 다시 실행해 주세요.")
    st.stop()
try:
    with open(data_path, "rb") as f:
        file_bytes = f.read()
except OSError as e:
    st.error(f"데이터 파일을 여는 중 문제가 생겼습니다: {e}")
    st.stop()
source_name = DATA_FILE

try:
    raw_df, clean_df, log = load_and_clean(file_bytes)
except Exception as e:
    st.error(f"파일을 처리하는 중 문제가 생겼습니다: {e}")
    st.stop()

cols = list(clean_df.columns)
G1 = find_col(cols, COL_G1, ("구분", "1"))
G2 = find_col(cols, COL_G2, ("구분", "2"))
NCOL = find_col(cols, COL_N, ("사례수",))
USE = find_col(cols, COL_USE, ("이용률",))
AVG = find_col(cols, COL_AVG, ("평균", "이용시간"))

if AVG is None:
    st.error("'평균이용시간' 열을 찾을 수 없어 분석을 진행할 수 없습니다. 열 이름을 확인해 주세요.")
    show_df(pd.DataFrame({"열 이름": cols}))
    st.stop()

total_df, analysis = split_total(clean_df, G1, G2)
analysis = analysis.reset_index(drop=True)
if analysis.empty:
    st.error("총인원(집계 행)을 제외하고 나면 분석할 집단이 없습니다.")
    st.stop()
analysis["집단"] = analysis[G1].astype(str) + " · " + analysis[G2].astype(str)
dims = list(pd.unique(analysis[G1]))
overall_avg = float(total_df[AVG].iloc[0]) if len(total_df) > 0 else None

TIME_AVAILABLE = [(c, lab, mid) for c, lab, mid in TIME_BINS if c in cols]
FREQ_COLS = {}
for p in CONTENTS:
    items = []
    for lab in FREQ_ORDER:
        c = find_col(cols, f"{p} {lab} (퍼센트)")
        if c:
            items.append((lab, c))
    if len(items) >= 3:
        FREQ_COLS[p] = items

# 사이드바 옵션
st.sidebar.header("분석 설정")
scope = st.sidebar.selectbox(
    "2번·3번 질문에서 사용할 집단",
    ["모든 집단 (총인원 제외)"] + dims,
    help="한 가지 기준(예: 연령)만 고르면 집단 수가 적어 결과가 불안정할 수 있어요.",
)
if scope.startswith("모든"):
    scope_df = analysis.copy()
    st.sidebar.caption("기준이 다른 집단(성별, 연령, 직업 등)이 함께 들어갑니다. 서로 겹치는 집단이 있어 '경향 살펴보기'용으로만 보세요.")
else:
    scope_df = analysis[analysis[G1] == scope].copy()
    st.sidebar.caption(f"'{scope}' 기준 집단 {len(scope_df)}개만 사용합니다.")

st.sidebar.markdown("---")
st.sidebar.caption(f"불러온 파일: {source_name} (인코딩: {log['encoding']})")

tabs = st.tabs(["데이터 확인", "1. 연령대별 이용 시간", "2. 이용 시간대 비율", "3. 콘텐츠 이용 빈도",
                "4. 집단 특성 비교", "5. 예측 모델", "다운로드"])

# ============================================================================
# 탭 0: 데이터 확인
# ============================================================================
with tabs[0]:
    st.header("데이터 확인 및 전처리")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행(집단) 개수", f"{log['raw_rows']:,}")
    c2.metric("열 개수", f"{log['raw_cols']:,}")
    c3.metric("결측치(원본)", f"{log['raw_missing']:,}")
    c4.metric("중복 행(원본)", f"{log['raw_duplicates']:,}")

    st.subheader("처리 전후 데이터 개수")
    steps = pd.DataFrame({
        "단계": ["원본 데이터", "중복 행 제거 후", "핵심 열(구분1·구분2·평균이용시간) 결측 행 제거 후",
                 "총인원(전체 집계) 행 제외 후 → 분석에 사용"],
        "행 개수": [log["raw_rows"], log["raw_rows"] - log["dup_removed"], log["after_clean_rows"], len(analysis)],
    })
    show_df(steps)
    if len(total_df) > 0:
        st.success(f"전체 응답을 나타내는 집계 행 {len(total_df)}개(구분1 = '{total_df[G1].iloc[0]}')를 찾아 "
                   "**집단별 비교 분석에서 제외**했습니다. 단, 그래프에서 '전체 평균' 기준선으로는 사용합니다.")
    else:
        st.info("전체 응답을 나타내는 집계 행이 발견되지 않아 제외한 행이 없습니다.")
    st.write(
        f"- 문자형 열 {log['stripped_cols']}개의 앞뒤 불필요한 공백을 정리했습니다.\n"
        f"- 숫자형이 아니었던 열 {len(log['converted_cols'])}개를 숫자형으로 변환했습니다.\n"
        f"- 중복 행 {log['dup_removed']}개, 핵심 열 결측으로 제거한 행 {log['missing_removed']}개, "
        f"0~100을 벗어나 결측으로 바꾼 퍼센트 값 {log['out_of_range']}개.\n"
        f"- 처리 후 남은 결측치는 {log['remaining_missing']}개입니다. 집계 통계는 값을 평균 등으로 채우면 왜곡되기 때문에 "
        "채우지 않고, 각 분석에서 해당 값만 제외합니다."
    )

    st.subheader("열 이름과 데이터 유형")
    info = pd.DataFrame({
        "열 이름": clean_df.columns,
        "데이터 유형": [str(t) for t in clean_df.dtypes],
        "결측치 개수": clean_df.isna().sum().values,
        "서로 다른 값 개수": [clean_df[c].nunique() for c in clean_df.columns],
    })
    show_df(info)

    st.subheader("자료의 구조: 집계 기준(구분1)")
    dim_tbl = analysis.groupby(G1, sort=False).agg(집단_수=(G2, "count"))
    if NCOL:
        dim_tbl["사례수_합계(명)"] = analysis.groupby(G1, sort=False)[NCOL].sum()
    dim_tbl = dim_tbl.reset_index().rename(columns={G1: "집계 기준(구분1)", "집단_수": "집단 수"})
    show_df(dim_tbl)
    st.caption("구분1마다 사례수 합계가 거의 같다면, 같은 응답자를 서로 다른 기준으로 나눈 것입니다. "
               "그래서 서로 다른 기준의 집단은 겹치며, 하나의 표본으로 합칠 수 없습니다.")

    sums_msgs = []
    if len(TIME_AVAILABLE) == 5:
        s = analysis[[c for c, _, _ in TIME_AVAILABLE]].sum(axis=1)
        sums_msgs.append(f"이용 시간대 5개 비율의 합: {s.min():.1f}~{s.max():.1f}% (반올림 때문에 100%에서 ±0.2 정도 차이)")
    for p, items in FREQ_COLS.items():
        s = analysis[[c for _, c in items]].sum(axis=1)
        sums_msgs.append(f"{CONTENT_NAME[p]} 이용 빈도 비율의 합: {s.min():.1f}~{s.max():.1f}%")
    if sums_msgs:
        st.markdown("**비율 자료 검증**\n\n" + "\n".join(f"- {m}" for m in sums_msgs))

    st.subheader("데이터 미리보기 (전처리 후)")
    show_df(clean_df.head(15))

# ============================================================================
# 탭 1: 연령대별
# ============================================================================
with tabs[1]:
    st.header("질문 1. 연령대별 하루 평균 스마트폰 이용 시간에는 어떤 차이가 있을까?")
    age_dim = next((d for d in dims if "연령" in str(d)), None)
    if age_dim is None:
        st.warning("연령 집단(구분1에 '연령'이 포함된 행)을 찾을 수 없어 이 분석을 할 수 없습니다.")
    else:
        age = analysis[analysis[G1] == age_dim].copy()
        order = list(age[G2])
        fig = px.bar(age, x=G2, y=AVG, text=age[AVG].round(1), category_orders={G2: order},
                     labels={G2: "연령대", AVG: "하루 평균 이용 시간(분)"},
                     title="연령대별 하루 평균 스마트폰 이용 시간")
        fig.update_traces(textposition="outside", marker_color="#4C78A8")
        if overall_avg is not None:
            fig.add_hline(y=overall_avg, line_dash="dash", line_color="#E45756",
                          annotation_text=f"전체 평균 {overall_avg:.1f}분", annotation_position="top right")
        fig.update_yaxes(range=[0, float(age[AVG].max()) * 1.2])
        show_fig(fig)

        tbl = age[[G2, AVG] + ([USE] if USE else []) + ([NCOL] if NCOL else [])].copy()
        tbl["가장 낮은 집단과의 차이(분)"] = (tbl[AVG] - tbl[AVG].min()).round(1)
        if overall_avg is not None:
            tbl["전체 평균과의 차이(분)"] = (tbl[AVG] - overall_avg).round(1)
        show_df(tbl.rename(columns={G2: "연령대"}))

        hi, lo = age.loc[age[AVG].idxmax()], age.loc[age[AVG].idxmin()]
        gap = hi[AVG] - lo[AVG]
        lines = [
            f"하루 평균 이용 시간이 가장 긴 연령대는 **{hi[G2]}**({fmt_min(hi[AVG])})이고, "
            f"가장 짧은 연령대는 **{lo[G2]}**({fmt_min(lo[AVG])})입니다. 두 집단의 차이는 **{gap:.1f}분**"
            f"(약 {hi[AVG] / lo[AVG]:.1f}배)입니다.",
        ]
        vals = age[AVG].reset_index(drop=True)
        peak = int(vals.idxmax())
        after = vals.iloc[peak:].diff().dropna()
        if len(after) >= 2 and (after < 0).all():
            lines.append(f"가장 긴 연령대({hi[G2]}) 이후에는 연령대가 높아질수록 이용 시간이 계속 줄어듭니다.")
        if overall_avg is not None:
            above = [str(g) for g, v in zip(age[G2], age[AVG]) if v > overall_avg]
            lines.append(f"전체 평균({overall_avg:.1f}분)보다 긴 연령대: {', '.join(above) if above else '없음'}.")
        if USE and pd.notna(lo[USE]) and lo[USE] < 90:
            lines.append(f"참고로 {lo[G2]} 집단은 스마트폰 이용률 자체가 {lo[USE]:.1f}%로 낮아, 이용 시간이 짧게 나타난 데에는 "
                         "'스마트폰을 아예 쓰지 않는 사람이 많다'는 점도 영향을 줄 수 있습니다.")
        lines.append("이 자료에는 개인별 응답이 없어 t검정·ANOVA 같은 통계 검정은 할 수 없습니다. "
                     "위 차이는 집단 평균을 비교한 **기술(요약) 결과**입니다.")
        interpret(lines)

# ============================================================================
# 탭 2: 시간대별 비율과 평균 이용 시간
# ============================================================================
with tabs[2]:
    st.header("질문 2. 이용 시간대별 비율과 평균 이용 시간 사이에는 어떤 관계가 있을까?")
    if len(TIME_AVAILABLE) < 3:
        st.warning("이용 시간대별 비율 열을 충분히 찾지 못해 이 분석을 할 수 없습니다.")
    else:
        st.markdown(
            "각 집단에서 '하루에 1시간 미만 쓰는 사람이 몇 %, 4시간 이상 쓰는 사람이 몇 %…'처럼 나눈 비율입니다. "
            "**5개 비율을 모두 더하면 100%** 이므로, 어떤 구간의 비율이 커지면 다른 구간의 비율은 줄어드는 관계가 "
            "자연스럽게 생깁니다. 그래서 한 구간만 보지 않고 5개 구간을 함께 살펴봅니다."
        )
        st.caption(f"분석 대상: {scope if not scope.startswith('모든') else '모든 집단'} ({len(scope_df)}개 집단)")

        rows = []
        for c, lab, _ in TIME_AVAILABLE:
            r = safe_corr(scope_df[c], scope_df[AVG])
            if r:
                rows.append({"구간": lab, "열": c, "피어슨 r": r["pearson"], "스피어만 ρ": r["spearman"],
                             "p값(피어슨)": r["p_pearson"], "집단 수": r["n"]})
        if not rows:
            st.warning("선택한 범위에서는 상관관계를 계산하기에 집단 수가 부족합니다. 사이드바에서 '모든 집단'을 선택해 보세요.")
        else:
            cr = pd.DataFrame(rows)
            fig = px.bar(cr, x="구간", y="피어슨 r", text=cr["피어슨 r"].round(2),
                         category_orders={"구간": [lab for _, lab, _ in TIME_AVAILABLE]},
                         title="시간대별 비율과 평균 이용 시간의 상관계수 (피어슨 r)",
                         labels={"피어슨 r": "상관계수 (-1 ~ +1)"})
            fig.update_traces(textposition="outside", marker_color="#4C78A8")
            fig.update_yaxes(range=[-1.15, 1.15])
            fig.add_hline(y=0, line_color="gray")
            show_fig(fig)

            top = cr.loc[cr["피어슨 r"].abs().idxmax()]
            pos = cr.loc[cr["피어슨 r"].idxmax()]
            neg = cr.loc[cr["피어슨 r"].idxmin()]
            lines = [
                f"상관계수가 +1에 가까우면 '비율이 높은 집단일수록 평균 이용 시간이 길다', -1에 가까우면 '짧다'는 뜻입니다.",
                f"**{pos['구간']}** 비율은 평균 이용 시간과 {corr_strength(pos['피어슨 r'])} 관계(r={pos['피어슨 r']:.2f})이고, "
                f"**{neg['구간']}** 비율은 {corr_strength(neg['피어슨 r'])} 관계(r={neg['피어슨 r']:.2f})입니다.",
            ]
            if pos["피어슨 r"] > 0 and neg["피어슨 r"] < 0:
                lines.append(f"즉 {pos['구간']} 이용자가 많은 집단일수록 평균 이용 시간이 길고, "
                             f"{neg['구간']} 이용자가 많은 집단일수록 짧은 경향이 있습니다. "
                             "비율의 합이 100%라서 두 방향이 서로 반대로 나타나는 것은 당연한 결과이기도 합니다.")
            lines.append(f"상관관계는 '함께 변하는 정도'일 뿐, 원인과 결과를 뜻하지는 않습니다. (집단 수 {int(top['집단 수'])}개)")
            interpret(lines)

            st.subheader("구간을 골라 산점도로 살펴보기")
            labs = [lab for _, lab, _ in TIME_AVAILABLE]
            pick = st.selectbox("시간대 구간", labs, index=len(labs) - 1, key="q2_pick")
            pcol = next(c for c, lab, _ in TIME_AVAILABLE if lab == pick)
            fig = px.scatter(scope_df, x=pcol, y=AVG, hover_name="집단", color=G1,
                             labels={pcol: f"'{pick}' 이용자 비율(%)", AVG: "하루 평균 이용 시간(분)", G1: "집계 기준"},
                             title=f"'{pick}' 비율과 평균 이용 시간")
            add_fit_line(fig, scope_df[pcol], scope_df[AVG])
            show_fig(fig)
            r = safe_corr(scope_df[pcol], scope_df[AVG])
            if r:
                direction = "길어지는" if r["pearson"] > 0 else "짧아지는"
                interpret([
                    f"'{pick}' 비율이 높은 집단일수록 평균 이용 시간이 {direction} 경향이 보입니다. "
                    f"피어슨 r = {r['pearson']:.2f}, 스피어만 ρ = {r['spearman']:.2f}로 {corr_strength(r['pearson'])} 관계입니다.",
                    f"(p값 {r['p_pearson']:.3g}: 집단 {r['n']}개로 계산한 값이며, 서로 겹치는 집단이 섞여 있을 수 있어 참고용입니다.)",
                ])

        st.subheader("집단별 시간대 구성 (100% 누적 막대)")
        dim_for_stack = st.selectbox("집계 기준", dims, index=dims.index("연령") if "연령" in dims else 0, key="q2_stack")
        sub = analysis[analysis[G1] == dim_for_stack].sort_values(AVG, ascending=False)
        long = sub.melt(id_vars=[G2, AVG], value_vars=[c for c, _, _ in TIME_AVAILABLE],
                        var_name="열", value_name="비율")
        mapname = {c: lab for c, lab, _ in TIME_AVAILABLE}
        long["구간"] = long["열"].map(mapname)
        fig = px.bar(long, x=G2, y="비율", color="구간", barmode="stack",
                     category_orders={G2: list(sub[G2]), "구간": [lab for _, lab, _ in TIME_AVAILABLE]},
                     labels={G2: "집단 (평균 이용 시간이 긴 순서)", "비율": "비율(%)"},
                     title=f"{dim_for_stack}: 이용 시간대 구성")
        show_fig(fig)
        if len(sub) >= 2:
            first, last = sub.iloc[0], sub.iloc[-1]
            big = TIME_AVAILABLE[-1]
            interpret(f"평균 이용 시간이 가장 긴 **{first[G2]}**에서는 '{big[1]}' 비율이 {first[big[0]]:.1f}%, "
                      f"가장 짧은 **{last[G2]}**에서는 {last[big[0]]:.1f}%입니다. 막대 색의 구성이 달라질수록 평균 이용 시간도 달라집니다.")

        if len(TIME_AVAILABLE) == 5:
            st.subheader("확인: 비율로 계산한 평균과 실제 평균은 얼마나 비슷할까?")
            est = sum(scope_df[c] / 100 * mid for c, _, mid in TIME_AVAILABLE)
            tmp = scope_df.assign(추정평균=est)
            fig = px.scatter(tmp, x="추정평균", y=AVG, hover_name="집단",
                             labels={"추정평균": "구간 대표값으로 계산한 평균(분)", AVG: "실제 평균 이용 시간(분)"},
                             title="시간대 비율로 계산한 평균 vs 실제 평균")
            lo_v, hi_v = float(min(tmp["추정평균"].min(), tmp[AVG].min())), float(max(tmp["추정평균"].max(), tmp[AVG].max()))
            fig.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines", name="같은 값(y=x)",
                                     line=dict(dash="dash", color="gray")))
            show_fig(fig)
            rr = safe_corr(tmp["추정평균"], tmp[AVG])
            if rr:
                interpret(f"각 구간의 대표값(30·90·150·210·300분, '4시간 이상'은 300분이라고 가정)에 비율을 곱해 더한 값은 "
                          f"실제 평균과 r = {rr['pearson']:.2f}로 {corr_strength(rr['pearson'])} 관계입니다. "
                          "시간대 비율과 평균 이용 시간은 서로 **같은 정보를 다른 방식으로 나타낸 것**에 가까우므로, "
                          "둘의 강한 관계는 '발견'이라기보다 당연한 결과임을 기억하세요.")

# ============================================================================
# 탭 3: 콘텐츠 이용 빈도
# ============================================================================
with tabs[3]:
    st.header("질문 3. OTT·뉴스정보검색·SNS및메신저 이용 빈도에 따라 평균 이용 시간은 어떻게 달라질까?")
    if not FREQ_COLS:
        st.warning("콘텐츠 이용 빈도 열을 찾지 못해 이 분석을 할 수 없습니다.")
    else:
        st.markdown(
            "이 자료에는 '매일 OTT를 보는 사람들만의 평균 이용 시간'이 따로 들어 있지 않습니다. 대신 **집단마다 '매일 이용'하는 사람의 비율**이 "
            "있으므로, 그 비율이 높은 집단과 낮은 집단의 평균 이용 시간을 비교합니다. 이용 빈도는 "
            "**매일 → 주 5-6일 → … → 전혀 안 함** 순서로 정렬했습니다."
        )
        st.caption(f"분석 대상: {scope if not scope.startswith('모든') else '모든 집단'} ({len(scope_df)}개 집단)")

        rows = []
        for p, items in FREQ_COLS.items():
            for lab, c in items:
                r = safe_corr(scope_df[c], scope_df[AVG])
                if r:
                    rows.append({"콘텐츠": CONTENT_NAME[p], "이용 빈도": lab, "피어슨 r": r["pearson"],
                                 "스피어만 ρ": r["spearman"], "집단 수": r["n"]})
        if not rows:
            st.warning("선택한 범위에서는 상관관계를 계산하기에 집단 수가 부족합니다. 사이드바에서 '모든 집단'을 선택해 보세요.")
        else:
            fc = pd.DataFrame(rows)
            fig = px.bar(fc, x="이용 빈도", y="피어슨 r", color="콘텐츠", barmode="group",
                         category_orders={"이용 빈도": FREQ_ORDER},
                         title="이용 빈도별 비율과 평균 이용 시간의 상관계수",
                         labels={"피어슨 r": "상관계수 (-1 ~ +1)"})
            fig.update_yaxes(range=[-1.1, 1.1])
            fig.add_hline(y=0, line_color="gray")
            show_fig(fig)

            lines = ["막대가 위쪽(+)이면 그 빈도로 이용하는 사람의 비율이 높은 집단일수록 평균 이용 시간이 길다는 뜻입니다."]
            for cname, g in fc.groupby("콘텐츠", sort=False):
                bp, bn = g.loc[g["피어슨 r"].idxmax()], g.loc[g["피어슨 r"].idxmin()]
                lines.append(f"**{cname}**: '{bp['이용 빈도']}' 비율이 높을수록 평균 이용 시간이 긴 경향(r={bp['피어슨 r']:.2f}), "
                             f"'{bn['이용 빈도']}' 비율이 높을수록 짧은 경향(r={bn['피어슨 r']:.2f})입니다.")
            interpret(lines)

            st.subheader("이용 빈도를 점수로 바꿔 보기")
            st.caption("각 집단의 빈도 분포를 '일주일에 평균 며칠 이용하는가'(매일=7일, 주 5-6일=5.5일, 주 3-4일=3.5일, "
                       "주 1-2일=1.5일, 월 1-3일=0.5일, 2-3달에 2일 이하=0.1일, 전혀 안 함=0일)로 환산한 근사값입니다.")
            score_df = scope_df[["집단", G1, AVG]].copy()
            score_cols = {}
            for p, items in FREQ_COLS.items():
                sc = sum(scope_df[c] / 100 * FREQ_DAYS[lab] for lab, c in items)
                name = f"{CONTENT_NAME[p]} 주간 이용일(근사)"
                score_df[name] = sc
                score_cols[p] = name
            pick_p = st.selectbox("콘텐츠 선택", list(FREQ_COLS.keys()), format_func=lambda k: CONTENT_NAME[k], key="q3_pick")
            xcol = score_cols[pick_p]
            fig = px.scatter(score_df, x=xcol, y=AVG, hover_name="집단", color=G1,
                             labels={xcol: "일주일에 이용하는 평균 일수(근사)", AVG: "하루 평균 이용 시간(분)", G1: "집계 기준"},
                             title=f"{CONTENT_NAME[pick_p]} 이용 빈도 점수와 평균 이용 시간")
            add_fit_line(fig, score_df[xcol], score_df[AVG])
            show_fig(fig)
            r = safe_corr(score_df[xcol], score_df[AVG])
            if r:
                if r["pearson"] > 0:
                    msg = "자주 이용하는 집단일수록 스마트폰 평균 이용 시간도 긴 경향이 있습니다."
                else:
                    msg = "자주 이용하는 집단일수록 오히려 스마트폰 평균 이용 시간이 짧은 경향이 있습니다."
                interpret([f"{CONTENT_NAME[pick_p]} 이용 빈도가 높은 집단일수록 점수가 높습니다. 피어슨 r = {r['pearson']:.2f}, "
                           f"스피어만 ρ = {r['spearman']:.2f} → {corr_strength(r['pearson'])} 관계입니다. {msg}",
                           "다만 이는 집단 수준의 경향입니다. 예를 들어 젊은 집단은 OTT·SNS 이용도 많고 스마트폰 이용 시간도 길어서, "
                           "'연령'이라는 공통 요인이 둘을 함께 움직였을 가능성이 큽니다."])

            st.subheader("'매일 이용' 비율이 높은 집단 vs 낮은 집단")
            comp = []
            for p, items in FREQ_COLS.items():
                daily_col = next((c for lab, c in items if lab == "매일"), None)
                if daily_col is None:
                    continue
                d = scope_df[[daily_col, AVG]].dropna()
                if len(d) < 4:
                    continue
                med = d[daily_col].median()
                hi_g, lo_g = d[d[daily_col] > med], d[d[daily_col] <= med]
                if len(hi_g) and len(lo_g):
                    comp.append({"콘텐츠": CONTENT_NAME[p], "구분": "'매일 이용' 비율이 높은 집단", "평균 이용 시간(분)": hi_g[AVG].mean()})
                    comp.append({"콘텐츠": CONTENT_NAME[p], "구분": "'매일 이용' 비율이 낮은 집단", "평균 이용 시간(분)": lo_g[AVG].mean()})
            if comp:
                cdf = pd.DataFrame(comp)
                fig = px.bar(cdf, x="콘텐츠", y="평균 이용 시간(분)", color="구분", barmode="group",
                             text=cdf["평균 이용 시간(분)"].round(1), title="'매일 이용' 비율 상위 절반 vs 하위 절반 집단의 평균 이용 시간")
                fig.update_traces(textposition="outside")
                show_fig(fig)
                msgs = []
                for cname, g in cdf.groupby("콘텐츠", sort=False):
                    h = g[g["구분"].str.contains("높은")]["평균 이용 시간(분)"].iloc[0]
                    l = g[g["구분"].str.contains("낮은")]["평균 이용 시간(분)"].iloc[0]
                    msgs.append(f"**{cname}**: 높은 집단 {h:.1f}분, 낮은 집단 {l:.1f}분 (차이 {h - l:+.1f}분)")
                interpret(["집단들을 '매일 이용' 비율의 중앙값을 기준으로 둘로 나눠 평균 이용 시간(집단 값의 단순 평균)을 비교했습니다."] + msgs)

# ============================================================================
# 탭 4: 집단 특성 비교
# ============================================================================
with tabs[4]:
    st.header("질문 4. 성별·연령·직업·학력 중 평균 이용 시간과 가장 관련이 높은 요소는?")
    st.markdown(
        "성별·직업·학력 같은 **범주형 변수는 숫자로 바꿔 상관계수를 구하지 않습니다.** (예: 사무직=1, 학생=2라고 하면 "
        "아무 의미 없는 숫자 크기가 생기기 때문입니다.) 대신 '그 기준으로 나눴을 때 집단들의 평균 이용 시간이 얼마나 "
        "달라지는가'를 비교합니다. 순서가 분명한 **연령·가구소득만** 순위상관(스피어만)을 추가로 계산합니다."
    )
    defaults = [d for d in CORE_DIMS if d in dims] or dims[:4]
    chosen = st.multiselect("비교할 집계 기준", dims, default=defaults)
    if not chosen:
        st.warning("비교할 집계 기준을 하나 이상 선택해 주세요.")
    else:
        rows = []
        for d in chosen:
            g = analysis[analysis[G1] == d]
            if len(g) < 2:
                continue
            w = g[NCOL] if NCOL else None
            wm, ws = weighted_mean_std(g[AVG], w)
            hi, lo = g.loc[g[AVG].idxmax()], g.loc[g[AVG].idxmin()]
            rows.append({"집계 기준": d, "집단 수": len(g), "가장 긴 집단": f"{hi[G2]} ({hi[AVG]:.1f}분)",
                         "가장 짧은 집단": f"{lo[G2]} ({lo[AVG]:.1f}분)", "최대-최소 격차(분)": hi[AVG] - lo[AVG],
                         "집단 간 흩어짐(가중 표준편차, 분)": ws})
        if not rows:
            st.warning("선택한 기준에 비교 가능한 집단이 충분하지 않습니다.")
        else:
            res = pd.DataFrame(rows).sort_values("집단 간 흩어짐(가중 표준편차, 분)", ascending=False).reset_index(drop=True)

            cols2 = st.columns(2)
            for i, d in enumerate(chosen):
                g = analysis[analysis[G1] == d]
                if g.empty:
                    continue
                fig = px.bar(g, x=G2, y=AVG, text=g[AVG].round(1), category_orders={G2: list(g[G2])},
                             labels={G2: d, AVG: "평균 이용 시간(분)"}, title=f"{d}별 평균 스마트폰 이용 시간")
                fig.update_traces(textposition="outside", marker_color="#4C78A8")
                fig.update_yaxes(range=[0, float(analysis[AVG].max()) * 1.2])
                if overall_avg is not None:
                    fig.add_hline(y=overall_avg, line_dash="dash", line_color="#E45756")
                with cols2[i % 2]:
                    show_fig(fig)
            st.caption("빨간 점선은 전체 평균입니다.")

            fig = px.bar(res, x="집단 간 흩어짐(가중 표준편차, 분)", y="집계 기준", orientation="h",
                         text=res["집단 간 흩어짐(가중 표준편차, 분)"].round(1),
                         category_orders={"집계 기준": list(res["집계 기준"])},
                         title="어떤 기준으로 나눌 때 집단 간 이용 시간 차이가 가장 클까?")
            fig.update_traces(textposition="outside", marker_color="#F58518")
            fig.update_yaxes(autorange="reversed")
            show_fig(fig)
            out = res.copy()
            for c in ["최대-최소 격차(분)", "집단 간 흩어짐(가중 표준편차, 분)"]:
                out[c] = out[c].round(1)
            show_df(out)

            top = res.iloc[0]
            lines = [
                f"'가중 표준편차'는 사례수를 고려해 집단들의 평균 이용 시간이 서로 얼마나 흩어져 있는지를 나타냅니다. "
                f"값이 클수록 그 기준으로 나눴을 때 집단 간 차이가 큽니다.",
                f"선택한 기준 중 **{top['집계 기준']}** 기준의 흩어짐이 {top['집단 간 흩어짐(가중 표준편차, 분)']:.1f}분으로 가장 크고, "
                f"가장 긴 집단은 {top['가장 긴 집단']}, 가장 짧은 집단은 {top['가장 짧은 집단']}입니다. "
                f"→ 이 자료에서는 '{top['집계 기준']}' 기준이 평균 이용 시간과 가장 관련이 높아 보입니다.",
            ]
            if len(res) >= 2:
                last = res.iloc[-1]
                lines.append(f"반대로 **{last['집계 기준']}** 기준은 {last['집단 간 흩어짐(가중 표준편차, 분)']:.1f}분으로 집단 간 차이가 가장 작습니다.")
            lines.append("주의: 연령·직업·학력은 서로 겹치는 특성입니다(예: 학생은 10대·20대, 대학생과 겹침). 그래서 어느 하나가 '원인'이라고 "
                         "단정할 수는 없고, 이 자료는 집단 평균만 있어 통계적 유의성 검정도 할 수 없습니다.")
            interpret(lines)

            ord_rows = []
            for d in chosen:
                if any(k in str(d) for k in ORDINAL_KEYWORDS):
                    g = analysis[(analysis[G1] == d) & (~analysis[G2].isin(["기타"]))].reset_index(drop=True)
                    r = safe_corr(pd.Series(range(len(g))), g[AVG]) if len(g) >= 4 else None
                    if r:
                        ord_rows.append({"집계 기준": d, "집단 수": r["n"], "스피어만 ρ(순서-평균 이용시간)": round(r["spearman"], 2),
                                         "p값": round(r["p_spearman"], 3)})
            if ord_rows:
                st.subheader("순서가 있는 변수의 순위상관")
                show_df(pd.DataFrame(ord_rows))
                msg = []
                for o in ord_rows:
                    rho = o["스피어만 ρ(순서-평균 이용시간)"]
                    updown = "짧아지는" if rho < 0 else "길어지는"
                    msg.append(f"**{o['집계 기준']}**: ρ = {rho:.2f} → 작은 값에서 큰 값 순서로 갈수록 이용 시간이 {updown} "
                               f"{corr_strength(rho)} 경향이 있습니다.")
                interpret(msg + ["순위상관은 '순서대로 갈수록 한 방향으로 변하는가'를 보는 값이라, 중간에 오르내림이 있으면 값이 작아질 수 있습니다."])

# ============================================================================
# 탭 5: 예측 모델
# ============================================================================
with tabs[5]:
    st.header("질문 5. 집단별 평균 스마트폰 이용 시간을 예측하는 회귀 모델을 만들 수 있을까?")
    st.warning("이 분석은 **개인의 이용 시간을 맞히는 것이 아니라**, 집단별 평균 이용 시간에 대한 **탐색적 예측**입니다.")

    # --- 타당성 점검 ---
    dim_info = analysis.groupby(G1, sort=False).agg(집단수=(G2, "count"))
    if NCOL:
        dim_info["사례수합"] = analysis.groupby(G1, sort=False)[NCOL].sum()
    n_dims = len(dim_info)
    max_rows = int(dim_info["집단수"].max())
    best_dim = dim_info["집단수"].idxmax()

    overlap = False
    if NCOL and n_dims >= 2:
        sums = dim_info["사례수합"]
        ref = float(sums.median())
        overlap = bool((sums.sub(ref).abs() / ref <= 0.05).sum() >= 2)
    n_feat_pool = len([c for c in cols if "퍼센트" in c])

    checks = pd.DataFrame([
        {"점검 항목": "서로 다른 집계 기준을 하나의 표본으로 합쳐도 되는가?",
         "결과": "합칠 수 없음" if overlap or n_dims >= 2 else "해당 없음",
         "설명": (f"구분1이 {n_dims}종이고, 각 기준의 사례수 합계가 거의 같음 → 같은 응답자를 여러 방식으로 나눈 자료라 "
                  "합치면 같은 사람이 중복으로 계산됨") if overlap else
                 (f"구분1이 {n_dims}종으로 집계 단위가 서로 다름" if n_dims >= 2 else "집계 기준이 하나뿐임")},
        {"점검 항목": f"한 가지 집계 기준 안의 집단 수가 {MIN_ROWS_FOR_MODEL}개 이상인가?",
         "결과": "통과" if max_rows >= MIN_ROWS_FOR_MODEL else "부족",
         "설명": f"가장 많은 기준은 '{best_dim}'로 {max_rows}개 집단"},
        {"점검 항목": "집단 수에 비해 설명 변수(열)가 지나치게 많지 않은가?",
         "결과": "통과" if max_rows >= 5 * 5 else "부족",
         "설명": f"퍼센트 열이 {n_feat_pool}개인데 집단은 최대 {max_rows}개 → 변수 수가 표본 수보다 많거나 비슷함"},
    ])
    show_df(checks)

    candidates = [d for d in dims if int(dim_info.loc[d, "집단수"]) >= MIN_ROWS_FOR_MODEL]

    if not candidates:
        st.error("**이 자료로는 회귀 모델을 만들지 않았습니다.**")
        interpret([
            "**왜 만들지 않았나요?**",
            f"1) 이 자료의 54행은 '성별', '연령', '직업' 등 **서로 다른 기준으로 나눈 집계 결과**이고, 기준마다 같은 응답자(약 "
            f"{int(dim_info['사례수합'].median()) if NCOL else '약 8천'}명)를 다시 나눈 것입니다. 이를 하나의 표로 합치면 같은 사람이 "
            "여러 번 들어가 학습 데이터와 테스트 데이터가 서로 섞이고(정보 누출), 성능이 실제보다 좋게 보입니다.",
            f"2) 한 가지 기준 안에서만 보면 집단이 최대 {max_rows}개({best_dim})뿐입니다. 학습용과 테스트용으로 나누면 "
            "몇 개 안 되는 점으로 모델을 만들고 검증하게 되어 결과가 매우 불안정합니다.",
            "3) 설명 변수로 쓸 수 있는 퍼센트 열은 수십 개인데 집단은 훨씬 적어, 모델이 우연한 패턴을 외워 버리는 과적합 위험이 큽니다. "
            "또 이용 시간대 비율은 평균 이용 시간을 거의 그대로 계산할 수 있는 값이라 '예측'이 아니라 '순환 논리'가 됩니다.",
            "**대신 할 수 있는 것**: 2~4번 탭처럼 집단별 평균을 비교하고 상관관계를 살펴보는 **기술 분석**은 타당합니다.",
            "**모델을 만들려면**: ① 개인 단위 응답 자료, 또는 ② 같은 기준·같은 모집단으로 집계된 집단이 최소 "
            f"{MIN_ROWS_FOR_MODEL}개 이상인 자료가 필요합니다.",
        ])
    else:
        st.success(f"집단이 {MIN_ROWS_FOR_MODEL}개 이상인 집계 기준이 있어 탐색용 모델을 만들 수 있습니다. "
                   "(한 가지 기준의 집단만 사용하며, 다른 기준의 행과 섞지 않습니다.)")
        mdim = st.selectbox("모델에 사용할 집계 기준", candidates)
        mdf = analysis[analysis[G1] == mdim].copy()
        time_cols = {c for c, _, _ in TIME_AVAILABLE}
        feat_options = [c for c in mdf.columns if pd.api.types.is_numeric_dtype(mdf[c]) and c not in time_cols
                        and c not in {AVG, NCOL, "집단"}]
        default_feats = [c for c in feat_options if c.startswith(tuple(CONTENTS + ["TV시청"])) and "매일" in c]
        if USE and USE in feat_options:
            default_feats.append(USE)
        feats = st.multiselect("설명 변수 (이용 시간대 비율은 평균 이용 시간과 거의 같은 정보라 제외했습니다)",
                               feat_options, default=default_feats)
        test_size = st.slider("테스트 데이터 비율", 0.2, 0.4, 0.25, 0.05)
        alpha = st.select_slider("규제 강도 (클수록 모델이 단순해집니다)", options=[0.1, 1.0, 3.0, 10.0], value=1.0)

        if not feats:
            st.warning("설명 변수를 하나 이상 선택해 주세요.")
        else:
            data = mdf[feats + [AVG, "집단"]].dropna()
            n_total = len(data)
            n_test = max(int(round(n_total * test_size)), 2)
            n_train = n_total - n_test
            if n_train < 3 * len(feats) or n_train < 10:
                st.error(f"결측치를 제외하고 {n_total}개 집단이 남아, 변수 {len(feats)}개로 학습하기엔 데이터가 부족합니다. 변수를 줄여 주세요.")
            else:
                try:
                    Xtr, Xte, ytr, yte, ntr, nte = train_test_split(
                        data[feats], data[AVG], data["집단"], test_size=n_test, random_state=42)
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(Xtr, ytr)
                    pred_te, pred_tr = model.predict(Xte), model.predict(Xtr)
                    r2_te, r2_tr = r2_score(yte, pred_te), r2_score(ytr, pred_tr)
                    mae = mean_absolute_error(yte, pred_te)
                    rmse = float(np.sqrt(mean_squared_error(yte, pred_te)))

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("테스트 R²", f"{r2_te:.3f}")
                    m2.metric("학습 R²", f"{r2_tr:.3f}")
                    m3.metric("테스트 MAE(분)", f"{mae:.1f}")
                    m4.metric("테스트 RMSE(분)", f"{rmse:.1f}")

                    res_df = pd.DataFrame({"집단": nte.values, "실제값": yte.values, "예측값": pred_te})
                    fig = px.scatter(res_df, x="실제값", y="예측값", hover_name="집단",
                                     labels={"실제값": "실제 평균 이용 시간(분)", "예측값": "예측 평균 이용 시간(분)"},
                                     title="테스트 데이터: 실제값 vs 예측값")
                    lo_v, hi_v = float(min(res_df.min(numeric_only=True))), float(max(res_df.max(numeric_only=True)))
                    fig.add_trace(go.Scatter(x=[lo_v, hi_v], y=[lo_v, hi_v], mode="lines", name="완벽한 예측(y=x)",
                                             line=dict(dash="dash", color="gray")))
                    show_fig(fig)

                    coef = pd.DataFrame({"변수": feats, "표준화 계수": model[-1].coef_}).sort_values("표준화 계수")
                    fig2 = px.bar(coef, x="표준화 계수", y="변수", orientation="h", title="변수별 영향 방향과 크기(표준화)")
                    show_fig(fig2)

                    if r2_te >= 0.7:
                        quality = "테스트 데이터에서 변화를 꽤 잘 설명합니다."
                    elif r2_te >= 0.3:
                        quality = "일부만 설명하며, 예측력은 보통 수준입니다."
                    elif r2_te >= 0:
                        quality = "설명력이 낮습니다."
                    else:
                        quality = "단순히 평균값으로 예측하는 것보다도 못합니다."
                    interpret([
                        f"테스트 R² = {r2_te:.2f}: 집단별 평균 이용 시간의 차이 중 약 {max(r2_te, 0) * 100:.0f}%를 설명합니다. {quality}",
                        f"MAE = {mae:.1f}분: 예측이 실제 집단 평균에서 평균적으로 약 {mae:.1f}분 벗어납니다.",
                        f"학습 R²({r2_tr:.2f})가 테스트 R²({r2_te:.2f})보다 훨씬 높다면 과적합 신호입니다. "
                        f"사용한 집단은 '{mdim}' 기준 {n_total}개(학습 {len(Xtr)}·테스트 {len(Xte)})뿐이라 결과는 우연에 크게 좌우될 수 있습니다.",
                        "이 모델은 **집단 평균에 대한 탐색적 예측**이며, 한 사람의 스마트폰 이용 시간을 예측하는 모델이 아닙니다.",
                    ])
                except Exception as e:
                    st.error(f"모델을 만드는 중 문제가 발생했습니다: {e}")

# ============================================================================
# 탭 6: 다운로드
# ============================================================================
with tabs[6]:
    st.header("전처리된 데이터 내려받기")
    st.write("공백 정리·숫자형 변환·중복/결측 처리가 끝난 데이터입니다. 엑셀에서 한글이 깨지지 않도록 UTF-8(BOM)으로 저장됩니다.")
    d1, d2 = st.columns(2)
    d1.download_button("전처리 데이터 전체 (총인원 행 포함)",
                       data=clean_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="스마트폰_이용행태_전처리.csv", mime="text/csv")
    d2.download_button("집단 비교용 데이터 (총인원 행 제외)",
                       data=analysis.drop(columns=["집단"]).to_csv(index=False).encode("utf-8-sig"),
                       file_name="스마트폰_이용행태_전처리_집단비교용.csv", mime="text/csv")
    st.caption(f"전체 {len(clean_df)}행 / 집단 비교용 {len(analysis)}행")
