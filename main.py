# -*- coding: utf-8 -*-
"""
스마트폰 이용행태 조사(2025) 데이터 분석 웹앱
- 실행: streamlit run main.py
- 업로드한 CSV의 실제 열 이름을 읽어 '이용 시간 구간별 비율 열', '평균 이용시간 열' 등을 자동으로 찾아 사용합니다.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="스마트폰 이용행태 조사 분석", layout="wide")

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------
ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
TOTAL_WORDS = {"총인원", "전체", "합계", "총계", "계", "전체합계", "total", "all"}
NONE_OPT = "(없음)"

# 분석용으로 통일한 열 이름
C_DIM = "구분"          # 대분류 (예: 연령, 성별)
C_GRP = "집단"          # 세부 집단 (예: 10대, 20대)
C_N = "사례수"
C_MEAN = "평균 이용시간(분)"
C_RATE = "이용률"
C_TOTAL = "집계행"

DOWNLOAD_NAME = "스마트폰_이용행태_조사_2025_전처리결과.csv"
DEFAULT_CSV = "스마트폰_이용행태_조사_2025_전처리완료.csv"  # main.py와 같은 폴더에 둘 기본 데이터 파일


def _version():
    try:
        p = st.__version__.split(".")
        return int(p[0]), int(p[1])
    except Exception:
        return (0, 0)


NEW_API = _version() >= (1, 50)


# ------------------------------------------------------------------
# 기본 데이터 파일 찾기
# ------------------------------------------------------------------
def find_default_csv():
    """main.py와 같은 폴더의 기본 CSV를 찾는다. 이름이 조금 달라도 폴더 안의 CSV를 찾아 쓴다."""
    try:
        folders = [Path(__file__).resolve().parent, Path.cwd()]
    except NameError:
        folders = [Path.cwd()]
    for folder in folders:
        p = folder / DEFAULT_CSV
        if p.exists():
            return p
    for folder in folders:
        cands = sorted(folder.glob("*.csv"))
        if cands:
            return cands[0]
    return None


# ------------------------------------------------------------------
# 화면 출력 도우미 (Streamlit 버전 차이 대응)
# ------------------------------------------------------------------
def show_chart(fig, key=None):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=14),
    )
    if NEW_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df):
    if NEW_API:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def fmt_min(m):
    """분 → '2시간 10분' 형태"""
    if m is None or pd.isna(m):
        return "-"
    total = int(round(float(m)))
    h, mm = divmod(total, 60)
    return f"{mm}분" if h == 0 else f"{h}시간 {mm}분"


def korean_dtype(s):
    if pd.api.types.is_bool_dtype(s):
        return "참/거짓"
    if pd.api.types.is_integer_dtype(s):
        return "숫자(정수)"
    if pd.api.types.is_numeric_dtype(s):
        return "숫자(실수)"
    return "문자"


def josa(word, with_batchim, without_batchim):
    """받침 유무에 따라 조사를 붙인다. 예) josa('20대', '이', '가') → '20대가'"""
    w = str(word).strip()
    if w:
        ch = w[-1]
        if "가" <= ch <= "힣":
            return w + (with_batchim if (ord(ch) - 0xAC00) % 28 else without_batchim)
    return w + without_batchim


def josa_ro(word):
    w = str(word).strip()
    if w and "가" <= w[-1] <= "힣":
        b = (ord(w[-1]) - 0xAC00) % 28
        return w + ("로" if b in (0, 8) else "으로")
    return w + "로"


def strength_word(r):
    a = abs(r)
    if a >= 0.8:
        return "매우 강한"
    if a >= 0.6:
        return "강한"
    if a >= 0.4:
        return "뚜렷한"
    if a >= 0.2:
        return "약한"
    return "거의 없는"


# ------------------------------------------------------------------
# 파일 읽기 · 전처리
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes):
    """한글 인코딩(UTF-8, UTF-8 BOM, CP949, EUC-KR)을 차례로 시도해 CSV를 읽는다."""
    last_error = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:  # 구분자가 쉼표가 아닌 경우 대비
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return df, enc
        except UnicodeDecodeError as e:
            last_error = e
        except pd.errors.EmptyDataError:
            raise
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise ValueError(f"파일을 읽을 수 없습니다. ({last_error})")


def _clean_text(v):
    if isinstance(v, str):
        v2 = re.sub(r"\s+", " ", v.replace("\ufeff", "")).strip()
        return np.nan if v2 == "" else v2
    return v


def basic_clean(raw: pd.DataFrame):
    """열 이름 정리, 문자열 공백 정리, 숫자형 변환."""
    df = raw.copy()

    names, seen = [], {}
    for c in df.columns:
        name = re.sub(r"\s+", " ", str(c).replace("\ufeff", "")).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df.columns = names

    stripped_cells = 0
    converted_cols = []
    unconvertible = 0

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        cleaned = pd.Series([_clean_text(v) for v in s], index=s.index, dtype=object)
        stripped_cells += sum(1 for a, b in zip(s, cleaned) if isinstance(a, str) and a != b)

        nn = cleaned.notna()
        if nn.sum() > 0:
            t = (
                cleaned[nn]
                .astype(str)
                .str.replace(r"[,%\s]", "", regex=True)
                .str.replace("−", "-", regex=False)
            )
            conv = pd.to_numeric(t, errors="coerce")
            if conv.notna().sum() >= 0.8 * nn.sum():
                full = pd.Series(np.nan, index=s.index, dtype=float)
                full[nn] = conv.values
                unconvertible += int(conv.isna().sum())
                df[c] = full
                converted_cols.append(c)
                continue
        df[c] = cleaned

    info = {
        "stripped_cells": stripped_cells,
        "converted_cols": converted_cols,
        "unconvertible": unconvertible,
        "missing_total": int(df.isna().sum().sum()),
    }
    return df, info


def parse_hours(name):
    """열 이름에서 이용 시간 구간(시간 단위)을 읽는다. 예) '1-2시간 미만' → (1, 2)"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*시간", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간\s*미만", name)
    if m:
        return 0.0, float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*시간\s*이상", name)
    if m:
        n = float(m.group(1))
        return n, n + 1.0
    return None


def short_label(name):
    s = re.sub(r"\(.*?\)", "", name)
    s = s.replace("이용시간", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s or name


def detect_columns(df: pd.DataFrame):
    """실제 열 이름을 보고 분석에 쓸 열을 찾는다. (없으면 None)"""
    cols = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in cols if c not in num_cols]

    g1 = "구분1" if "구분1" in cols else (text_cols[0] if text_cols else None)
    g2 = "구분2" if "구분2" in cols else (text_cols[1] if len(text_cols) > 1 else None)

    mean_col = next((c for c in num_cols if "평균" in c and "시간" in c), None)
    cnt_col = next((c for c in num_cols if "사례수" in c), None)
    rate_col = next((c for c in num_cols if "이용률" in c), None)

    exclude = ("요금", "만원", "TV", "OTT", "뉴스", "SNS", "시청", "검색", "평균")
    bins = [
        c for c in num_cols
        if parse_hours(c) is not None and not any(w in c for w in exclude)
    ]
    bins.sort(key=lambda c: parse_hours(c)[0])
    return dict(g1=g1, g2=g2, mean=mean_col, cnt=cnt_col, rate=rate_col, bins=bins)


def handle_missing_dup(df: pd.DataFrame, key_cols):
    """중복 제거 → 핵심 열 결측 행 제거 → 그 밖의 숫자 열 결측은 중앙값으로 대체."""
    info = {"원본 행 수": len(df)}
    dup_n = int(df.duplicated().sum())
    d = df.drop_duplicates().reset_index(drop=True)
    info["중복 제거 후 행 수"] = len(d)

    keep = ~d[key_cols].isna().any(axis=1)
    removed_rows = int((~keep).sum())
    d = d[keep].reset_index(drop=True)
    info["결측 처리 후 행 수(최종)"] = len(d)

    filled = 0
    for c in d.columns:
        if c in key_cols:
            continue
        if pd.api.types.is_numeric_dtype(d[c]) and d[c].isna().any():
            filled += int(d[c].isna().sum())
            d[c] = d[c].fillna(d[c].median())

    return d, dict(counts=info, dup=dup_n, removed_rows=removed_rows, filled=filled)


def is_total_row(df, g1, g2):
    m = pd.Series(False, index=df.index)
    for c in (g1, g2):
        if c:
            m |= df[c].astype(str).str.strip().str.lower().isin(TOTAL_WORDS)
    return m


def build_std(df, g1, g2, mean_col, cnt_col, rate_col, bins, labels, total_mask):
    """분석에 쓰기 편하도록 열 이름을 통일한 표를 만든다."""
    out = pd.DataFrame(index=df.index)
    if g2:
        out[C_DIM] = df[g1].astype(str)
        out[C_GRP] = df[g2].astype(str)
    else:
        out[C_DIM] = "전체"
        out[C_GRP] = df[g1].astype(str)
    out[C_N] = df[cnt_col] if cnt_col else np.nan
    out[C_RATE] = df[rate_col] if rate_col else np.nan
    out[C_MEAN] = df[mean_col]
    for c, lab in zip(bins, labels):
        out[lab] = df[c]
    out[C_TOTAL] = total_mask.values
    return out


# ------------------------------------------------------------------
# 탭 1: 데이터 확인
# ------------------------------------------------------------------
def render_overview(ctx):
    raw, clean0, enc = ctx["raw"], ctx["clean0"], ctx["enc"]
    ci = ctx["clean_info"]

    st.header("데이터 확인과 전처리")
    st.markdown(
        "업로드한 파일이 어떤 모양인지 먼저 살펴보고, 분석하기 좋게 정리한 과정을 보여 줍니다. "
        "**전처리**란 분석 전에 자료의 공백, 자료형, 빠진 값, 겹치는 값을 정리하는 일입니다."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행(집단) 개수", f"{raw.shape[0]:,}")
    c2.metric("열 개수", f"{raw.shape[1]:,}")
    c3.metric("인코딩(글자 저장 방식)", enc)
    c4.metric("원본 결측치 개수", f"{ci['missing_total']:,}")

    st.subheader("1) 원본 데이터 미리보기")
    show_table(raw.head(10))
    st.caption("처음 10행입니다. 이 자료는 응답자 한 사람 한 사람의 답이 아니라, 집단별로 요약한 표입니다.")

    st.subheader("2) 열 이름과 데이터 유형")
    type_rows = []
    for rc, cc in zip(raw.columns, clean0.columns):
        type_rows.append({
            "열 이름": cc,
            "처리 전 유형": korean_dtype(raw[rc]),
            "처리 후 유형": korean_dtype(clean0[cc]),
            "결측치 수": int(clean0[cc].isna().sum()),
            "서로 다른 값 수": int(clean0[cc].nunique(dropna=True)),
        })
    show_table(pd.DataFrame(type_rows))

    st.subheader("3) 정리한 내용")
    st.markdown(
        f"- 문자 값의 앞뒤·중복 공백을 정리한 칸: **{ci['stripped_cells']:,}개**\n"
        f"- 숫자형으로 변환한 열: **{len(ci['converted_cols'])}개**"
        + (f" ({', '.join(ci['converted_cols'][:5])}{' 등' if len(ci['converted_cols']) > 5 else ''})"
           if ci["converted_cols"] else "")
        + f"\n- 숫자로 바꿀 수 없어 빈 값이 된 칸: **{ci['unconvertible']:,}개**"
    )

    st.subheader("4) 중복·결측치 처리 전후 비교")
    proc = ctx["proc"]
    counts = proc["counts"]
    show_table(pd.DataFrame({"단계": list(counts.keys()), "행 개수": list(counts.values())}))
    msgs = []
    msgs.append(
        f"완전히 똑같은 중복 행 **{proc['dup']}개**를 제거했습니다." if proc["dup"]
        else "완전히 똑같은 중복 행은 없었습니다."
    )
    msgs.append(
        f"핵심 열(구분, 평균 이용시간, 시간 구간 비율)에 빈 값이 있던 행 **{proc['removed_rows']}개**를 제거했습니다."
        if proc["removed_rows"] else "핵심 열에 빈 값이 있는 행은 없었습니다."
    )
    if proc["filled"]:
        msgs.append(f"그 밖의 숫자 열의 빈 칸 **{proc['filled']}개**는 열의 중앙값으로 채웠습니다.")
    st.info(" ".join(msgs))

    dup_labels = ctx["std_all"].duplicated(subset=[C_DIM, C_GRP]).sum()
    if dup_labels:
        st.warning(f"같은 '구분·집단' 이름이 겹치는 행이 {int(dup_labels)}개 있습니다. 해석할 때 주의하세요.")

    st.subheader("5) 전체 집계 행(총인원) 확인")
    n_total = int(ctx["std_all"][C_TOTAL].sum())
    if n_total:
        names = ", ".join(ctx["std_all"].loc[ctx["std_all"][C_TOTAL], C_GRP].tolist())
        st.success(
            f"전체를 한꺼번에 나타내는 집계 행 **{n_total}개**({names})를 찾았습니다. "
            "이 행은 하나의 '집단'이 아니라 모든 집단을 합친 값이므로 **집단 간 비교 분석에서는 제외**했습니다. "
            "(그래프의 '전체 평균' 기준선으로만 참고합니다.)"
        )
    else:
        st.info("'총인원', '전체' 같은 집계 행은 발견되지 않아 모든 행을 비교에 사용합니다.")

    st.subheader("6) 비교에 사용하는 집단")
    a = ctx["std"]
    grp_tbl = (
        a.groupby(C_DIM, sort=False)[C_GRP]
        .agg(["count", lambda x: ", ".join(x)])
        .reset_index()
    )
    grp_tbl.columns = ["구분", "집단 수", "집단 이름"]
    show_table(grp_tbl)
    st.caption(
        "성별·연령·지역 등은 같은 사람을 서로 다른 기준으로 나눈 것입니다. "
        "그래서 모든 구분을 한꺼번에 놓고 볼 때는 같은 사람이 여러 집단에 겹쳐 들어 있다는 점을 기억해야 합니다."
    )

    st.subheader("7) 이용 시간 구간 비율 검증")
    bins = ctx["bins"]
    sums = ctx["std_all"][bins].sum(axis=1)
    st.markdown(
        f"사용하는 구간 열: **{', '.join(bins)}**  \n"
        f"각 집단의 구간 비율 합계는 **{sums.min():.1f}% ~ {sums.max():.1f}%** 입니다."
    )
    if (sums - 100).abs().max() > 1.5:
        st.warning("일부 집단의 비율 합계가 100%에서 많이 벗어납니다. 열 선택이 올바른지 확인해 주세요.")
    else:
        st.caption("반올림 때문에 생기는 0.1%p 안팎의 차이를 빼면 합계가 100%이므로, 구간들이 전체를 나눈 자료임을 알 수 있습니다.")


# ------------------------------------------------------------------
# 탭 2: 탐구 질문 1
# ------------------------------------------------------------------
def render_q1(ctx):
    st.header("탐구 질문 1. 연령대별 하루 평균 스마트폰 이용 시간에는 어떤 차이가 있을까?")
    st.markdown(
        "연령대마다 하루에 스마트폰을 쓰는 시간이 얼마나 다른지 **평균 이용 시간(분)** 으로 비교합니다. "
        "막대가 길수록 그 연령대가 스마트폰을 더 오래 쓴다는 뜻입니다."
    )

    a = ctx["std"]
    std_all = ctx["std_all"]
    dims = list(dict.fromkeys(a[C_DIM]))
    age_dim = next((v for v in dims if "연령" in str(v)), None)
    if age_dim is None:
        st.warning("'연령' 구분을 찾지 못했습니다. 비교할 구분을 직접 골라 주세요.")
        age_dim = st.selectbox("비교할 구분", dims, key="q1_dim")
    sub = a[a[C_DIM] == age_dim].dropna(subset=[C_MEAN]).copy()
    if len(sub) < 2:
        st.warning("비교할 집단이 2개 미만이라 분석할 수 없습니다.")
        return

    vals = sub[C_MEAN].to_numpy(dtype=float)
    groups = sub[C_GRP].tolist()
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    mx, mn = vals[i_max], vals[i_min]

    tot = std_all[std_all[C_TOTAL]]
    total_mean = float(tot[C_MEAN].iloc[0]) if len(tot) else None

    c1, c2, c3 = st.columns(3)
    c1.metric("가장 오래 쓰는 집단", groups[i_max], fmt_min(mx))
    c2.metric("가장 적게 쓰는 집단", groups[i_min], fmt_min(mn))
    c3.metric("두 집단의 차이", fmt_min(mx - mn), f"약 {mx / mn:.1f}배" if mn > 0 else None, delta_color="off")

    colors = []
    for i in range(len(vals)):
        if i == i_max:
            colors.append("#1f4e9c")
        elif i == i_min:
            colors.append("#e07b39")
        else:
            colors.append("#9db7d9")
    fig = go.Figure(
        go.Bar(
            x=groups,
            y=vals,
            marker_color=colors,
            text=[f"{v:.1f}분" for v in vals],
            textposition="outside",
            hovertemplate="%{x}<br>평균 %{y:.1f}분<extra></extra>",
        )
    )
    if total_mean is not None:
        fig.add_hline(
            y=total_mean, line_dash="dash", line_color="gray",
            annotation_text=f"전체 평균 {total_mean:.1f}분", annotation_position="top right",
        )
    fig.update_layout(
        title="연령대별 하루 평균 스마트폰 이용 시간",
        xaxis_title="연령대",
        yaxis_title="하루 평균 이용 시간(분)",
        yaxis_range=[0, float(vals.max()) * 1.2],
        showlegend=False,
    )
    show_chart(fig, key="q1_bar")
    st.caption("진한 파란색: 가장 긴 집단 / 주황색: 가장 짧은 집단 / 점선: 전체 평균(비교 대상이 아닌 기준선)")

    # ---- 표
    tbl = pd.DataFrame({"연령대": groups})
    if sub[C_N].notna().any():
        tbl["사례수(명)"] = sub[C_N].round().astype("Int64").to_numpy()
    tbl["평균 이용시간(분)"] = np.round(vals, 1)
    tbl["시간·분으로 표시"] = [fmt_min(v) for v in vals]
    if total_mean is not None:
        tbl["전체 평균과의 차이(분)"] = [f"{v - total_mean:+.1f}" for v in vals]
    prev = [np.nan] + list(np.diff(vals))
    tbl["바로 앞 연령대와의 차이(분)"] = ["-" if pd.isna(v) else f"{v:+.1f}" for v in prev]
    show_table(tbl)

    # ---- 자동 해석
    st.subheader("핵심 해석")
    lines = [
        f"하루 평균 스마트폰 이용 시간이 가장 긴 연령대는 **{groups[i_max]}**{josa_ro(groups[i_max])[len(groups[i_max]):]} **{fmt_min(mx)}({mx:.1f}분)**, "
        f"가장 짧은 연령대는 **{groups[i_min]}**{josa_ro(groups[i_min])[len(groups[i_min]):]} **{fmt_min(mn)}({mn:.1f}분)** 입니다. "
        f"두 집단의 차이는 약 **{mx - mn:.1f}분**"
        + (f"으로, {josa(groups[i_max], '이', '가')} {groups[i_min]}의 약 **{mx / mn:.1f}배**입니다." if mn > 0 else "입니다.")
    ]

    after = vals[i_max:]
    before = vals[: i_max + 1]
    if len(after) >= 3 and np.all(np.diff(after) <= 0) and i_max > 0:
        lines.append(
            f"**{groups[i_max]}** 때 이용 시간이 가장 길고, 그 이후에는 연령대가 높아질수록 이용 시간이 계속 줄어드는 모습입니다."
        )
    elif len(after) >= 3 and np.all(np.diff(after) <= 0):
        lines.append("가장 어린 연령대부터 연령대가 높아질수록 이용 시간이 계속 줄어드는 모습입니다.")
    if len(before) >= 3 and np.all(np.diff(before) >= 0) and i_max > 0:
        lines.append(f"{groups[i_max]}까지는 연령대가 올라갈수록 이용 시간이 늘어납니다.")

    d = np.diff(vals)
    k = int(np.argmax(np.abs(d)))
    lines.append(
        f"바로 이웃한 연령대끼리 비교하면 **{groups[k]} → {groups[k + 1]}** 구간에서 "
        f"{'늘어난' if d[k] > 0 else '줄어든'} 폭이 **{abs(d[k]):.1f}분**으로 가장 큽니다."
    )

    if total_mean is not None:
        above = [g for g, v in zip(groups, vals) if v > total_mean]
        below = [g for g, v in zip(groups, vals) if v < total_mean]
        if above:
            lines.append(f"전체 평균({total_mean:.1f}분)보다 오래 쓰는 연령대: **{', '.join(above)}**.")
        if below:
            lines.append(f"전체 평균보다 적게 쓰는 연령대: **{', '.join(below)}**.")

    if sub[C_RATE].notna().any():
        low = sub.loc[sub[C_RATE].astype(float).idxmin()]
        if float(low[C_RATE]) < 90:
            lines.append(
                f"참고로 **{low[C_GRP]}**{josa(low[C_GRP], '은', '는')[len(str(low[C_GRP]).strip()):]} 스마트폰 이용률이 {float(low[C_RATE]):.1f}%로 다른 집단보다 낮아, "
                "스마트폰을 쓰는 사람 자체가 상대적으로 적다는 점도 함께 생각해 볼 수 있습니다."
            )

    st.info("\n\n".join(lines))
    st.caption(
        "이 자료에는 응답자 개인의 값이 없고 집단별 평균만 있습니다. 그래서 ANOVA 같은 통계 검정은 할 수 없고, "
        "집단 평균의 격차 크기로 차이를 비교했습니다. 사례수(표본 크기)가 작은 집단은 평균이 조금 더 흔들릴 수 있습니다."
    )


# ------------------------------------------------------------------
# 탭 3: 탐구 질문 2
# ------------------------------------------------------------------
def corr_table(d, bins):
    rows = []
    y = d[C_MEAN].astype(float)
    for b in bins:
        x = d[b].astype(float)
        if x.nunique() < 2 or y.nunique() < 2 or len(d) < 3:
            rows.append((b, np.nan, np.nan, np.nan))
            continue
        r, p = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)
        rows.append((b, r, rho, p))
    return pd.DataFrame(rows, columns=["구간", "r", "rho", "p"])


def render_q2(ctx):
    st.header("탐구 질문 2. 이용 시간대별 비율과 평균 이용 시간 사이에는 어떤 관계가 있을까?")
    bins = ctx["bins"]
    a = ctx["std"]

    st.markdown(
        "이 자료에서 **이용 시간대**는 '오전·오후' 같은 시각이 아니라, **하루 이용 시간의 길이 구간**"
        f"({', '.join(bins)})을 뜻합니다. 각 집단에서 이 구간에 속한 사람의 비율이 얼마나 되는지, "
        "그 비율이 높을수록 집단의 **평균 이용 시간**이 어떻게 달라지는지 살펴봅니다."
    )
    st.info(
        "**꼭 알아둘 점:** 구간별 비율을 모두 더하면 100%입니다. 한 구간의 비율이 늘어나면 다른 구간의 비율은 "
        "줄어들 수밖에 없습니다. 그래서 긴 시간 구간과 짧은 시간 구간의 결과가 서로 **반대 방향**으로 나오는 것은 자연스러운 일입니다."
    )

    ALL = "전체 집단 (모든 구분을 함께)"
    dims = list(dict.fromkeys(a[C_DIM]))
    scope = st.selectbox("분석할 집단 범위", [ALL] + dims, key="q2_scope")
    d = a if scope == ALL else a[a[C_DIM] == scope]
    d = d.dropna(subset=[C_MEAN] + bins).copy()
    d["표시이름"] = (d[C_DIM] + " · " + d[C_GRP]) if scope == ALL else d[C_GRP]
    n = len(d)
    if n < 4:
        st.warning(f"선택한 범위의 집단이 {n}개뿐이라 관계를 분석하기 어렵습니다. 다른 범위를 골라 주세요.")
        return
    if n < 8:
        st.warning(f"집단이 {n}개로 적어서 결과는 '참고용'으로만 보세요. 점 몇 개가 달라져도 결과가 크게 바뀔 수 있습니다.")
    if scope == ALL:
        st.caption(
            f"총 {n}개 집단을 사용합니다. 같은 사람이 성별·연령·직업 등 여러 집단에 겹쳐 있으므로, "
            "아래의 p값은 정확한 검정값이 아니라 참고용입니다."
        )

    # ---------------- (1) 상관계수 막대그래프
    st.subheader("1) 구간별 비율과 평균 이용 시간의 상관관계")
    st.markdown(
        "**상관계수(r)** 는 두 값이 함께 움직이는 정도를 -1~+1로 나타낸 값입니다. "
        "+1에 가까우면 '비율이 높을수록 평균 이용 시간도 길다', -1에 가까우면 '비율이 높을수록 평균 이용 시간은 짧다', "
        "0에 가까우면 뚜렷한 관계가 없다는 뜻입니다."
    )
    ct = corr_table(d, bins)
    figc = go.Figure(
        go.Bar(
            x=ct["구간"],
            y=ct["r"],
            marker_color=["#1f4e9c" if (pd.notna(v) and v >= 0) else "#d65f5f" for v in ct["r"]],
            text=[("-" if pd.isna(v) else f"{v:+.2f}") for v in ct["r"]],
            textposition="outside",
            hovertemplate="%{x}<br>r = %{y:.2f}<extra></extra>",
        )
    )
    figc.update_layout(
        title="이용 시간 구간별 비율과 평균 이용 시간의 상관계수",
        xaxis_title="이용 시간 구간",
        yaxis_title="상관계수(r)",
        yaxis_range=[-1.2, 1.2],
        showlegend=False,
    )
    figc.add_hline(y=0, line_color="gray")
    show_chart(figc, key="q2_corr")
    st.caption("파란색: 비율이 높을수록 평균 이용 시간이 긴 경향(+) / 빨간색: 비율이 높을수록 평균 이용 시간이 짧은 경향(-)")

    valid = ct.dropna(subset=["r"])
    lines = []
    if len(valid):
        best = valid.loc[valid["r"].idxmax()]
        worst = valid.loc[valid["r"].idxmin()]
        if best["r"] > 0.2:
            lines.append(
                f"**'{best['구간']}'** 비율이 높은 집단일수록 평균 이용 시간이 길어지는 경향이 있습니다. "
                f"(r = {best['r']:+.2f}, {strength_word(best['r'])} 양(+)의 상관)"
            )
        if worst["r"] < -0.2:
            lines.append(
                f"반대로 **'{worst['구간']}'** 비율이 높은 집단일수록 평균 이용 시간이 짧아지는 경향이 있습니다. "
                f"(r = {worst['r']:+.2f}, {strength_word(worst['r'])} 음(-)의 상관)"
            )
        weak = valid[valid["r"].abs() < 0.4]["구간"].tolist()
        if weak:
            lines.append(
                f"{', '.join(weak)} 구간은 상관이 약해서(|r| < 0.4), 이 구간의 비율만으로는 평균 이용 시간을 설명하기 어렵습니다."
            )
        lines.append(
            "비율의 합이 100%이기 때문에, 긴 시간 구간(예: 가장 오른쪽 구간)에 사람이 많이 몰린 집단은 "
            "짧은 시간 구간에는 사람이 적을 수밖에 없습니다. 그래서 양의 상관과 음의 상관이 양쪽 끝에서 대칭처럼 나타납니다."
        )
    if lines:
        st.subheader("핵심 해석")
        st.info("\n\n".join(lines))
    with st.expander("상관계수 표로 보기"):
        t = ct.copy()
        t["관계 세기"] = [("-" if pd.isna(r) else strength_word(r)) for r in t["r"]]
        t["방향"] = [("-" if pd.isna(r) else ("양(+)" if r >= 0 else "음(-)")) for r in t["r"]]
        t = t.rename(columns={"r": "피어슨 r", "rho": "스피어만 ρ(순위 상관)", "p": "p값(참고용)"})
        for c in ["피어슨 r", "스피어만 ρ(순위 상관)"]:
            t[c] = t[c].round(3)
        t["p값(참고용)"] = t["p값(참고용)"].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
        show_table(t)

    # ---------------- (2) 산점도
    st.subheader("2) 구간 하나를 골라 산점도로 확인하기")
    pick = st.selectbox("살펴볼 이용 시간 구간", bins, index=len(bins) - 1, key="q2_bin")
    x = d[pick].astype(float)
    y = d[C_MEAN].astype(float)
    color_col = C_DIM if scope == ALL else "표시이름"
    size_col = None
    if d[C_N].notna().all():
        size_col = C_N
    figs = px.scatter(
        d, x=pick, y=C_MEAN, color=color_col, size=size_col, size_max=22,
        hover_name="표시이름",
        labels={pick: f"'{pick}' 비율(%)", C_MEAN: "평균 이용 시간(분)", C_DIM: "구분"},
        title=f"'{pick}' 비율과 평균 이용 시간",
    )
    if x.nunique() > 1:
        res = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        figs.add_trace(
            go.Scatter(
                x=xs, y=res.intercept + res.slope * xs, mode="lines",
                line=dict(color="black", dash="dash"), name="추세선", hoverinfo="skip",
            )
        )
    figs.update_layout(xaxis_title=f"'{pick}' 비율(%)", yaxis_title="평균 이용 시간(분)")
    show_chart(figs, key="q2_scatter")
    st.caption("점 하나는 집단 하나입니다." + (" 점이 클수록 사례수(조사 인원)가 많은 집단입니다." if size_col else "") + " 점선은 전체 경향을 나타내는 추세선입니다.")

    if x.nunique() > 1:
        res = stats.linregress(x, y)
        r = res.rvalue
        direction = "길어지는" if res.slope > 0 else "짧아지는"
        txt = (
            f"**'{pick}'** 비율이 1%p(퍼센트포인트) 높아질 때 평균 이용 시간은 대략 **{abs(res.slope):.1f}분 {direction}** 경향입니다. "
            f"상관계수는 r = {r:+.2f}({strength_word(r)} 상관)이고, 이 구간의 비율이 평균 이용 시간 차이의 약 "
            f"**{r ** 2 * 100:.0f}%** 를 설명합니다(결정계수 R² = {r ** 2:.2f}). "
        )
        if abs(r) >= 0.6:
            txt += "점들이 추세선 가까이에 모여 있어 관계가 비교적 분명합니다."
        elif abs(r) >= 0.4:
            txt += "점들이 어느 정도 흩어져 있어, 경향은 보이지만 예외도 꽤 있습니다."
        else:
            txt += "점들이 넓게 흩어져 있어 뚜렷한 관계라고 말하기 어렵습니다."
        txt += " 다만 상관관계가 있다고 해서 '원인과 결과'라고 단정할 수는 없습니다."
        st.info(txt)
    else:
        st.warning("선택한 구간의 비율이 모든 집단에서 같아 추세선을 그릴 수 없습니다.")

    # ---------------- (3) 100% 누적 막대
    st.subheader("3) 집단별 시간대 구성(100%)과 평균 이용 시간")
    ds = d.sort_values(C_MEAN).copy()
    note = ""
    if len(ds) > 20:
        ds = pd.concat([ds.head(10), ds.tail(10)])
        note = f" (집단이 많아 평균 이용 시간이 가장 짧은 10개와 가장 긴 10개 집단만 표시했습니다.)"
    ds["표시"] = ds["표시이름"] + " (" + ds[C_MEAN].round(0).astype(int).astype(str) + "분)"
    long = ds.melt(id_vars=["표시"], value_vars=bins, var_name="이용 시간 구간", value_name="비율(%)")
    try:
        k = len(bins)
        palette = px.colors.sample_colorscale("Blues", [0.15 + 0.8 * i / max(k - 1, 1) for i in range(k)])
    except Exception:
        palette = px.colors.qualitative.Set2
    figb = px.bar(
        long, x="비율(%)", y="표시", color="이용 시간 구간", orientation="h",
        category_orders={"이용 시간 구간": bins, "표시": ds["표시"].tolist()},
        color_discrete_sequence=palette,
        title="집단별 이용 시간 구간 비율 (아래쪽일수록 평균 이용 시간이 짧음 → 위쪽일수록 김)",
    )
    figb.update_layout(
        barmode="stack", xaxis_title="비율(%)", yaxis_title="",
        height=max(380, 26 * len(ds) + 140), legend_title_text="이용 시간 구간",
    )
    show_chart(figb, key="q2_stack")
    st.caption("막대 전체 길이는 100%입니다. 이름 옆 괄호는 그 집단의 평균 이용 시간입니다." + note)

    kk = max(1, len(d) // 3)
    ranked = d.sort_values(C_MEAN)
    low_g, high_g = ranked.head(kk), ranked.tail(kk)
    first_b, last_b = bins[0], bins[-1]
    st.info(
        f"평균 이용 시간이 **긴 상위 {kk}개 집단**은 '{last_b}' 비율이 평균 **{high_g[last_b].mean():.1f}%**, "
        f"'{first_b}' 비율이 평균 **{high_g[first_b].mean():.1f}%** 입니다. "
        f"반대로 평균 이용 시간이 **짧은 하위 {kk}개 집단**은 '{last_b}' 비율이 **{low_g[last_b].mean():.1f}%**, "
        f"'{first_b}' 비율이 **{low_g[first_b].mean():.1f}%** 입니다. "
        "즉, 오래 쓰는 집단은 긴 시간 구간 쪽으로 사람이 쏠려 있고, 적게 쓰는 집단은 짧은 시간 구간에 사람이 많이 모여 있습니다."
    )

    # ---------------- (4) 한 걸음 더
    with st.expander("한 걸음 더: 비율만으로 평균을 얼마나 맞힐 수 있을까?"):
        mids = ctx["bin_mid"]
        if all(b in mids for b in bins):
            w = d[bins].astype(float)
            est = (w * np.array([mids[b] for b in bins])).sum(axis=1) / w.sum(axis=1)
            actual = d[C_MEAN].astype(float)
            r_est = np.corrcoef(est, actual)[0, 1] if est.nunique() > 1 else np.nan
            mae = float(np.mean(np.abs(est - actual)))
            lo, hi = float(min(est.min(), actual.min())), float(max(est.max(), actual.max()))
            fige = px.scatter(
                x=est, y=actual, hover_name=d["표시이름"],
                labels={"x": "구간 비율로 계산한 추정 평균(분)", "y": "실제 평균 이용 시간(분)"},
                title="구간 비율로 계산한 추정 평균 vs 실제 평균",
            )
            fige.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                      line=dict(color="gray", dash="dash"), name="일치선", hoverinfo="skip"))
            show_chart(fige, key="q2_est")
            mid_txt = ", ".join(f"{b} → {mids[b]:.0f}분" for b in bins)
            st.markdown(
                f"각 구간의 가운데 값({mid_txt}; 마지막 구간은 한 시간 더 긴 값으로 가정)에 구간 비율을 곱해 "
                "평균을 직접 계산해 보았습니다."
            )
            if pd.notna(r_est):
                st.info(
                    f"계산한 추정 평균과 실제 평균의 상관계수는 **{r_est:.2f}**, 평균적인 오차는 약 **{mae:.1f}분**입니다. "
                    "점이 점선에 가까울수록 '시간대별 비율이 평균 이용 시간을 거의 결정한다'는 뜻입니다. "
                    "평균은 결국 구간별 비율로 만들어지는 값이기 때문에 둘의 관계가 강하게 나오는 것은 자연스럽습니다."
                )
        else:
            st.caption("구간 열 이름에서 시간 범위를 읽지 못해 이 계산은 생략했습니다.")

        st.markdown("---")
        st.markdown(
            f"**여러 구간을 동시에 고려한 선형 회귀(scikit-learn)** — 비율의 합이 100%이므로 "
            f"가장 짧은 구간 **'{bins[0]}'** 을 기준으로 두고 나머지 구간만 사용합니다."
        )
        if len(bins) >= 3 and len(d) >= len(bins) + 2:
            Xr = d[bins[1:]].astype(float).to_numpy()
            yr = d[C_MEAN].astype(float).to_numpy()
            lr = LinearRegression().fit(Xr, yr)
            r2 = lr.score(Xr, yr)
            coef = pd.DataFrame({
                "구간": bins[1:],
                f"'{bins[0]}' 대신 1%p 늘면 평균 변화(분)": np.round(lr.coef_, 2),
            })
            show_table(coef)
            st.info(
                f"이 식은 평균 이용 시간 차이의 약 **{r2 * 100:.0f}%**(R² = {r2:.2f})를 설명합니다. "
                f"예를 들어 '{bins[0]}' 비율이 줄고 '{bins[-1]}' 비율이 1%p 늘면 평균 이용 시간은 약 "
                f"**{lr.coef_[-1]:+.1f}분** 달라지는 것으로 계산됩니다. 집단 수가 적고 구간들이 서로 얽혀 있어 "
                "계수 하나하나를 정확한 값으로 믿기보다 '방향과 대략적인 크기'로 참고하세요."
            )
        else:
            st.caption("집단 수가 부족해 회귀 분석은 생략했습니다.")


# ------------------------------------------------------------------
# 탐구 질문 3~5 공통 도우미: 이용 빈도 열 찾기, 점수 계산
# ------------------------------------------------------------------
# 이용 빈도 응답을 '자주 → 드물게' 순서로 정렬 (실제 의미에 맞는 순서)
FREQ_LEVELS = ["매일", "주 5-6일", "주 3-4일", "주 1-2일", "월 1-3일", "2-3달에 2일 이하", "전혀안함"]
# 한 달 기준 대표 이용일수 (주 5-6일 → 5.5일 × 4주 = 22일처럼 구간의 가운데 값으로 환산)
FREQ_DAYS = {
    "매일": 30.0, "주 5-6일": 22.0, "주 3-4일": 14.0, "주 1-2일": 6.0,
    "월 1-3일": 2.0, "2-3달에 2일 이하": 0.5, "전혀안함": 0.0,
}
FREQ_HEAVY = FREQ_LEVELS[:3]            # '자주 이용' = 주 3일 이상
Q3_KEYWORDS = ["OTT", "뉴스", "SNS"]    # 탐구 질문 3에서 비교할 콘텐츠(열 이름에 포함된 말)
MIN_GROUPS_ABS = 30                     # 회귀 모델을 만들기 위한 최소 집단(행) 수
MIN_ROWS_PER_FEATURE = 10               # 예측에 쓰는 특성 1개당 필요한 최소 행 수(경험 규칙)
ORDINAL_HINTS = ("연령", "소득")         # 순서가 있는 구분(순위상관 참고용)
SCORE_SUFFIX = " 이용빈도 점수"
ALL_SCOPE = "전체 집단 (모든 구분을 함께)"


def detect_freq(df):
    """'콘텐츠 이름 + 이용 빈도' 형태의 열을 찾아 {콘텐츠: {levels, cols}}로 돌려준다."""
    found = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        base = re.sub(r"\s*\(.*?\)\s*$", "", str(c)).strip()
        for lv in FREQ_LEVELS:
            if base.endswith(" " + lv):
                content = base[: -(len(lv) + 1)].strip()
                if content:
                    found.setdefault(content, {})[lv] = c
                break
    out = {}
    for content, m in found.items():
        if len(m) >= 4:
            levels = [lv for lv in FREQ_LEVELS if lv in m]
            out[content] = {"levels": levels, "cols": [m[lv] for lv in levels]}
    return out


def pick_q3_contents(freq_all):
    picked = []
    for kw in Q3_KEYWORDS:
        for ct in freq_all:
            if kw.upper() in ct.upper() and ct not in picked:
                picked.append(ct)
                break
    return picked


def freq_frame(ctx, content):
    """분석 대상 집단(총인원 제외)의 이용 빈도별 비율 표. 열 이름은 '매일', '주 5-6일' ... 순서."""
    info = ctx["freq_all"][content]
    f = ctx["cleaned"].loc[ctx["std"].index, info["cols"]].astype(float).copy()
    f.columns = info["levels"]
    return f


def freq_score(f):
    """각 집단의 '한 달 평균 이용일수' 추정값 (비율 × 구간 대표 이용일수의 가중평균)."""
    days = np.array([FREQ_DAYS[c] for c in f.columns], dtype=float)
    total = f.sum(axis=1).replace(0, np.nan)
    return (f * days).sum(axis=1) / total


def parse_fee(name):
    """'3만원 이상-4만원 미만' 같은 이용요금 구간의 (하한, 상한)을 만 원 단위로 읽는다."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*만원\s*이상\s*[-~]\s*(\d+(?:\.\d+)?)\s*만원", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*만원\s*미만", name)
    if m:
        return 0.0, float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*만원\s*이상", name)
    if m:
        n = float(m.group(1))
        return n, n * 1.4
    return None


def fee_score(ctx):
    cl = ctx["cleaned"]
    cols = [c for c in cl.columns
            if pd.api.types.is_numeric_dtype(cl[c]) and "만원" in str(c) and parse_fee(str(c))]
    if len(cols) < 3:
        return None
    mids = np.array([sum(parse_fee(str(c))) / 2 for c in cols], dtype=float)
    f = cl.loc[ctx["std"].index, cols].astype(float)
    return (f * mids).sum(axis=1) / f.sum(axis=1).replace(0, np.nan)


def build_features(ctx):
    """탐구 질문 5에서 예측에 쓸 수 있는 후보 특성(이용 행태 + 콘텐츠 이용 빈도)."""
    std = ctx["std"]
    F = pd.DataFrame(index=std.index)
    if std[C_RATE].notna().any():
        F["스마트폰 이용률(%)"] = std[C_RATE].astype(float)
    fee = fee_score(ctx)
    if fee is not None:
        F["평균 이용요금 추정(만원)"] = fee
    for ct in ctx["freq_all"]:
        F[f"{ct}{SCORE_SUFFIX}"] = freq_score(freq_frame(ctx, ct))
    return F


def safe_corr(x, y):
    """(피어슨 r, 스피어만 ρ, ρ의 p값). 계산할 수 없으면 NaN."""
    x = pd.Series(x, dtype=float).reset_index(drop=True)
    y = pd.Series(y, dtype=float).reset_index(drop=True)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, np.nan
    r, _ = stats.pearsonr(x, y)
    rho, p = stats.spearmanr(x, y)
    return float(r), float(rho), float(p)


def rel_text(r, pos="길어지는", neg="짧아지는"):
    return f"{strength_word(r)} {'양(+)' if r >= 0 else '음(-)'}의 상관"


# ------------------------------------------------------------------
# 탭 4: 탐구 질문 3
# ------------------------------------------------------------------
def render_q3(ctx):
    st.header("탐구 질문 3. OTT, 뉴스정보검색, SNS및메신저 이용 빈도에 따라 스마트폰 평균 이용 시간은 어떻게 달라질까?")
    contents = ctx["freq3"]
    if not contents:
        st.warning("OTT, 뉴스정보검색, SNS및메신저의 이용 빈도 열을 찾지 못했습니다. 파일의 열 이름을 확인해 주세요.")
        return
    st.markdown(
        "이 자료에는 콘텐츠마다 **'매일', '주 5-6일', … , '전혀안함'** 으로 얼마나 자주 이용하는지의 **비율(%)** 이 들어 있습니다. "
        "이 비율이 집단마다 다르므로, 어떤 콘텐츠를 자주 이용하는 사람이 많은 집단이 평균 스마트폰 이용 시간도 긴지 살펴봅니다. "
        "이용 빈도는 **자주 → 드물게** 순서(매일 → 주 5-6일 → 주 3-4일 → 주 1-2일 → 월 1-3일 → 2-3달에 2일 이하 → 전혀안함)로 정렬했습니다."
    )
    st.info(
        "**꼭 알아둘 점:** 이 자료는 개인별 기록이 아니라 집단별 비율표입니다. 그래서 '자주 이용하는 사람의 평균 이용 시간'을 직접 구할 수는 없고, "
        "**'자주 이용하는 사람의 비율이 높은 집단일수록 그 집단의 평균 이용 시간이 어떤가'** 를 비교합니다. "
        "또 각 콘텐츠의 빈도별 비율은 합이 100%라서, 한 빈도의 비율이 늘면 다른 빈도의 비율은 줄어듭니다."
    )

    a = ctx["std"]
    dims = list(dict.fromkeys(a[C_DIM]))
    scope = st.selectbox("분석할 집단 범위", [ALL_SCOPE] + dims, key="q3_scope")
    d = a if scope == ALL_SCOPE else a[a[C_DIM] == scope]
    d = d.dropna(subset=[C_MEAN]).copy()
    n = len(d)
    if n < 4:
        st.warning(f"선택한 범위의 집단이 {n}개뿐이라 관계를 분석하기 어렵습니다. 다른 범위를 골라 주세요.")
        return
    if n < 8:
        st.warning(f"집단이 {n}개로 적어서 결과는 '참고용'으로만 보세요. 점 몇 개가 달라져도 결과가 크게 바뀔 수 있습니다.")
    if scope == ALL_SCOPE:
        st.caption(
            f"총 {n}개 집단을 사용합니다. 같은 사람이 성별·연령·직업 등 여러 집단에 겹쳐 있으므로 "
            "p값 같은 통계 수치는 참고용입니다."
        )
    d["표시이름"] = (d[C_DIM] + " · " + d[C_GRP]) if scope == ALL_SCOPE else d[C_GRP]

    info = {}
    for ct in contents:
        f = freq_frame(ctx, ct).loc[d.index]
        levels = list(f.columns)
        info[ct] = dict(
            f=f, levels=levels, score=freq_score(f),
            heavy=f[[lv for lv in levels if lv in FREQ_HEAVY]].sum(axis=1),
        )

    # ---------------- (1) 자주 이용 비율 상위 vs 하위 집단
    st.subheader("1) '자주 이용'하는 사람이 많은 집단 vs 적은 집단")
    st.markdown(
        "**자주 이용 = 주 3일 이상**(매일 + 주 5-6일 + 주 3-4일)으로 보았습니다. 콘텐츠마다 이 비율이 높은 **상위 1/3 집단**과 "
        "낮은 **하위 1/3 집단**으로 나누고, 각 묶음의 평균 이용 시간(집단 평균의 평균)을 비교합니다."
    )
    if n >= 6:
        k = max(1, n // 3)
        rows = []
        for ct in contents:
            h = info[ct]["heavy"].sort_values()
            lo_idx, hi_idx = h.index[:k], h.index[-k:]
            rows.append(dict(
                ct=ct, lo=float(d.loc[lo_idx, C_MEAN].mean()), hi=float(d.loc[hi_idx, C_MEAN].mean()),
                lo_h=float(h.iloc[:k].mean()), hi_h=float(h.iloc[-k:].mean()),
                lo_names=", ".join(d.loc[lo_idx, "표시이름"]), hi_names=", ".join(d.loc[hi_idx, "표시이름"]),
            ))
        figt = go.Figure()
        figt.add_trace(go.Bar(
            name="자주 이용 비율이 낮은 집단 (하위 1/3)", x=[r["ct"] for r in rows], y=[r["lo"] for r in rows],
            marker_color="#e07b39", text=[f"{r['lo']:.1f}분" for r in rows], textposition="outside",
        ))
        figt.add_trace(go.Bar(
            name="자주 이용 비율이 높은 집단 (상위 1/3)", x=[r["ct"] for r in rows], y=[r["hi"] for r in rows],
            marker_color="#1f4e9c", text=[f"{r['hi']:.1f}분" for r in rows], textposition="outside",
        ))
        ymax = max(max(r["lo"], r["hi"]) for r in rows)
        figt.update_layout(
            barmode="group", title="콘텐츠를 자주 이용하는 비율에 따른 집단별 평균 스마트폰 이용 시간",
            xaxis_title="콘텐츠", yaxis_title="평균 이용 시간(분)", yaxis_range=[0, ymax * 1.25],
            legend=dict(orientation="h", y=-0.2),
        )
        show_chart(figt, key="q3_tertile")
        st.caption(f"집단 {n}개를 자주 이용 비율 순으로 세워 위·아래 {k}개씩 비교했습니다. 가운데 집단은 비교에서 뺐습니다.")

        lines = []
        for r in rows:
            diff = r["hi"] - r["lo"]
            lines.append(
                f"**{r['ct']}** : 주 3일 이상 이용하는 사람의 비율이 높은 집단(평균 {r['hi_h']:.0f}%)의 평균 이용 시간은 "
                f"**{r['hi']:.1f}분**, 낮은 집단(평균 {r['lo_h']:.0f}%)은 **{r['lo']:.1f}분**으로 "
                f"{'앞쪽이 ' + format(abs(diff), '.1f') + '분 더 깁니다' if diff > 0 else '앞쪽이 ' + format(abs(diff), '.1f') + '분 더 짧습니다'}."
            )
        biggest = max(rows, key=lambda r: abs(r["hi"] - r["lo"]))
        lines.append(
            f"세 콘텐츠 중 두 묶음의 차이가 가장 큰 것은 **{biggest['ct']}**(차이 {abs(biggest['hi'] - biggest['lo']):.1f}분)입니다."
        )
        st.subheader("핵심 해석")
        st.info("\n\n".join(lines))
        with st.expander("어떤 집단이 상위·하위에 들어갔는지 보기"):
            show_table(pd.DataFrame({
                "콘텐츠": [r["ct"] for r in rows],
                "하위 1/3 집단": [r["lo_names"] for r in rows],
                "상위 1/3 집단": [r["hi_names"] for r in rows],
            }))
    else:
        st.caption("집단이 6개 미만이라 상·하위 1/3 비교는 생략합니다.")

    # ---------------- (2) 콘텐츠별 이용 빈도 점수와 평균 이용 시간의 상관
    st.subheader("2) 콘텐츠별 '이용 빈도'와 평균 이용 시간의 상관관계")
    day_txt = ", ".join(f"{lv} {FREQ_DAYS[lv]:g}일" for lv in FREQ_LEVELS)
    st.markdown(
        "빈도별 비율을 한 숫자로 줄이기 위해 **'한 달 평균 이용일수 점수'** 를 만들었습니다. 각 빈도 구간의 대표 이용일수"
        f"({day_txt})에 그 비율을 곱해 평균을 낸 값입니다. 점수가 클수록 그 집단은 해당 콘텐츠를 더 자주 이용합니다. "
        "이용 빈도는 크기에 순서가 있는 자료이므로 **순위를 기준으로 하는 스피어만 상관계수(ρ)** 를 중심으로 봅니다."
    )
    corr_rows = []
    for ct in contents:
        r, rho, p = safe_corr(info[ct]["score"], d[C_MEAN])
        corr_rows.append(dict(ct=ct, r=r, rho=rho, p=p))
    cdf = pd.DataFrame(corr_rows)
    figc = go.Figure(go.Bar(
        x=cdf["ct"], y=cdf["rho"],
        marker_color=["#1f4e9c" if (pd.notna(v) and v >= 0) else "#d65f5f" for v in cdf["rho"]],
        text=[("-" if pd.isna(v) else f"{v:+.2f}") for v in cdf["rho"]], textposition="outside",
        hovertemplate="%{x}<br>ρ = %{y:.2f}<extra></extra>",
    ))
    figc.add_hline(y=0, line_color="gray")
    figc.update_layout(
        title="콘텐츠 이용 빈도 점수와 평균 이용 시간의 스피어만 상관계수",
        xaxis_title="콘텐츠", yaxis_title="상관계수(ρ)", yaxis_range=[-1.2, 1.2], showlegend=False,
    )
    show_chart(figc, key="q3_corr")
    st.caption("파란색: 자주 이용하는 집단일수록 평균 이용 시간이 긴 경향(+) / 빨간색: 반대 경향(-)")

    valid = cdf.dropna(subset=["rho"])
    if len(valid):
        lines = []
        for _, row in valid.iterrows():
            direction = "길어지는" if row["rho"] >= 0 else "짧아지는"
            lines.append(
                f"**{row['ct']}**{josa(row['ct'], '을', '를')[len(row['ct']):]} 자주 이용하는 집단일수록 평균 스마트폰 이용 시간이 {direction} 경향이 있습니다. "
                f"(ρ = {row['rho']:+.2f}, {rel_text(row['rho'])})"
            )
        best = valid.loc[valid["rho"].abs().idxmax()]
        lines.append(
            f"세 콘텐츠 중 평균 이용 시간과 가장 관련이 큰 것은 **{best['ct']}** 입니다. "
            "다만 어떤 콘텐츠를 많이 보기 때문에 폰을 오래 쓰는지, 폰을 오래 쓰기 때문에 콘텐츠도 많이 보는지는 이 자료만으로 알 수 없습니다(상관관계 ≠ 인과관계)."
        )
        if (valid["rho"].abs() < 0.2).any():
            weak = ", ".join(valid.loc[valid["rho"].abs() < 0.2, "ct"])
            lines.append(f"{josa(weak, '은', '는')} 상관이 거의 없어, 이용 빈도만으로는 평균 이용 시간 차이를 설명하기 어렵습니다.")
        st.subheader("핵심 해석")
        st.info("\n\n".join(lines))
    with st.expander("상관계수 표로 보기"):
        t = cdf.rename(columns={"ct": "콘텐츠", "r": "피어슨 r", "rho": "스피어만 ρ", "p": "ρ의 p값(참고용)"}).copy()
        t["피어슨 r"] = t["피어슨 r"].round(3)
        t["스피어만 ρ"] = t["스피어만 ρ"].round(3)
        t["ρ의 p값(참고용)"] = t["ρ의 p값(참고용)"].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
        show_table(t)

    # ---------------- (3) 콘텐츠 하나를 골라 자세히
    st.subheader("3) 콘텐츠 하나를 골라 자세히 보기")
    pick = st.selectbox("살펴볼 콘텐츠", contents, key="q3_content")
    pi = info[pick]
    sd = d.copy()
    sd["_score"] = pi["score"]
    sd = sd.dropna(subset=["_score"])
    size_col = C_N if sd[C_N].notna().all() else None
    color_col = C_DIM if scope == ALL_SCOPE else "표시이름"
    figs = px.scatter(
        sd, x="_score", y=C_MEAN, color=color_col, size=size_col, size_max=22, hover_name="표시이름",
        labels={"_score": "한 달 평균 이용일수 점수", C_MEAN: "평균 이용 시간(분)", C_DIM: "구분"},
        title=f"'{pick}' 이용 빈도 점수와 평균 스마트폰 이용 시간",
    )
    slope = None
    if sd["_score"].nunique() > 1 and len(sd) >= 3:
        res = stats.linregress(sd["_score"].astype(float), sd[C_MEAN].astype(float))
        slope = res
        xs = np.linspace(float(sd["_score"].min()), float(sd["_score"].max()), 50)
        figs.add_trace(go.Scatter(
            x=xs, y=res.intercept + res.slope * xs, mode="lines",
            line=dict(color="black", dash="dash"), name="추세선", hoverinfo="skip",
        ))
    figs.update_layout(xaxis_title=f"{pick} 한 달 평균 이용일수 점수(일)", yaxis_title="평균 이용 시간(분)")
    show_chart(figs, key="q3_scatter")
    st.caption("점 하나는 집단 하나입니다." + (" 점이 클수록 사례수가 많은 집단입니다." if size_col else "") + " 점선은 전체 경향을 나타내는 추세선입니다.")
    if slope is not None:
        r_, rho_, _ = safe_corr(sd["_score"], sd[C_MEAN])
        direction = "길어지는" if slope.slope > 0 else "짧아지는"
        txt = (
            f"**{pick}** 의 한 달 평균 이용일수 점수가 1일 높아질 때 집단의 평균 이용 시간은 대략 "
            f"**{abs(slope.slope):.1f}분 {direction}** 경향입니다(순위상관 ρ = {rho_:+.2f}, {rel_text(rho_)}). "
        )
        if abs(rho_) >= 0.6:
            txt += "점들이 추세선 가까이에 모여 있어 관계가 비교적 분명합니다."
        elif abs(rho_) >= 0.4:
            txt += "경향은 보이지만 예외가 되는 집단도 꽤 있습니다."
        else:
            txt += "점들이 넓게 흩어져 있어 뚜렷한 관계라고 말하기 어렵습니다."
        st.info(txt)

    levels = pi["levels"]
    lc = []
    for lv in levels:
        r, _, _ = safe_corr(pi["f"][lv], d[C_MEAN])
        lc.append(r)
    figl = go.Figure(go.Bar(
        x=levels, y=lc,
        marker_color=["#1f4e9c" if (pd.notna(v) and v >= 0) else "#d65f5f" for v in lc],
        text=[("-" if pd.isna(v) else f"{v:+.2f}") for v in lc], textposition="outside",
        hovertemplate="%{x}<br>r = %{y:.2f}<extra></extra>",
    ))
    figl.add_hline(y=0, line_color="gray")
    figl.update_layout(
        title=f"'{pick}' 이용 빈도 구간별 비율과 평균 이용 시간의 상관계수 (자주 → 드물게 순서)",
        xaxis_title="이용 빈도", yaxis_title="상관계수(r)", yaxis_range=[-1.2, 1.2], showlegend=False,
    )
    show_chart(figl, key="q3_levels")
    st.caption("왼쪽일수록 자주 이용하는 구간, 오른쪽일수록 드물게 이용하는 구간입니다.")
    lcs = pd.Series(lc, index=levels).dropna()
    if len(lcs):
        bp, bn = lcs.idxmax(), lcs.idxmin()
        lines = []
        if lcs[bp] > 0.2:
            lines.append(f"**'{bp}'** 이용 비율이 높은 집단일수록 평균 이용 시간이 긴 경향입니다(r = {lcs[bp]:+.2f}).")
        if lcs[bn] < -0.2:
            lines.append(f"**'{bn}'** 이용 비율이 높은 집단일수록 평균 이용 시간이 짧은 경향입니다(r = {lcs[bn]:+.2f}).")
        if not lines:
            lines.append("어느 빈도 구간도 평균 이용 시간과 뚜렷한 관계를 보이지 않습니다.")
        lines.append("비율의 합이 100%라서 '자주' 쪽과 '드물게' 쪽의 결과가 서로 반대 방향으로 나타나는 것은 자연스러운 현상입니다.")
        st.info("\n\n".join(lines))


# ------------------------------------------------------------------
# 탭 5: 탐구 질문 4
# ------------------------------------------------------------------
def render_q4(ctx):
    st.header("탐구 질문 4. 성별, 연령, 직업, 학력 등 집단 특성 중 평균 스마트폰 이용 시간과 가장 관련이 높은 요소는 무엇일까?")
    st.markdown(
        "성별·연령·직업·학력 같은 **집단의 특성**마다 평균 이용 시간이 얼마나 갈라지는지 비교합니다. "
        "특성에 따라 집단 평균이 크게 벌어질수록, 그 특성이 평균 이용 시간과 **관련이 높다**고 볼 수 있습니다."
    )
    st.info(
        "**왜 상관계수를 쓰지 않나요?** 성별·직업 같은 범주형 변수를 '남자=1, 여자=2', '학생=3'처럼 숫자로 바꾸면 "
        "숫자의 크기에 아무 의미가 없는데도 상관계수가 계산되어 버립니다. 그래서 숫자로 바꾸지 않고, "
        "**특성별로 집단 평균이 얼마나 흩어져 있는지**(분산분석 ANOVA의 '집단 간 변동'과 같은 생각)를 비교합니다. "
        "모든 특성은 같은 응답자들을 서로 다른 기준으로 나눈 것이라 전체 변동의 크기가 같으므로, 집단 평균이 더 크게 흩어진 특성이 더 설명력이 큽니다."
    )

    a = ctx["std"]
    std_all = ctx["std_all"]
    tot = std_all[std_all[C_TOTAL]]
    total_mean = float(tot[C_MEAN].iloc[0]) if len(tot) else None
    total_n = float(tot[C_N].iloc[0]) if (len(tot) and pd.notna(tot[C_N].iloc[0])) else None

    rows = []
    for dim, g in a.groupby(C_DIM, sort=False):
        g = g.dropna(subset=[C_MEAN])
        if len(g) < 2:
            continue
        m = g[C_MEAN].astype(float).to_numpy()
        use_w = g[C_N].notna().all() and (g[C_N] > 0).all()
        w = g[C_N].astype(float).to_numpy() if use_w else np.ones(len(g))
        M = float(np.average(m, weights=w))
        sd = float(np.sqrt(np.average((m - M) ** 2, weights=w)))
        imax, imin = int(np.argmax(m)), int(np.argmin(m))
        rho = np.nan
        if any(k in str(dim) for k in ORDINAL_HINTS) and len(g) >= 4:
            rho = safe_corr(np.arange(len(m)), m)[1]
        rows.append(dict(
            dim=dim, n_groups=len(g), sd=sd, gmax=g[C_GRP].iloc[imax], vmax=float(m[imax]),
            gmin=g[C_GRP].iloc[imin], vmin=float(m[imin]), gap=float(m[imax] - m[imin]),
            cover=(float(w.sum()) / total_n * 100 if (use_w and total_n) else np.nan), rho=rho,
        ))
    if len(rows) < 2:
        st.warning("비교할 수 있는 집단 특성(구분)이 2개 미만이라 분석할 수 없습니다.")
        return
    rk = pd.DataFrame(rows).sort_values("sd", ascending=False).reset_index(drop=True)
    sd_max = float(rk["sd"].max())

    # ---------------- (1) 특성별 비교
    st.subheader("1) 어떤 특성에서 집단 평균이 가장 크게 갈릴까?")
    st.markdown(
        "**집단 평균의 흩어짐**은 각 집단의 평균 이용 시간이 전체 평균에서 평균적으로 얼마나 떨어져 있는지를 "
        "(사례수가 많은 집단에 더 큰 비중을 주어) 계산한 표준편차입니다. 값이 클수록 그 특성에 따라 평균 이용 시간이 크게 달라집니다."
    )
    figr = go.Figure(go.Bar(
        x=rk["dim"], y=rk["sd"], marker_color=["#1f4e9c"] + ["#9db7d9"] * (len(rk) - 1),
        text=[f"{v:.1f}분" for v in rk["sd"]], textposition="outside",
        hovertemplate="%{x}<br>집단 평균의 흩어짐 %{y:.1f}분<extra></extra>",
    ))
    figr.update_layout(
        title="집단 특성별 평균 이용 시간의 흩어짐 (클수록 관련이 높음)",
        xaxis_title="집단 특성(구분)", yaxis_title="집단 평균의 흩어짐(분)",
        yaxis_range=[0, sd_max * 1.2], showlegend=False,
    )
    show_chart(figr, key="q4_rank")
    st.caption("진한 파란색: 평균 이용 시간과 가장 관련이 높게 나온 특성")

    tbl = pd.DataFrame({
        "순위": range(1, len(rk) + 1),
        "집단 특성": rk["dim"],
        "집단 수": rk["n_groups"],
        "집단 평균의 흩어짐(분)": rk["sd"].round(1),
        "가장 큰 요인 대비(%)": (rk["sd"] / sd_max * 100).round(0).astype(int),
        "가장 긴 집단": [f"{g} ({v:.1f}분)" for g, v in zip(rk["gmax"], rk["vmax"])],
        "가장 짧은 집단": [f"{g} ({v:.1f}분)" for g, v in zip(rk["gmin"], rk["vmin"])],
        "최고-최저 차이(분)": rk["gap"].round(1),
        "사례수 합계/총인원(%)": [("-" if pd.isna(v) else f"{v:.0f}") for v in rk["cover"]],
        "순위상관 ρ(순서 있는 특성만)": [("-" if pd.isna(v) else f"{v:+.2f}") for v in rk["rho"]],
    })
    show_table(tbl)

    top, bot = rk.iloc[0], rk.iloc[-1]
    lines = [
        f"집단 평균이 가장 크게 갈리는 특성은 **{top['dim']}** 입니다. {top['dim']}에 따라 평균 이용 시간의 흩어짐이 "
        f"**{top['sd']:.1f}분**이고, 가장 긴 집단(**{top['gmax']}**, {top['vmax']:.1f}분)과 가장 짧은 집단(**{top['gmin']}**, {top['vmin']:.1f}분)의 차이는 "
        f"**{top['gap']:.1f}분**입니다."
    ]
    if len(rk) >= 3:
        second = rk.iloc[1]
        lines.append(
            f"그다음은 **{second['dim']}**({second['sd']:.1f}분)이고, 가장 영향이 작게 나온 특성은 **{bot['dim']}**({bot['sd']:.1f}분, "
            f"1위의 약 {bot['sd'] / sd_max * 100:.0f}%)입니다. "
            f"{josa(bot['dim'], '은', '는')} 집단이 달라도 평균 이용 시간이 비슷하다는 뜻입니다."
        )
    ord_rows = rk.dropna(subset=["rho"])
    for _, r in ord_rows.iterrows():
        shape = (
            "연령이 높아질수록 이용 시간이 줄어드는 방향이 뚜렷합니다." if (r["rho"] <= -0.7 and "연령" in str(r["dim"]))
            else ("순서가 올라갈수록 평균이 늘어나는 방향입니다." if r["rho"] >= 0.7 else "순서에 따라 한 방향으로 변한다고 보기는 어렵습니다.")
        )
        lines.append(f"순서가 있는 특성인 **{r['dim']}** 의 순위상관은 ρ = {r['rho']:+.2f}입니다. {shape}")
    dim_names = rk["dim"].astype(str).tolist()
    if any("연령" in x for x in dim_names) and any(("직업" in x or "학력" in x) for x in dim_names):
        lines.append(
            "다만 연령·직업·학력은 서로 겹칠 수 있습니다. 예를 들어 학생이나 대학생·대학원생 집단은 젊은 연령층일 가능성이 높아서, "
            "한 특성의 영향이 다른 특성의 영향과 섞여 있을 수 있습니다. 어느 특성이 '원인'인지는 이 자료만으로 단정할 수 없습니다."
        )
    low_cover = rk[(rk["cover"] < 90) | (rk["cover"] > 110)] if rk["cover"].notna().any() else rk.iloc[0:0]
    if len(low_cover):
        lines.append(
            "참고: " + ", ".join(low_cover["dim"].astype(str)) + " 의 사례수 합계가 총인원과 10% 넘게 달라서, 이 특성의 흩어짐은 조금 덜 정확할 수 있습니다."
        )
    st.subheader("핵심 해석")
    st.info("\n\n".join(lines))
    st.caption(
        "개인별 값이 없어 분산분석의 F검정(통계적 유의성 검정)은 할 수 없습니다. 대신 집단 평균이 흩어진 정도를 비교했습니다. "
        "사례수가 아주 적은 집단(예: '기타')은 평균이 불안정할 수 있습니다."
    )

    # ---------------- (2) 특성 하나 골라 집단별 평균 비교
    st.subheader("2) 특성 하나를 골라 집단별 평균 이용 시간 비교")
    pick = st.selectbox("살펴볼 집단 특성", rk["dim"].tolist(), index=0, key="q4_dim")
    sub = a[a[C_DIM] == pick].dropna(subset=[C_MEAN]).copy()
    vals = sub[C_MEAN].to_numpy(dtype=float)
    groups = sub[C_GRP].tolist()
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    colors = ["#1f4e9c" if i == i_max else ("#e07b39" if i == i_min else "#9db7d9") for i in range(len(vals))]
    figb = go.Figure(go.Bar(
        x=groups, y=vals, marker_color=colors, text=[f"{v:.1f}분" for v in vals], textposition="outside",
        customdata=sub[C_N].to_numpy(dtype=float) if sub[C_N].notna().all() else None,
        hovertemplate="%{x}<br>평균 %{y:.1f}분<extra></extra>",
    ))
    if total_mean is not None:
        figb.add_hline(
            y=total_mean, line_dash="dash", line_color="gray",
            annotation_text=f"전체 평균 {total_mean:.1f}분", annotation_position="top right",
        )
    figb.update_layout(
        title=f"{pick}별 하루 평균 스마트폰 이용 시간", xaxis_title=pick, yaxis_title="하루 평균 이용 시간(분)",
        yaxis_range=[0, float(vals.max()) * 1.2], showlegend=False,
    )
    show_chart(figb, key="q4_bar")
    st.caption("진한 파란색: 가장 긴 집단 / 주황색: 가장 짧은 집단 / 점선: 전체 평균")
    gap = vals[i_max] - vals[i_min]
    txt = (
        f"{pick}에서는 **{josa(groups[i_max], '이', '가')}** **{vals[i_max]:.1f}분({fmt_min(vals[i_max])})** 으로 가장 길고, "
        f"**{josa(groups[i_min], '이', '가')}** **{vals[i_min]:.1f}분({fmt_min(vals[i_min])})** 으로 가장 짧습니다. 두 집단의 차이는 약 **{gap:.1f}분**"
    )
    txt += f"(약 {vals[i_max] / vals[i_min]:.1f}배)입니다. " if vals[i_min] > 0 else "입니다. "
    if total_mean is not None:
        above = [g for g, v in zip(groups, vals) if v > total_mean]
        if above:
            txt += f"전체 평균({total_mean:.1f}분)보다 오래 쓰는 집단은 {', '.join(above)}입니다. "
    rank_pos = int(rk.index[rk["dim"] == pick][0]) + 1
    txt += f"이 특성은 전체 {len(rk)}개 특성 중 평균 이용 시간과의 관련이 **{rank_pos}위**입니다."
    st.info(txt)


# ------------------------------------------------------------------
# 탭 6: 탐구 질문 5
# ------------------------------------------------------------------
def fit_group_model(d, feats, y_col):
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    X = feats.loc[d.index].astype(float)
    y = d[y_col].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ytr)
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    base = np.full(len(yte), float(ytr.mean()))
    return dict(
        Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, ptr=ptr, pte=pte,
        r2_tr=float(r2_score(ytr, ptr)), r2_te=float(r2_score(yte, pte)) if len(yte) >= 2 else np.nan,
        mae_te=float(mean_absolute_error(yte, pte)), mae_base=float(mean_absolute_error(yte, base)),
        coef=pd.Series(model[-1].coef_, index=X.columns),
    )


def render_q5(ctx):
    st.header("탐구 질문 5. 스마트폰 이용 행태와 콘텐츠 이용 빈도로 집단별 평균 스마트폰 이용 시간을 예측하는 회귀 모델을 만들 수 있을까?")
    st.markdown(
        "이 질문에서는 모델을 **무조건 만들지 않고**, 먼저 지금 자료로 회귀 모델을 만드는 것이 타당한지 점검합니다. "
        "점검을 통과한 경우에만 학습 데이터와 테스트 데이터를 나누어 모델을 만들고, 통과하지 못하면 그 이유를 설명합니다."
    )
    st.warning(
        "여기서 말하는 예측은 **집단별 평균 이용 시간에 대한 탐색적 예측**입니다. "
        "한 사람 한 사람의 스마트폰 이용 시간을 맞히는 모델이 아닙니다."
    )

    a = ctx["std"]
    feats = ctx["features"]
    std_all = ctx["std_all"]
    if feats is None or feats.shape[1] == 0:
        st.warning("예측에 쓸 수 있는 특성(이용률, 이용요금, 콘텐츠 이용 빈도 등)을 찾지 못해 모델 점검을 할 수 없습니다.")
        return
    n_feat = feats.shape[1]
    req = max(MIN_GROUPS_ABS, MIN_ROWS_PER_FEATURE * n_feat)

    tot = std_all[std_all[C_TOTAL]]
    total_n = float(tot[C_N].iloc[0]) if (len(tot) and pd.notna(tot[C_N].iloc[0])) else None

    complete = a[C_MEAN].notna() & feats.loc[a.index].notna().all(axis=1)
    rows = []
    for dim, g in a.groupby(C_DIM, sort=False):
        idx = g.index[complete.loc[g.index]]
        cover = float(g.loc[idx, C_N].sum()) / total_n * 100 if (total_n and g[C_N].notna().all()) else np.nan
        ok = (len(idx) >= req) and (np.isnan(cover) or 90 <= cover <= 110)
        rows.append(dict(dim=dim, n=len(idx), cover=cover, ok=ok))
    chk = pd.DataFrame(rows)
    n_dims = len(chk)
    n_pooled = int(complete.sum())
    max_n = int(chk["n"].max())
    max_dim = chk.loc[chk["n"].idxmax(), "dim"]
    eligible = chk[chk["ok"]]["dim"].tolist()

    # ---------------- (1) 모델링 전 점검
    st.subheader("1) 모델을 만들기 전 점검")
    st.markdown(
        f"예측에 쓸 수 있는 특성은 **{n_feat}개**입니다: {', '.join(feats.columns)}. "
        "(이용 시간 구간별 비율과 평균 이용시간은 '정답'을 거의 그대로 담고 있어 입력으로 쓰면 안 되므로 제외했습니다.)  \n"
        f"특성 1개당 최소 {MIN_ROWS_PER_FEATURE}행은 있어야 안정적으로 학습할 수 있다는 경험 규칙과 절대 최소 {MIN_GROUPS_ABS}행을 함께 적용하면, "
        f"**필요한 최소 집단(행) 수는 {req}개**입니다."
    )
    covers = chk["cover"].dropna()
    check_tbl = pd.DataFrame({
        "점검 항목": [
            "집계 기준(구분)이 하나뿐인가?",
            "서로 다른 기준을 합치면 같은 응답자가 중복되지 않는가?",
            "한 기준 안의 집단 수가 필요한 행 수 이상인가?",
            "특성 수에 비해 행이 충분한가?",
        ],
        "이 자료의 상태": [
            f"{n_dims}가지 기준이 섞여 있음 ({', '.join(chk['dim'].astype(str))})",
            (f"각 기준의 사례수 합이 총인원의 약 {covers.median():.0f}% → 같은 응답자가 기준마다 한 번씩, 약 {n_dims}번 집계됨"
             if len(covers) else f"기준마다 같은 응답자를 다시 나눈 표로 보여, 최대 약 {n_dims}번 중복 집계됨"),
            f"가장 많은 기준은 {max_dim}의 {max_n}개 (필요: {req}개)",
            f"특성 {n_feat}개 vs 한 기준 안의 최대 {max_n}행",
        ],
        "판정": [
            "하나의 기준만 사용해야 함",
            "합쳐서 학습하면 안 됨",
            "충족" if max_n >= req else "부족",
            "충족" if max_n >= MIN_ROWS_PER_FEATURE * n_feat else "부족",
        ],
    })
    show_table(check_tbl)

    plot_df = pd.concat([
        chk[["dim", "n"]].rename(columns={"dim": "x", "n": "y"}),
        pd.DataFrame({"x": ["모든 구분 합침(사용 불가)"], "y": [n_pooled]}),
    ], ignore_index=True)
    colors = ["#1f4e9c" if (x in eligible) else "#9db7d9" for x in plot_df["x"][:-1]] + ["#d65f5f"]
    figk = go.Figure(go.Bar(
        x=plot_df["x"], y=plot_df["y"], marker_color=colors,
        text=plot_df["y"], textposition="outside", hovertemplate="%{x}<br>%{y}개<extra></extra>",
    ))
    figk.add_hline(
        y=req, line_dash="dash", line_color="gray",
        annotation_text=f"필요한 최소 집단 수 {req}개", annotation_position="top left",
    )
    figk.update_layout(
        title="구분별 집단(행) 수와 모델 학습에 필요한 최소 행 수",
        xaxis_title="집계 기준(구분)", yaxis_title="집단(행) 수",
        yaxis_range=[0, max(req, n_pooled) * 1.25], showlegend=False,
    )
    show_chart(figk, key="q5_check")
    st.caption("파란색·연한 파란색: 하나의 구분 안의 집단 수 / 빨간색: 모든 구분을 합친 행 수(합치면 안 되는 이유는 아래 설명)")

    lines = [
        f"가장 집단이 많은 구분도 **{max_dim}의 {max_n}개**뿐이라, 필요한 **{req}개**에 못 미칩니다."
        if max_n < req else
        f"집단 수가 필요한 {req}개 이상인 구분이 있어 모델 학습을 시도할 수 있습니다.",
        f"모든 구분을 합치면 {n_pooled}개로 늘어나는 것처럼 보이지만, 성별·연령·지역·직업 같은 구분은 **같은 응답자를 서로 다른 기준으로 나눈 표**입니다. "
        f"합치면 같은 사람이 여러 행에 반복해서 들어가 행들이 서로 독립이 아니게 되고, 학습·테스트 데이터를 나눠도 같은 사람이 양쪽에 섞여 "
        "성능이 실제보다 좋게 보이는 문제(데이터 누수)가 생깁니다. 그래서 합치지 않았습니다.",
    ]
    st.subheader("핵심 해석")
    st.info("\n\n".join(lines))

    # ---------------- (2) 결론 / 모델
    if not eligible:
        st.subheader("2) 결론: 지금 자료로는 회귀 모델을 만들지 않습니다")
        st.error(
            "점검을 통과한 구분이 없어서 **예측 모델을 학습하지 않았습니다.** 억지로 만든 모델은 숫자(R² 등)가 나오더라도 믿을 수 없기 때문입니다."
        )
        st.markdown(
            "**한계**\n\n"
            f"- 자료가 개인별이 아니라 **집단별로 집계된 통계표**이고, 하나의 기준 안에서는 집단이 최대 {max_n}개뿐입니다.\n"
            f"- 예측에 쓸 특성은 {n_feat}개인데 행이 이보다 적거나 비슷해, 모델이 우연한 패턴까지 외워 버립니다(과적합).\n"
            "- 서로 다른 기준의 행을 합치면 같은 응답자가 중복 집계되어 모델 평가가 왜곡됩니다.\n"
            "- 집단의 평균을 집단의 평균으로 설명하는 것이라, 집단 수준의 관계를 개인에게 그대로 적용할 수 없습니다(생태학적 오류).\n\n"
            "**그러면 어떤 자료가 있으면 가능할까요?**\n\n"
            f"- 응답자 개인별 원자료(수천 행)가 있으면, 개인의 이용 시간을 설명하는 회귀 모델을 학습·검증할 수 있습니다.\n"
            f"- 또는 '연령 × 성별 × 직업'처럼 더 잘게 나눈 **같은 기준의 집계표**로 집단이 {req}개 이상 확보되면 가능합니다.\n\n"
            "**지금 자료로 할 수 있는 분석**\n\n"
            "- 탐구 질문 2·3에서 이용 시간 구간 비율, 콘텐츠 이용 빈도와 평균 이용 시간의 **상관관계**를 확인했습니다.\n"
            "- 탐구 질문 4에서 어떤 집단 특성에서 평균이 크게 갈리는지 비교했습니다."
        )
        return

    st.subheader("2) 점검을 통과한 구분으로 만든 탐색적 예측 모델")
    pick = st.selectbox("모델을 만들 구분(한 가지 기준만 사용)", eligible, key="q5_dim")
    d = a[(a[C_DIM] == pick) & complete]
    st.markdown(
        f"**{pick}** 구분의 집단 {len(d)}개만 사용합니다. 학습용 70%, 테스트용 30%로 나누어(무작위, 고정 시드) "
        "표준화 후 릿지 회귀(Ridge)를 학습했고, 성능은 **테스트 데이터로만** 평가합니다."
    )
    try:
        res = fit_group_model(d, feats, C_MEAN)
    except Exception as e:  # noqa: BLE001
        st.error(f"모델을 학습하는 중 문제가 발생했습니다: {e}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("테스트 결정계수 R²", "-" if pd.isna(res["r2_te"]) else f"{res['r2_te']:.2f}")
    m2.metric("테스트 평균 절대 오차 MAE", f"{res['mae_te']:.1f}분")
    m3.metric("평균값만 쓸 때의 MAE(비교 기준)", f"{res['mae_base']:.1f}분")

    figp = go.Figure()
    figp.add_trace(go.Scatter(
        x=res["ytr"], y=res["ptr"], mode="markers", name="학습 데이터",
        marker=dict(color="#9db7d9", size=9), text=d.loc[res["ytr"].index, C_GRP],
    ))
    figp.add_trace(go.Scatter(
        x=res["yte"], y=res["pte"], mode="markers", name="테스트 데이터",
        marker=dict(color="#1f4e9c", size=11), text=d.loc[res["yte"].index, C_GRP],
    ))
    lo = float(min(res["ytr"].min(), res["yte"].min(), res["ptr"].min(), res["pte"].min()))
    hi = float(max(res["ytr"].max(), res["yte"].max(), res["ptr"].max(), res["pte"].max()))
    figp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽히 맞힌 경우",
                              line=dict(color="gray", dash="dash"), hoverinfo="skip"))
    figp.update_layout(
        title="집단별 평균 이용 시간: 실제값 vs 예측값", xaxis_title="실제 평균 이용 시간(분)",
        yaxis_title="예측 평균 이용 시간(분)", legend=dict(orientation="h", y=-0.2),
    )
    show_chart(figp, key="q5_pred")
    st.caption("점이 점선에 가까울수록 예측이 정확합니다. 진한 파란색 점(테스트 데이터)이 모델의 진짜 성적표입니다.")

    msg = []
    if pd.notna(res["r2_te"]):
        msg.append(
            f"테스트 데이터에서 결정계수 R²는 **{res['r2_te']:.2f}**, 평균 절대 오차는 **{res['mae_te']:.1f}분**입니다. "
            f"집단 평균만 계속 찍는 단순한 방법의 오차({res['mae_base']:.1f}분)와 비교하면 "
            + ("모델이 더 낫습니다." if res["mae_te"] < res["mae_base"] else "모델이 더 낫다고 보기 어렵습니다.")
        )
        if res["r2_tr"] - res["r2_te"] > 0.3:
            msg.append(f"학습 R²({res['r2_tr']:.2f})보다 테스트 R²가 많이 낮아 과적합이 의심됩니다.")
    msg.append(
        f"테스트 데이터는 {len(res['yte'])}개 집단뿐이라 이 숫자는 크게 흔들릴 수 있는 **탐색적 결과**입니다. "
        "개인의 이용 시간을 예측한다는 뜻이 아니라, 집단의 평균 이용 시간을 어림해 본 것입니다."
    )
    st.info("\n\n".join(msg))

    coef = res["coef"].sort_values()
    figf = go.Figure(go.Bar(
        x=coef.values, y=coef.index, orientation="h",
        marker_color=["#1f4e9c" if v >= 0 else "#d65f5f" for v in coef.values],
        text=[f"{v:+.1f}" for v in coef.values], textposition="outside",
    ))
    figf.update_layout(
        title="특성이 1표준편차 커질 때 예측 평균 이용 시간의 변화(분)", xaxis_title="계수(분)", yaxis_title="",
        showlegend=False, height=max(320, 40 * len(coef) + 140),
    )
    show_chart(figf, key="q5_coef")
    top_f = coef.abs().sort_values(ascending=False).index[0]
    st.info(
        f"모델이 평균 이용 시간을 어림할 때 가장 크게 반영한 특성은 **{top_f}** 입니다"
        f"(계수 {coef[top_f]:+.1f}분). 특성들끼리 서로 비슷하게 움직이는 경우가 많아, 계수 하나하나를 정확한 영향으로 해석하기보다 "
        "'방향과 대략적인 크기'로만 참고하세요."
    )


# ------------------------------------------------------------------
# 머신러닝 모델 평가 (분류: 집단의 평균 이용 시간이 '높음'인가 '낮음'인가)
# ------------------------------------------------------------------
ML_CLASS_NAMES = ["낮음", "높음"]
ML_BASELINE = "기준선(다수 클래스로만 예측)"


def ml_model_factories():
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return {
        "로지스틱 회귀": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "의사결정나무(깊이 3)": lambda: DecisionTreeClassifier(max_depth=3, random_state=0),
        "랜덤 포레스트(깊이 3)": lambda: RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        ML_BASELINE: lambda: DummyClassifier(strategy="most_frequent"),
    }


def cv_predict(factory, X, y, splits):
    """교차검증: 각 행은 자신이 빠진(테스트로 쓰인) 학습에서 나온 예측값만 받는다."""
    pred = np.full(len(y), -1, dtype=int)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:          # 학습 데이터에 한 종류만 있으면 그 종류로 예측
            pred[te] = y[tr][0]
            continue
        model = factory().fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred


def cls_metrics(y, pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "정확도": float(accuracy_score(y, pred)),
        "정밀도": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "재현율": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "F1 점수": float(f1_score(y, pred, pos_label=1, zero_division=0)),
    }


def ml_feasibility(ctx):
    """탐구 질문 5와 같은 기준으로, 한 가지 구분만으로 모델을 평가해도 되는지 점검한다."""
    a, feats, std_all = ctx["std"], ctx["features"], ctx["std_all"]
    n_feat = feats.shape[1]
    req = max(MIN_GROUPS_ABS, MIN_ROWS_PER_FEATURE * n_feat)
    tot = std_all[std_all[C_TOTAL]]
    total_n = float(tot[C_N].iloc[0]) if (len(tot) and pd.notna(tot[C_N].iloc[0])) else None
    complete = a[C_MEAN].notna() & feats.loc[a.index].notna().all(axis=1)
    rows = []
    for dim, g in a.groupby(C_DIM, sort=False):
        idx = g.index[complete.loc[g.index]]
        cover = float(g.loc[idx, C_N].sum()) / total_n * 100 if (total_n and g[C_N].notna().all()) else np.nan
        ok = (len(idx) >= req) and (np.isnan(cover) or 90 <= cover <= 110)
        rows.append(dict(dim=dim, n=len(idx), ok=ok))
    return n_feat, req, pd.DataFrame(rows), complete


def render_ml(ctx):
    st.header("머신러닝 모델 평가: 집단의 평균 이용 시간이 '높은지·낮은지' 분류하기")
    st.markdown(
        "정확도·정밀도·재현율·F1 점수는 **분류 모델**을 평가하는 지표입니다. 그래서 이 탭에서는 평균 스마트폰 이용 시간 자체를 맞히는 대신, "
        "**집단의 평균 이용 시간이 중앙값 이상이면 '높음', 미만이면 '낮음'** 으로 나누고 이를 맞히는 모델을 평가합니다. "
        "입력(특성)은 스마트폰 이용률, 이용요금, TV·OTT·뉴스정보검색·SNS및메신저 이용 빈도 점수입니다."
    )
    st.warning(
        "이 모델은 **집단별 평균**이 높은지 낮은지를 어림하는 탐색용 모델입니다. 한 사람 한 사람의 이용 시간을 예측하는 모델이 아닙니다."
    )

    feats = ctx["features"]
    if feats is None or feats.shape[1] == 0:
        st.warning("예측에 쓸 수 있는 특성을 찾지 못해 모델을 평가할 수 없습니다.")
        return
    a = ctx["std"]
    n_feat, req, chk, complete = ml_feasibility(ctx)
    eligible = chk[chk["ok"]]["dim"].tolist()
    max_n = int(chk["n"].max())
    max_dim = chk.loc[chk["n"].idxmax(), "dim"]

    st.subheader("1) 모델을 평가해도 되는 자료인지 점검")
    st.markdown(
        f"**모델 선택 이유:** 행이 적고(최대 {int(complete.sum())}행) 특성이 {n_feat}개의 숫자뿐이며 서로 비슷하게 움직이는 것이 많아서, "
        "복잡한 모델보다 **단순하고 안정적인 로지스틱 회귀**를 기본 모델로 권합니다. "
        "의사결정나무와 랜덤 포레스트는 비교용으로 함께 평가하고, 아무것도 학습하지 않은 **기준선**과도 비교해 모델이 실제로 도움이 되는지 확인합니다."
    )

    if eligible:
        mode = "single"
        dim = st.selectbox("평가할 구분(한 가지 기준만 사용)", eligible, key="ml_dim")
        d = a[(a[C_DIM] == dim) & complete].copy()
        st.success(f"**{dim}** 구분의 집단 {len(d)}개(필요 {req}개 이상)로 평가합니다. 한 가지 집계 기준만 쓰므로 응답자 중복 문제가 없습니다.")
        eval_note = "집단을 하나씩 빼고 나머지로 학습하는 방식(Leave-One-Out)으로 평가했습니다."
    else:
        st.error(
            f"**타당한 평가 조건을 충족하지 못했습니다.** 한 가지 기준 안에서 집단이 가장 많은 구분은 {max_dim}의 {max_n}개뿐인데, "
            f"특성 {n_feat}개에는 최소 {req}개가 필요합니다. 성별·연령·직업 등을 합치면 54행 가까이 되지만, 같은 응답자가 기준마다 한 번씩 "
            "중복해서 들어 있어 행들이 서로 독립이 아닙니다. (탐구 질문 5와 같은 이유입니다.)"
        )
        explore = st.checkbox(
            "그래도 학습·시연용으로 모든 구분을 합쳐 평가해 보기 (결과의 신뢰도 낮음)", value=False, key="ml_explore",
        )
        if not explore:
            st.info(
                "체크하면 모든 구분을 합친 탐색용 평가를 보여 드립니다. 이때는 **구분 단위 교차검증**을 사용합니다. "
                "성별 행 전체, 연령 행 전체처럼 한 구분을 통째로 테스트로 빼고 나머지 구분으로 학습해, 같은 구분의 행이 학습과 테스트에 섞이는 것만큼은 막습니다. "
                "그래도 같은 응답자가 다른 구분에 남아 있어서 성능이 실제보다 좋게 나올 수 있습니다."
            )
            return
        mode = "pooled"
        d = a[complete].copy()
        st.warning(
            f"**탐색용 결과입니다.** {len(d)}개 집단은 같은 응답자를 여러 기준으로 나눈 것이라 서로 독립이 아니며, 집단 수도 적습니다. "
            "아래 점수는 '새로운 자료에서도 이 정도 성능이 나온다'는 증거가 아니라, 분류 평가 방법을 연습하고 이해하기 위한 예시로만 보세요."
        )
        eval_note = "구분(성별, 연령, 직업 등) 단위 교차검증으로 평가했습니다. 한 구분을 통째로 테스트에 쓰고 나머지로 학습합니다."

    n = len(d)
    if n < 10:
        st.warning(f"집단이 {n}개뿐이라 평가할 수 없습니다.")
        return
    med = float(d[C_MEAN].median())
    y = (d[C_MEAN].to_numpy(dtype=float) >= med).astype(int)
    if y.sum() < 3 or (n - y.sum()) < 3:
        st.warning("'높음' 또는 '낮음' 집단이 3개 미만이라 평가할 수 없습니다.")
        return
    X = feats.loc[d.index].to_numpy(dtype=float)

    if mode == "pooled":
        from sklearn.model_selection import GroupKFold

        groups = d[C_DIM].astype(str).to_numpy()
        k = min(5, len(np.unique(groups)))
        if k < 3:
            st.warning("구분이 3개 미만이라 구분 단위 교차검증을 할 수 없습니다.")
            return
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    else:
        from sklearn.model_selection import LeaveOneOut

        splits = list(LeaveOneOut().split(X))

    st.markdown(
        f"**'높음'의 기준:** 평균 이용 시간 **{med:.1f}분**(중앙값) 이상. 높음 {int(y.sum())}개, 낮음 {int(n - y.sum())}개 집단입니다. "
        f"{eval_note}"
    )

    factories = ml_model_factories()
    results = {}
    errors = []
    for name, fac in factories.items():
        try:
            pred = cv_predict(fac, X, y, splits)
            results[name] = dict(pred=pred, **cls_metrics(y, pred))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    for msg in errors:
        st.warning(f"모델 평가 중 문제가 생겨 건너뛰었습니다 → {msg}")
    models = [m for m in results if m != ML_BASELINE]
    if not models:
        st.error("평가할 수 있는 모델이 없습니다.")
        return
    metric_names = ["정확도", "정밀도", "재현율", "F1 점수"]

    # ---------------- (2) 성능 표
    st.subheader("2) 모델별 성능 지표 (표)")
    st.markdown(
        "- **정확도**: 전체 집단 중 '높음/낮음'을 맞힌 비율\n"
        "- **정밀도**: 모델이 '높음'이라고 한 집단 중 실제로 '높음'인 비율 (틀린 경보가 적을수록 높음)\n"
        "- **재현율**: 실제 '높음' 집단 중 모델이 찾아낸 비율 (놓친 것이 적을수록 높음)\n"
        "- **F1 점수**: 정밀도와 재현율을 함께 보는 점수 (둘이 모두 높아야 높음)"
    )
    table = pd.DataFrame(
        [{"모델": m, **{k: round(results[m][k], 3) for k in metric_names}} for m in results]
    )
    show_table(table)

    best = max(models, key=lambda m: (results[m]["F1 점수"], results[m]["정확도"]))
    b, base = results[best], results.get(ML_BASELINE)
    lines = [
        f"F1 점수가 가장 높은 모델은 **{best}** 입니다. 정확도 **{b['정확도'] * 100:.1f}%**, 정밀도 **{b['정밀도']:.2f}**, "
        f"재현율 **{b['재현율']:.2f}**, F1 **{b['F1 점수']:.2f}** 입니다."
    ]
    if base is not None:
        gap = b["정확도"] - base["정확도"]
        if gap > 0.1:
            lines.append(
                f"항상 다수 클래스로만 찍는 기준선의 정확도는 {base['정확도'] * 100:.1f}%입니다. {best}가 약 {gap * 100:.0f}%p 더 높아 "
                "특성이 평균 이용 시간의 높고 낮음을 가려내는 데 어느 정도 도움이 되는 것으로 보입니다."
            )
        else:
            lines.append(
                f"항상 다수 클래스로만 찍는 기준선의 정확도는 {base['정확도'] * 100:.1f}%로, {best}와 큰 차이가 없습니다. "
                "이 특성들로는 모델이 기준선보다 확실히 낫다고 말하기 어렵습니다."
            )
    if b["정밀도"] > b["재현율"] + 0.1:
        lines.append("정밀도가 재현율보다 높아, '높음'이라고 할 때는 믿을 만하지만 실제 '높음' 집단을 일부 놓치는 편입니다.")
    elif b["재현율"] > b["정밀도"] + 0.1:
        lines.append("재현율이 정밀도보다 높아, 실제 '높음' 집단은 잘 찾지만 '낮음' 집단을 '높음'으로 잘못 부르는 경우가 섞여 있습니다.")
    else:
        lines.append("정밀도와 재현율이 비슷해서, 한쪽으로 치우치지 않고 균형 있게 분류하고 있습니다.")
    if mode == "pooled":
        lines.append("탐색용 평가이므로 이 숫자를 모델의 진짜 성능으로 받아들이면 안 됩니다.")
    else:
        lines.append(f"집단이 {n}개뿐이라 집단 한두 개만 달라져도 점수가 크게 바뀔 수 있습니다.")
    st.subheader("핵심 해석")
    st.info("\n\n".join(lines))

    # ---------------- (3) 성능 막대그래프
    st.subheader("3) 모델별 성능 지표 (그래프)")
    palette = ["#1f4e9c", "#e07b39", "#4c9f70", "#9aa0a6"]
    figm = go.Figure()
    for i, m in enumerate(results):
        figm.add_trace(go.Bar(
            name=m, x=metric_names, y=[results[m][k] for k in metric_names],
            marker_color=palette[i % len(palette)],
            text=[f"{results[m][k]:.2f}" for k in metric_names], textposition="outside",
        ))
    figm.update_layout(
        barmode="group", title="모델별 정확도·정밀도·재현율·F1 점수", xaxis_title="평가 지표",
        yaxis_title="점수(0~1)", yaxis_range=[0, 1.15], legend=dict(orientation="h", y=-0.25),
    )
    show_chart(figm, key="ml_metrics")
    st.caption("막대가 높을수록 성능이 좋습니다. 회색 막대(기준선)보다 높아야 모델이 의미 있게 학습한 것입니다.")
    ordered = sorted(models, key=lambda m: -results[m]["F1 점수"])
    gtxt = f"F1 점수 기준 순위는 {' > '.join(ordered)} 입니다. "
    if len(ordered) >= 2 and results[ordered[0]]["F1 점수"] - results[ordered[-1]]["F1 점수"] < 0.05:
        gtxt += "모델 사이의 차이가 작아서, 어떤 모델을 쓰든 결과가 비슷합니다. 이럴 때는 설명하기 쉬운 단순한 모델이 더 좋은 선택입니다."
    else:
        gtxt += "모델에 따라 성능 차이가 있어, 단순한 모델이 더 좋을 때도 복잡한 모델이 더 좋을 때도 있음을 보여 줍니다."
    st.info(gtxt)

    # ---------------- (4) 혼동행렬
    st.subheader("4) 어디에서 틀렸는지 보기 (혼동행렬)")
    pick = st.selectbox("살펴볼 모델", models, index=models.index(best), key="ml_cm_model")
    from sklearn.metrics import confusion_matrix

    pred = results[pick]["pred"]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    figcm = go.Figure(go.Heatmap(
        z=cm, x=[f"예측: {c}" for c in ML_CLASS_NAMES], y=[f"실제: {c}" for c in ML_CLASS_NAMES],
        text=cm.astype(str), texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate="%{y}, %{x}: %{z}개<extra></extra>",
    ))
    figcm.update_layout(title=f"{pick}의 혼동행렬", yaxis_autorange="reversed")
    show_chart(figcm, key="ml_cm")
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    st.info(
        f"실제로 '높음'인 집단 {tp + fn}개 중 **{tp}개**를 '높음'으로 맞혔고 **{fn}개**는 '낮음'으로 잘못 판단했습니다. "
        f"실제로 '낮음'인 집단 {tn + fp}개 중 **{tn}개**를 맞혔고 **{fp}개**는 '높음'으로 잘못 판단했습니다. "
        "대각선(왼쪽 위, 오른쪽 아래) 숫자가 클수록 잘 맞힌 것입니다."
    )
    wrong = d.assign(_pred=pred, _y=y)
    wrong = wrong[wrong["_pred"] != wrong["_y"]]
    if len(wrong):
        label = (wrong[C_DIM] + " · " + wrong[C_GRP]) if mode == "pooled" else wrong[C_GRP]
        with st.expander(f"잘못 분류된 집단 {len(wrong)}개 보기"):
            show_table(pd.DataFrame({
                "집단": label.to_numpy(),
                "평균 이용시간(분)": wrong[C_MEAN].round(1).to_numpy(),
                "실제": [ML_CLASS_NAMES[v] for v in wrong["_y"]],
                "예측": [ML_CLASS_NAMES[v] for v in wrong["_pred"]],
            }))

    # ---------------- (5) 특성 중요도
    st.subheader("5) 모델이 어떤 특성을 중요하게 봤을까?")
    try:
        full = factories[pick]().fit(X, y)
        last = full[-1] if hasattr(full, "steps") else full
        if hasattr(last, "coef_"):
            imp = pd.Series(np.abs(last.coef_[0]), index=feats.columns)
            signed = pd.Series(last.coef_[0], index=feats.columns)
            title, xlab = "특성별 영향력 (표준화 계수의 크기)", "영향력(절댓값)"
        elif hasattr(last, "feature_importances_"):
            imp = pd.Series(last.feature_importances_, index=feats.columns)
            signed = None
            title, xlab = "특성별 중요도", "중요도(합계 1)"
        else:
            imp = None
        if imp is not None and imp.sum() > 0:
            imp = imp.sort_values()
            figi = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h", marker_color="#1f4e9c",
                text=[f"{v:.2f}" for v in imp.values], textposition="outside",
            ))
            figi.update_layout(
                title=f"{pick}: {title}", xaxis_title=xlab, yaxis_title="",
                showlegend=False, height=max(320, 45 * len(imp) + 140),
            )
            show_chart(figi, key="ml_imp")
            top = imp.index[-1]
            txt = f"{pick}이(가) 높고 낮음을 가를 때 가장 크게 참고한 특성은 **{top}** 입니다. "
            if signed is not None:
                txt += f"이 특성 값이 클수록 '{'높음' if signed[top] > 0 else '낮음'}'으로 분류하는 쪽으로 작용했습니다. "
            txt += (
                "이 그림은 모델 전체 자료로 다시 학습해 그린 설명용 그래프이며, 특성들끼리 서로 비슷하게 움직이는 경우가 많아 "
                "하나하나의 영향을 정확히 나눠 말하기는 어렵습니다."
            )
            st.info(txt)
        else:
            st.caption("이 모델은 특성 중요도를 제공하지 않습니다.")
    except Exception as e:  # noqa: BLE001
        st.caption(f"특성 중요도를 계산하지 못했습니다. ({e})")


# ------------------------------------------------------------------
# 마지막 탭: 다운로드
# ------------------------------------------------------------------
def render_download(ctx):
    st.header("전처리된 데이터 내려받기")
    st.markdown(
        "공백 정리, 숫자형 변환, 중복·결측치 처리를 마친 데이터를 CSV 파일로 저장할 수 있습니다. "
        "엑셀에서도 한글이 깨지지 않도록 UTF-8(BOM) 형식으로 저장합니다."
    )
    cleaned = ctx["cleaned"]
    mask = ctx["total_mask"]
    has_total = bool(mask.any())
    exclude = False
    if has_total:
        exclude = st.checkbox("전체 집계 행(총인원)을 제외하고 저장", value=False, key="dl_excl")
    out = cleaned[~mask] if exclude else cleaned
    st.caption(f"저장될 데이터: {out.shape[0]:,}행 × {out.shape[1]:,}열")
    show_table(out.head(20))
    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "전처리된 CSV 다운로드", data=csv_bytes, file_name=DOWNLOAD_NAME, mime="text/csv",
    )


# ------------------------------------------------------------------
# 시작 화면 · 사이드바
# ------------------------------------------------------------------
def idx_of(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default


def sidebar_settings(clean0, det):
    cols = list(clean0.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(clean0[c])]
    with st.sidebar.expander("분석에 사용할 열 (자동 인식 결과)", expanded=False):
        st.caption("자동으로 찾은 열입니다. 잘못 인식되었다면 직접 바꿀 수 있습니다.")
        g1 = st.selectbox("집단 구분(대분류) 열", cols, index=idx_of(cols, det["g1"]), key="sel_g1")
        g2_opts = [NONE_OPT] + cols
        g2 = st.selectbox("집단 이름(세부) 열", g2_opts, index=idx_of(g2_opts, det["g2"]), key="sel_g2")
        mean_default = det["mean"] if det["mean"] else (num_cols[0] if num_cols else cols[0])
        mean_col = st.selectbox("평균 이용시간 열", cols, index=idx_of(cols, mean_default), key="sel_mean")
        cnt_opts = [NONE_OPT] + cols
        cnt = st.selectbox("사례수 열", cnt_opts, index=idx_of(cnt_opts, det["cnt"]), key="sel_cnt")
        rate = st.selectbox("이용률 열", cnt_opts, index=idx_of(cnt_opts, det["rate"]), key="sel_rate")
        bins = st.multiselect("이용 시간 구간별 비율 열", cols, default=det["bins"], key="sel_bins")
    return dict(
        g1=g1,
        g2=None if g2 == NONE_OPT else g2,
        mean=mean_col,
        cnt=None if cnt == NONE_OPT else cnt,
        rate=None if rate == NONE_OPT else rate,
        bins=bins,
    )


def validate_selection(clean0, sel):
    problems = []
    if sel["g2"] and sel["g2"] == sel["g1"]:
        problems.append("집단 구분 열과 집단 이름 열이 같습니다.")
    if not pd.api.types.is_numeric_dtype(clean0[sel["mean"]]):
        problems.append(f"'{sel['mean']}' 열이 숫자형이 아니라 평균 이용 시간으로 쓸 수 없습니다.")
    if len(sel["bins"]) < 3:
        problems.append("이용 시간 구간별 비율 열이 3개 이상 필요합니다. 사이드바에서 열을 지정해 주세요.")
    for b in sel["bins"]:
        if not pd.api.types.is_numeric_dtype(clean0[b]):
            problems.append(f"'{b}' 열이 숫자형이 아닙니다.")
    for k in ("cnt", "rate"):
        c = sel[k]
        if c and not pd.api.types.is_numeric_dtype(clean0[c]):
            problems.append(f"'{c}' 열이 숫자형이 아닙니다. (사례수/이용률 열을 '(없음)'으로 바꿔 보세요.)")
    return problems


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
def main():
    st.title("스마트폰 이용행태 조사 2025 데이터 분석")
    st.caption("스마트폰 이용행태 조사 자료를 자동으로 정리하고, 두 가지 탐구 질문에 대한 그래프와 해석을 보여 줍니다.")

    uploaded = st.sidebar.file_uploader("다른 CSV로 분석하려면 업로드 (선택)", type=["csv"])
    default_path = find_default_csv()

    if uploaded is not None:
        data, source = uploaded.getvalue(), f"업로드한 파일: {uploaded.name}"
    elif default_path is not None:
        try:
            data, source = default_path.read_bytes(), f"기본 데이터: {default_path.name}"
        except Exception as e:  # noqa: BLE001
            st.error(f"기본 데이터 파일을 읽지 못했습니다. ({e})")
            return
    else:
        st.error(
            f"기본 데이터 파일(`{DEFAULT_CSV}`)을 찾지 못했습니다. "
            "이 파일을 main.py와 같은 폴더에 넣거나, 왼쪽 사이드바에서 CSV를 업로드해 주세요."
        )
        return
    st.sidebar.caption(source)

    try:
        raw, enc = read_csv_bytes(data)
    except pd.errors.EmptyDataError:
        st.error("파일이 비어 있습니다. 내용이 있는 CSV 파일을 확인해 주세요.")
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"파일을 읽는 중 문제가 생겼습니다. CSV 형식이 맞는지 확인해 주세요. ({e})")
        return

    if raw.shape[0] == 0 or raw.shape[1] < 3:
        st.error("행이나 열이 너무 적어 분석할 수 없습니다. 파일 내용을 확인해 주세요.")
        return

    clean0, clean_info = basic_clean(raw)
    det = detect_columns(clean0)
    sel = sidebar_settings(clean0, det)
    problems = validate_selection(clean0, sel)

    tab_names = [
        "데이터 확인·전처리",
        "탐구 질문 1: 연령대별 이용 시간",
        "탐구 질문 2: 시간대 비율과 평균",
        "탐구 질문 3: 콘텐츠 이용 빈도",
        "탐구 질문 4: 집단 특성 비교",
        "탐구 질문 5: 예측 모델 검토",
        "머신러닝 모델 평가",
        "전처리 데이터 다운로드",
    ]
    tabs = st.tabs(tab_names)

    if problems:
        with tabs[0]:
            st.header("데이터 확인과 전처리")
            c1, c2, c3 = st.columns(3)
            c1.metric("행 개수", f"{raw.shape[0]:,}")
            c2.metric("열 개수", f"{raw.shape[1]:,}")
            c3.metric("인코딩", enc)
            show_table(raw.head(10))
            st.error("분석에 필요한 열을 정하는 중 문제가 발견되었습니다.")
            for p in problems:
                st.markdown(f"- {p}")
            st.markdown("**현재 인식된 열 이름:** " + ", ".join(clean0.columns))
        for t in tabs[1:]:
            with t:
                st.warning("먼저 '데이터 확인·전처리' 탭의 안내에 따라 열 설정을 바로잡아 주세요.")
        return

    g1, g2, mean_col = sel["g1"], sel["g2"], sel["mean"]
    bins_raw = sorted(sel["bins"], key=lambda c: (parse_hours(c)[0] if parse_hours(c) else 1e9))
    key_cols = [g1] + ([g2] if g2 else []) + [mean_col] + bins_raw

    cleaned, proc = handle_missing_dup(clean0, key_cols)
    if cleaned.empty:
        st.error("전처리 후 남은 데이터가 없습니다. 열 설정을 확인해 주세요.")
        return

    total_mask = is_total_row(cleaned, g1, g2)

    labels = []
    for c in bins_raw:
        lab = short_label(c)
        if lab in labels:
            lab = f"{lab}_{len(labels) + 1}"
        labels.append(lab)

    std_all = build_std(cleaned, g1, g2, mean_col, sel["cnt"], sel["rate"], bins_raw, labels, total_mask)
    std = std_all[~std_all[C_TOTAL]].copy()
    if std.empty:
        st.error("집계 행을 제외하고 나니 비교할 집단이 남지 않았습니다.")
        return

    bin_mid = {}
    for c, lab in zip(bins_raw, labels):
        ph = parse_hours(c)
        if ph:
            bin_mid[lab] = (ph[0] + ph[1]) / 2 * 60

    ctx = dict(
        raw=raw, clean0=clean0, enc=enc, clean_info=clean_info, proc=proc,
        cleaned=cleaned, total_mask=total_mask, std_all=std_all, std=std,
        bins=labels, bin_mid=bin_mid,
    )
    ctx["freq_all"] = detect_freq(cleaned)
    ctx["freq3"] = pick_q3_contents(ctx["freq_all"])
    try:
        ctx["features"] = build_features(ctx)
    except Exception:  # noqa: BLE001
        ctx["features"] = pd.DataFrame(index=std.index)

    with tabs[0]:
        render_overview(ctx)
    with tabs[1]:
        try:
            render_q1(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[2]:
        try:
            render_q2(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[3]:
        try:
            render_q3(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[4]:
        try:
            render_q4(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[5]:
        try:
            render_q5(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 중 문제가 발생했습니다: {e}")
    with tabs[6]:
        try:
            render_ml(ctx)
        except Exception as e:  # noqa: BLE001
            st.error(f"모델 평가 중 문제가 발생했습니다: {e}")
    with tabs[7]:
        render_download(ctx)


main()
