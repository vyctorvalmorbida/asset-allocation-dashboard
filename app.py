# app.py — Asset Allocation Backtest (Web, visual institucional)
# deps: streamlit pandas numpy yfinance plotly xlsxwriter openpyxl

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import io

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG INICIAL
# =========================
DEFAULT_START_DATE = "2004-01-01"
DEFAULT_REBALANCE  = "Q"    # M/Q/A
DEFAULT_RF_MODE    = "IRX"  # IRX/BIL/ZERO

TICKER_FIXED_INCOME = "AGG"       # proxy renda fixa US
TICKER_EQUITY       = "SPY"       # proxy equity US
TICKERS_ALTS        = ["GLD", "VNQ", "DBC"]  # ouro, REIT, commodities

CRISIS_WINDOWS = {
    "GFC_2007-2009": ("2007-10-09", "2009-03-09"),
    "Euro_2011":     ("2011-04-29", "2011-10-03"),
    "Covid_2020":    ("2020-02-19", "2020-03-23"),
    "Bear_Inf_2022": ("2022-01-03", "2022-10-12"),
}

ALLOCATIONS = {
    "Defensiva":       {"Fixed Income": 1.00, "Equity": 0.00, "Alternatives": 0.00},
    "Conservadora_1":  {"Fixed Income": 0.90, "Equity": 0.08, "Alternatives": 0.02},
    "Conservadora_2":  {"Fixed Income": 0.80, "Equity": 0.16, "Alternatives": 0.04},
    "Conservadora_3":  {"Fixed Income": 0.70, "Equity": 0.24, "Alternatives": 0.06},
    "Moderada_1":      {"Fixed Income": 0.60, "Equity": 0.32, "Alternatives": 0.08},
    "Moderada_2":      {"Fixed Income": 0.50, "Equity": 0.40, "Alternatives": 0.10},
    "Moderada_3":      {"Fixed Income": 0.40, "Equity": 0.48, "Alternatives": 0.12},
    "Arrojada_1":      {"Fixed Income": 0.30, "Equity": 0.56, "Alternatives": 0.14},
    "Arrojada_2":      {"Fixed Income": 0.20, "Equity": 0.64, "Alternatives": 0.16},
    "Arrojada_3":      {"Fixed Income": 0.10, "Equity": 0.72, "Alternatives": 0.18},
    "Crescimento":     {"Fixed Income": 0.00, "Equity": 0.80, "Alternatives": 0.20},
}

# =========================
# ESTILO
# =========================
st.set_page_config(page_title="Asset Allocation Backtest", page_icon="📈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
h1,h2,h3 { font-weight: 700; letter-spacing: .2px; }
.block-container { padding-top: 1rem; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
table { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
COLORWAY = px.colors.qualitative.D3  # paleta consistente

# =========================
# FUNÇÕES FINANCE
# =========================
@st.cache_data(show_spinner=False)
def download_adjclose(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.dropna(how="all")

def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    mp = prices.resample("ME").last()
    return mp.pct_change().dropna(how="all")

@st.cache_data(show_spinner=True)
def build_class_returns(start: str, end: Optional[str]) -> pd.DataFrame:
    fi = to_monthly_returns(download_adjclose([TICKER_FIXED_INCOME], start, end)).iloc[:,0].rename("Fixed Income")
    eq = to_monthly_returns(download_adjclose([TICKER_EQUITY], start, end)).iloc[:,0].rename("Equity")
    al_prices = download_adjclose(TICKERS_ALTS, start, end)
    al_rets = to_monthly_returns(al_prices)
    alts = al_rets.apply(lambda row: row.dropna().mean() if row.dropna().size>0 else np.nan, axis=1)
    alts.name = "Alternatives"
    return pd.concat([fi,eq,alts], axis=1).dropna()

def nav_from_returns(r: pd.Series, base=100.0) -> pd.Series:
    return (1 + r).cumprod() * base

def drawdown_from_returns(r: pd.Series) -> pd.Series:
    nav = (1 + r).cumprod()
    peak = nav.cummax()
    return nav/peak - 1.0

def annualize_return(m: pd.Series) -> float:
    r = (1+m).prod()
    n = len(m)/12.0
    return r**(1/n) - 1 if n>0 else np.nan

def annualize_vol(m: pd.Series) -> float:
    return float(m.std(ddof=0)) * np.sqrt(12)

def max_dd_value(m: pd.Series) -> float:
    return float(drawdown_from_returns(m).min())

@st.cache_data(show_spinner=False)
def get_risk_free_monthly(start: str, end: Optional[str], mode: str) -> pd.Series:
    """
    Retorna série mensal RF (nome 'RF').
    - ZERO: série vazia (Sharpe contra 0)
    - BIL : retornos mensais do ETF BIL
    - IRX : ^IRX (3m T-Bill, anual em %) -> converte p/ retorno mensal
    """
    mode = (mode or "IRX").upper()

    if mode == "ZERO":
        return pd.Series(dtype=float, name="RF")

    if mode == "BIL":
        try:
            bil_prices = download_adjclose(["BIL"], start, end)
            bil_rets = to_monthly_returns(bil_prices)
            s = bil_rets.iloc[:, 0] if isinstance(bil_rets, pd.DataFrame) else bil_rets
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.name = "RF"
            return s
        except Exception:
            return pd.Series(dtype=float, name="RF")

    # === IRX (default) ===
    try:
        data = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)
        # pega a coluna 'Adj Close' se existir; senão tenta 'Close' ou a 1ª coluna
        irx = None
        for col in ["Adj Close", "Close"]:
            if col in data.columns:
                irx = data[col]
                break
        if irx is None:
            if isinstance(data, pd.Series):
                irx = data
            elif isinstance(data, pd.DataFrame) and not data.empty:
                irx = data.iloc[:, 0]

        if irx is None or irx.dropna().empty:
            return pd.Series(dtype=float, name="RF")

        # garante Series
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]
        irx = pd.to_numeric(irx, errors="coerce").dropna()

        # anual (%) -> mensal (retorno)
        irx_m = irx.resample("ME").last() / 100.0
        rf_m = (1.0 + irx_m) ** (1.0 / 12.0) - 1.0

        # garante Series 1-d
        if isinstance(rf_m, pd.DataFrame):
            rf_m = rf_m.iloc[:, 0]
        rf_m = pd.Series(rf_m.values, index=rf_m.index, name="RF")
        rf_m = pd.to_numeric(rf_m, errors="coerce").dropna()
        rf_m.name = "RF"
        return rf_m
    except Exception:
        return pd.Series(dtype=float, name="RF")

def sharpe_ratio(m: pd.Series, rf_m: Optional[pd.Series]) -> float:
    r = pd.to_numeric(m, errors="coerce").dropna()
    if rf_m is None or rf_m.empty: ex = r
    else:
        rf = pd.to_numeric(rf_m, errors="coerce").reindex(r.index).ffill().fillna(0.0)
        ex = r - rf
    if ex.empty: return np.nan
    mu, sd = float(ex.mean()), float(ex.std(ddof=0))
    return (mu/sd)*np.sqrt(12) if sd>0 else np.nan

def sortino_ratio(m: pd.Series, rf_m: Optional[pd.Series]) -> float:
    r = pd.to_numeric(m, errors="coerce").dropna()
    if rf_m is None or rf_m.empty: ex = r
    else:
        rf = pd.to_numeric(rf_m, errors="coerce").reindex(r.index).ffill().fillna(0.0)
        ex = r - rf
    downside = ex[ex<0]
    if downside.empty: return np.nan
    mu, dd = float(ex.mean()), float(downside.std(ddof=0))
    return (mu/dd)*np.sqrt(12) if dd>0 else np.nan

def calmar_ratio(cagr: float, mdd: float) -> float:
    d = abs(float(mdd))
    return float(cagr)/d if d>0 else np.nan

def skewness(m: pd.Series) -> float:
    return float(pd.to_numeric(m, errors="coerce").dropna().skew())

def rebalanced_portfolio_returns(class_rets: pd.DataFrame, weights: Dict[str,float], freq: str="Q") -> pd.Series:
    r = class_rets.dropna().copy()
    if freq == "M": marks = r.index
    elif freq == "Q": marks = r.index[r.index.month.isin([3,6,9,12])]
    elif freq == "A": marks = r.index[r.index.month==12]
    else: raise ValueError("freq deve ser 'M','Q' ou 'A'")
    w = pd.Series(weights).reindex(r.columns).fillna(0.0); w = w/w.sum()
    cur_w = w.copy(); out=[]
    for dt,row in r.iterrows():
        if dt in marks: cur_w = w.copy()
        out.append(float((cur_w*row).sum()))
    return pd.Series(out, index=r.index)

def calc_metrics(m: pd.Series, rf_m: Optional[pd.Series]) -> Dict[str,float]:
    cagr = annualize_return(m)
    vol  = annualize_vol(m)
    mdd  = max_dd_value(m)
    sr   = sharpe_ratio(m, rf_m)
    sor  = sortino_ratio(m, rf_m)
    cal  = calmar_ratio(cagr, mdd)
    sk   = skewness(m)
    return {"CAGR":cagr, "Vol":vol, "Sharpe":sr, "Sortino":sor, "MaxDD":mdd, "Calmar":cal, "Skew":sk}

# =========================
# PIPELINE (cacheado)
# =========================
@st.cache_data(show_spinner=True)
def run_backtest(start_date: str, rebalance: str, rf_mode: str):
    classes = build_class_returns(start_date, None)
    if classes.empty:
        return pd.DataFrame(), {}, {}, {}, pd.Series(dtype=float), {}
    last = classes.index.max()

    rf = get_risk_free_monthly(classes.index.min().strftime("%Y-%m-%d"),
                               classes.index.max().strftime("%Y-%m-%d"),
                               rf_mode)

    port_rets = {name: rebalanced_portfolio_returns(classes, w, freq=rebalance).reindex(classes.index).dropna()
                 for name,w in ALLOCATIONS.items()}

    windows = {
        "Full Sample": (classes.index.min(), last),
        "Last 20Y":    (last - pd.DateOffset(years=20), last),
        "Last 10Y":    (last - pd.DateOffset(years=10), last),
        "Last 5Y":     (last - pd.DateOffset(years=5),  last),
    }

    results = {}
    for wname,(ws,we) in windows.items():
        rows={}
        for name,r in port_rets.items():
            sub = r.loc[(r.index>=ws)&(r.index<=we)].dropna()
            if len(sub)>=24: rows[name]=calc_metrics(sub, rf)
        results[wname]=pd.DataFrame(rows).T.sort_index()

    # Crises (corrigido MaxDD p/ janelas de 1 mês)
    # Crises (robusto a janelas sem dados)
    crisis_tables = {}
    expected_cols = ["Cumulative", "MaxDD", "WorstMonth", "Obs"]

    for cname, (cs, ce) in CRISIS_WINDOWS.items():
        cs, ce = pd.to_datetime(cs), pd.to_datetime(ce)
        rows = {}
        for name, r in port_rets.items():
            sub = r.loc[(r.index >= cs) & (r.index <= ce)].dropna()
            if len(sub) == 0:
                continue

            cum = float((1 + sub).prod() - 1)

            # janela de 1 mês: usa o próprio retorno como "drawdown" (se negativo) para evitar NaN/KeyError
            if len(sub) == 1:
                wm = float(sub.iloc[0])
                mdd = wm if wm < 0 else 0.0
            else:
                mdd = max_dd_value(sub)

            rows[name] = {
                "Cumulative": cum,
                "MaxDD": float(mdd),
                "WorstMonth": float(sub.min()),
                "Obs": int(len(sub)),
            }

        if rows:
            crisis_tables[cname] = pd.DataFrame(rows).T.sort_values("Cumulative")
        else:
            # mantém estrutura de colunas para não quebrar o front-end
            crisis_tables[cname] = pd.DataFrame(columns=expected_cols)

# =========================
# GRÁFICOS
# =========================
def fig_nav_overlay(port_rets: Dict[str,pd.Series], start=None, end=None, title="NAV (base 100)"):
    fig = go.Figure()
    for i,(name,r) in enumerate(sorted(port_rets.items(), key=lambda x: x[0])):
        s = r if start is None else r.loc[(r.index>=start)&(r.index<=end)]
        if s.empty: continue
        nav = nav_from_returns(s)
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, mode="lines",
                                 name=name, line=dict(width=2)))
    fig.update_layout(template=PLOTLY_TEMPLATE, colorway=COLORWAY,
                      title=title, legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=30,r=20,t=60,b=60), height=520)
    fig.update_yaxes(title_text="NAV (base 100)")
    return fig

def fig_drawdown_overlay(port_rets: Dict[str,pd.Series], start=None, end=None, title="Drawdown (%)"):
    fig = go.Figure()
    for name,r in sorted(port_rets.items(), key=lambda x: x[0]):
        s = r if start is None else r.loc[(r.index>=start)&(r.index<=end)]
        if s.empty: continue
        dd = drawdown_from_returns(s)*100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name, line=dict(width=2)))
    fig.update_layout(template=PLOTLY_TEMPLATE, colorway=COLORWAY,
                      title=title, legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=30,r=20,t=60,b=60), height=460)
    fig.update_yaxes(title_text="%", rangemode="tozero")
    return fig

def fig_heatmap(corr: pd.DataFrame, title: str):
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f",
                    aspect="auto", template=PLOTLY_TEMPLATE, title=title)
    fig.update_layout(margin=dict(l=30,r=20,t=60,b=40), height=520)
    return fig

def fig_weights_bars(allocs_df: pd.DataFrame):
    df = allocs_df.reset_index().melt(id_vars="index", var_name="Classe", value_name="Peso")
    df = df.rename(columns={"index":"Portfolio"})
    fig = px.bar(df, x="Portfolio", y="Peso", color="Classe", barmode="stack",
                 text=df["Peso"].map(lambda x: f"{x*100:.0f}%"),
                 template=PLOTLY_TEMPLATE, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(title="Composição por Carteira (barras empilhadas)",
                      yaxis_tickformat=".0%", height=520, margin=dict(l=30,r=20,t=60,b=40))
    return fig

def fig_weights_pie(weights: Dict[str,float], name: str):
    labels = list(weights.keys()); vals = list(weights.values())
    fig = px.pie(values=vals, names=labels, hole=0.45, template=PLOTLY_TEMPLATE,
                 title=f"Composição — {name}")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
    return fig

# =========================
# EXPORTS
# =========================
def build_excel(results, crises, classes, port_rets, windows) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for wname, df in results.items():
            if df.empty: continue
            df2 = df.copy()
            ws,we = windows[wname]
            obs_map = {p: len(r.loc[(r.index>=ws)&(r.index<=we)]) for p,r in port_rets.items()}
            df2["Obs"] = pd.Series(obs_map)
            if wname.startswith("Last "):
                years = int(wname.split()[1].replace("Y",""))
                df2["Cumulative"] = (1 + df2["CAGR"])**years - 1
            df2.to_excel(writer, sheet_name=f"Metrics_{wname.replace(' ','_')}")
        if crises:
            concat=[]
            for cname,cdf in crises.items():
                t = cdf.copy(); t.insert(0,"Crisis",cname); concat.append(t)
            pd.concat(concat).to_excel(writer, sheet_name="Crises")
        pd.DataFrame(port_rets).corr(min_periods=12).to_excel(writer, sheet_name="Corr_Portfolios")
        classes.corr(min_periods=12).to_excel(writer, sheet_name="Corr_Classes")
        pd.DataFrame(ALLOCATIONS).T.to_excel(writer, sheet_name="Inputs_Allocations")
        pd.DataFrame({
            "Fixed Income":[TICKER_FIXED_INCOME],
            "Equity":[TICKER_EQUITY],
            "Alternatives":[", ".join(TICKERS_ALTS)],
            "RF mode":[DEFAULT_RF_MODE],
        }).to_excel(writer, sheet_name="Inputs_Tickers", index=False)
    return buf.getvalue()

# =========================
# UI — SIDEBAR
# =========================
st.sidebar.header("⚙️ Parâmetros")
start_date = st.sidebar.date_input("Data inicial", value=pd.to_datetime(DEFAULT_START_DATE))
rebalance = st.sidebar.selectbox("Rebalanceamento", ["M","Q","A"], index=["M","Q","A"].index(DEFAULT_REBALANCE))
rf_mode   = st.sidebar.selectbox("Risco livre (Sharpe/Sortino)", ["IRX","BIL","ZERO"], index=["IRX","BIL","ZERO"].index(DEFAULT_RF_MODE))
st.sidebar.markdown("---")
st.sidebar.caption("Proxies: FI=AGG | Equity=SPY | Alternativos= média de GLD/VNQ/DBC.")

# =========================
# EXECUÇÃO
# =========================
if classes.empty:
    return pd.DataFrame(), {}, {}, {}, pd.Series(dtype=float), {}

st.title("Asset Allocation — Dashboard")
st.caption("Simulador histórico de carteiras por classe de ativos (wealth management).")

# =========================
# SEÇÃO: COMO AS CARTEIRAS SÃO MONTADAS
# =========================
st.markdown("## Estratégia & Composição das Carteiras")
allocs_df = pd.DataFrame(ALLOCATIONS).T
show_alloc = allocs_df.copy().map(lambda x: f"{x*100:.0f}%")
c1,c2 = st.columns([2,1])
with c1:
    st.dataframe(show_alloc, use_container_width=True)
with c2:
    pick = st.selectbox("Carteira para visualizar a composição:", list(ALLOCATIONS.keys()), index=list(ALLOCATIONS.keys()).index("Moderada_2"))
    st.plotly_chart(fig_weights_pie(ALLOCATIONS[pick], pick), use_container_width=True)
st.plotly_chart(fig_weights_bars(allocs_df), use_container_width=True)
st.info("Rebalanceamento periódico (M/Q/A) para voltar aos pesos-alvo. Proxies: "
        f"Fixed Income = {TICKER_FIXED_INCOME}, Equity = {TICKER_EQUITY}, "
        f"Alternativos = média equiponderada de {', '.join(TICKERS_ALTS)}.")

# =========================
# SEÇÃO: EXECUTIVO (NAV/DRAWDOWN/CRISES)
# =========================
st.markdown("## Visão Executiva (Comercial)")
ws_full = windows["Full Sample"][0]; we_full = windows["Full Sample"][1]
ws_10 = windows["Last 10Y"][0]; ws_5 = windows["Last 5Y"][0]

st.plotly_chart(fig_nav_overlay(port_rets, title="NAV — Amostra Completa"), use_container_width=True)
c3,c4 = st.columns(2)
with c3:
    st.plotly_chart(fig_nav_overlay(port_rets, start=ws_10, end=we_full, title="NAV — Últimos 10 anos"), use_container_width=True)
with c4:
    st.plotly_chart(fig_nav_overlay(port_rets, start=ws_5, end=we_full, title="NAV — Últimos 5 anos"), use_container_width=True)

st.plotly_chart(fig_drawdown_overlay(port_rets, title="Drawdown — Amostra Completa"), use_container_width=True)

st.subheader("Crises históricas (impacto por carteira)")
for cname, cdf in crises.items():
    st.markdown(f"**{cname}**")
    if cdf.empty:
        st.info("Sem dados para esta crise com a data inicial selecionada.")
        continue
    show = cdf.copy()
    for c in ["Cumulative", "MaxDD", "WorstMonth"]:
        show[c] = (show[c] * 100).map(lambda x: f"{x:.2f}%")
    st.dataframe(show, use_container_width=True)

# =========================
# SEÇÃO: TÉCNICA (KPIs, correlação, rolling, histogramas)
# =========================
st.markdown("## Visão Técnica (Asset)")
# === KPIs por janela (organizado) ===
st.markdown("### KPIs por janela")

sel_port = st.multiselect(
    "Selecione carteiras para comparar KPIs:",
    list(port_rets.keys()),
    default=["Moderada_2", "Crescimento"],
)

view_mode = st.radio(
    "Modo de exibição",
    ["Tabela comparativa", "Cards"],
    index=0,
    horizontal=True,
)

def _format_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    # ordem fixa de colunas
    cols = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar"]
    out = out.reindex(columns=cols)
    # formatação
    for c in ["CAGR", "Vol", "MaxDD"]:
        if c in out:
            out[c] = (out[c] * 100).map(lambda x: f"{x:.2f}%")
    for c in ["Sharpe", "Sortino", "Calmar"]:
        if c in out:
            out[c] = out[c].map(lambda x: f"{x:.2f}")
    return out

for wname, df in results.items():
    if df.empty:
        continue

    sub = df.loc[df.index.intersection(sel_port)].copy()
    if sub.empty:
        continue

    st.markdown(f"**{wname}**")

    if view_mode == "Tabela comparativa":
        st.dataframe(
            _format_kpis(sub),
            use_container_width=True,
            height=(len(sub) + 1) * 35 + 30,
        )
    else:
        # === Cards em grade 3 por linha ===
        ports = list(sub.index)
        n = len(ports)
        cols_per_row = 3
        for i, p in enumerate(ports):
            if i % cols_per_row == 0:
                row_cols = st.columns(min(cols_per_row, n - i))
            r = sub.loc[p]
            col = row_cols[i % cols_per_row]
            with col:
                st.markdown(f"#### {p}")
                c1, c2, c3 = st.columns(3)
                c1.metric("CAGR", f"{r['CAGR']*100:.2f}%")
                c2.metric("Vol", f"{r['Vol']*100:.2f}%")
                c3.metric("MaxDD", f"{r['MaxDD']*100:.2f}%")
                c4, c5, c6 = st.columns(3)
                c4.metric("Sharpe", f"{r['Sharpe']:.2f}")
                c5.metric("Sortino", f"{r['Sortino']:.2f}")
                c6.metric("Calmar", f"{r['Calmar']:.2f}")
        st.markdown("")  # espaçamento

st.markdown("### Correlação")
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(fig_heatmap(classes.corr(min_periods=12), "Correlação — Classes de Ativos"), use_container_width=True)
with cc2:
    subset = ["Defensiva","Conservadora_3","Moderada_2","Moderada_3","Arrojada_3","Crescimento"]
    dfp = pd.DataFrame({k:v for k,v in port_rets.items() if k in subset})
    st.plotly_chart(fig_heatmap(dfp.corr(min_periods=12), "Correlação — Carteiras Selecionadas"), use_container_width=True)

st.markdown("### Rolling (36 meses)")
sel_roll = {k: port_rets[k] for k in subset if k in port_rets}
def fig_rolling(series_dict: Dict[str,pd.Series], window=36, mode="vol"):
    fig = go.Figure()
    for name, r in series_dict.items():
        if mode=="vol":
            s = r.rolling(window).std(ddof=0)*np.sqrt(12); title=f"Volatilidade (anualizada), {window}m"; ylabel="Vol"
        else:
            s = r.rolling(window).mean()/r.rolling(window).std(ddof=0)*np.sqrt(12); title=f"Sharpe (aprox.), {window}m"; ylabel="Sharpe"
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(width=2)))
    fig.update_layout(template=PLOTLY_TEMPLATE, colorway=COLORWAY, title=title,
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=30,r=20,t=60,b=60), height=460)
    fig.update_yaxes(title_text=ylabel)
    return fig
r1,r2 = st.columns(2)
with r1: st.plotly_chart(fig_rolling(sel_roll, 36, "vol"), use_container_width=True)
with r2: st.plotly_chart(fig_rolling(sel_roll, 36, "sharpe"), use_container_width=True)

st.markdown("### Distribuição de retornos mensais")
d1,d2 = st.columns(2)
with d1:
    st.plotly_chart(px.histogram(port_rets["Moderada_2"].dropna(), nbins=40, template=PLOTLY_TEMPLATE,
                                 title="Moderada_2").update_xaxes(tickformat=".1%"), use_container_width=True)
with d2:
    st.plotly_chart(px.histogram(port_rets["Crescimento"].dropna(), nbins=40, template=PLOTLY_TEMPLATE,
                                 title="Crescimento").update_xaxes(tickformat=".1%"), use_container_width=True)

# =========================
# DOWNLOADS
# =========================
st.markdown("---")
st.subheader("Downloads")
excel_bytes = build_excel(results, crises, classes, port_rets, windows)
st.download_button("⬇️ Baixar Excel consolidado", data=excel_bytes,
                   file_name="Backtest_Report.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# CSV por janela
wname_csv = st.selectbox("CSV de métricas por janela", list(results.keys()))
st.download_button("⬇️ Baixar CSV (janela selecionada)",
                   data=results[wname_csv].to_csv(index=True).encode("utf-8"),
                   file_name=f"metrics_{wname_csv.replace(' ','_')}.csv",
                   mime="text/csv")

st.caption("© Estudo educacional. Retornos passados não garantem resultados futuros.")
