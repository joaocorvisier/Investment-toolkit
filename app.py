"""
=============================================================================
INVESTMENT TOOLKIT - Dashboard Streamlit
=============================================================================
Executar:  streamlit run app.py
Deploy:    Streamlit Cloud (gratuito) via GitHub

Abas:
  1. Screener de Ações (Fórmula Mágica + Multi-Factor)
  2. Painel Macro (séries BCB + Focus, gráficos interativos)
  3. Notícias (BR + US via RSS)
=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import feedparser
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Investment Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Clean light theme */
    .stApp {
        background-color: #f8f9fb;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e5ea;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #eef0f4;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        color: #374151;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #111827 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #111827;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #374151;
    }

    /* Headings */
    h1 { color: #111827 !important; font-weight: 800 !important; }
    h2 { color: #1f2937 !important; }
    h3 { color: #374151 !important; }
    h4 { color: #374151 !important; }
    p, span, label { color: #374151; }

    /* Links in news */
    a.news-link {
        color: #2563eb !important;
        text-decoration: none;
        font-weight: 600;
    }
    a.news-link:hover {
        color: #1d4ed8 !important;
        text-decoration: underline;
    }

    /* News cards */
    .news-card {
        padding: 14px 0;
        border-bottom: 1px solid #e5e7eb;
    }
    .news-card p {
        margin: 0;
    }
    .news-date { color: #6b7280; font-size: 0.8rem; margin: 4px 0 2px 0; }
    .news-summary { color: #4b5563; font-size: 0.85rem; }

    /* Success/info banners */
    .stSuccess, .stAlert { border-radius: 8px; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Download button */
    .stDownloadButton button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton button:hover {
        background-color: #1d4ed8 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNÇÕES DE DADOS: SCREENER
# =============================================================================

def parse_br_number(text: str) -> float:
    """Converte número BR para float."""
    if not isinstance(text, str):
        return float("nan")
    text = text.strip()
    if text in ("", "-", "N/A"):
        return float("nan")
    is_pct = "%" in text
    text = text.replace("%", "").replace(".", "").replace(",", ".").strip()
    try:
        v = float(text)
        return v / 100 if is_pct else v
    except (ValueError, TypeError):
        return float("nan")


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentus_data() -> pd.DataFrame:
    """Scraping completo do Fundamentus."""
    url = "https://www.fundamentus.com.br/resultado.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.encoding = "ISO-8859-1"
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "resultado"})
    if not table:
        return pd.DataFrame()

    thead = table.find("thead")
    col_names = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    data = []
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) == len(col_names):
            data.append(cells)

    df = pd.DataFrame(data, columns=col_names)
    ticker_col = col_names[0]
    for col in df.columns:
        if col != ticker_col:
            df[col] = df[col].apply(parse_br_number)
    return df


def _col(df, *candidates):
    """Busca flexível de coluna."""
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def apply_screener(df, filters, scoring_method="magic_formula"):
    """Aplica filtros e scoring."""
    result = df.copy()

    col_liq = _col(df, "Liq.2meses")
    col_pl = _col(df, "P/L")
    col_pvp = _col(df, "P/VP")
    col_ev = _col(df, "EV/EBIT")
    col_roic = _col(df, "ROIC")
    col_roe = _col(df, "ROE")
    col_dy = _col(df, "Div.Yield")
    col_mrg = _col(df, "Mrg Ebit")
    col_div_pat = _col(df, "Dív.Brut/ Patrim.")

    # Filtros
    if col_liq and "liq_min" in filters:
        result = result[result[col_liq] >= filters["liq_min"]]
    if col_pl and "pl_min" in filters and "pl_max" in filters:
        result = result[(result[col_pl] >= filters["pl_min"]) & (result[col_pl] <= filters["pl_max"])]
    if col_ev and "ev_ebit_min" in filters and "ev_ebit_max" in filters:
        result = result[(result[col_ev] >= filters["ev_ebit_min"]) & (result[col_ev] <= filters["ev_ebit_max"])]
    if col_roic and "roic_min" in filters:
        result = result[result[col_roic] >= filters["roic_min"]]
    if col_roe and "roe_min" in filters:
        result = result[result[col_roe] >= filters["roe_min"]]
    if col_dy and "dy_min" in filters:
        result = result[result[col_dy] >= filters["dy_min"]]
    if col_div_pat and "div_pat_max" in filters:
        mask = result[col_div_pat].isna() | (result[col_div_pat] <= filters["div_pat_max"])
        result = result[mask]
    if col_pvp and "pvp_max" in filters:
        result = result[result[col_pvp] <= filters["pvp_max"]]
    if col_mrg and "mrg_ebit_min" in filters:
        result = result[result[col_mrg] >= filters["mrg_ebit_min"]]

    if len(result) == 0:
        return result

    # Scoring
    if scoring_method == "magic_formula":
        if col_ev and col_roic:
            result = result.dropna(subset=[col_ev, col_roic])
            result["EY"] = 1 / result[col_ev]
            result["Rank_EY"] = result["EY"].rank(ascending=False)
            result["Rank_ROIC"] = result[col_roic].rank(ascending=False)
            result["Score"] = result["Rank_EY"] + result["Rank_ROIC"]

    elif scoring_method == "multi_factor":
        score = pd.Series(0.0, index=result.index)
        n_factors = 0
        if col_ev:
            score += result[col_ev].rank(ascending=True)
            n_factors += 1
        if col_roic:
            score += result[col_roic].rank(ascending=False)
            n_factors += 1
        if col_pl:
            score += result[col_pl].rank(ascending=True)
            n_factors += 1
        if col_dy:
            score += result[col_dy].rank(ascending=False)
            n_factors += 1
        if col_mrg:
            score += result[col_mrg].rank(ascending=False)
            n_factors += 1
        if col_div_pat:
            score += result[col_div_pat].rank(ascending=True)
            n_factors += 1
        result["Score"] = score

    elif scoring_method == "value":
        score = pd.Series(0.0, index=result.index)
        if col_pl: score += result[col_pl].rank(ascending=True)
        if col_pvp: score += result[col_pvp].rank(ascending=True)
        if col_ev: score += result[col_ev].rank(ascending=True)
        result["Score"] = score

    elif scoring_method == "dividendos":
        score = pd.Series(0.0, index=result.index)
        if col_dy: score += result[col_dy].rank(ascending=False)
        if col_roic: score += result[col_roic].rank(ascending=False) * 0.5
        result["Score"] = score

    elif scoring_method == "quality":
        score = pd.Series(0.0, index=result.index)
        if col_roic: score += result[col_roic].rank(ascending=False)
        if col_mrg: score += result[col_mrg].rank(ascending=False)
        if col_div_pat: score += result[col_div_pat].rank(ascending=True)
        if col_roe: score += result[col_roe].rank(ascending=False)
        result["Score"] = score

    if "Score" in result.columns:
        result["Rank"] = result["Score"].rank(ascending=True, method="min").astype(int)
        result = result.sort_values("Rank")

    return result


# =============================================================================
# FUNÇÕES DE DADOS: MACRO
# =============================================================================

@st.cache_data(ttl=7200, show_spinner=False)
def get_macro_series(series_dict, start="2015-01-01"):
    """Coleta séries do SGS/BCB."""
    from bcb import sgs
    try:
        df = sgs.get(series_dict, start=start)
        return df
    except Exception:
        frames = {}
        for nome, cod in series_dict.items():
            try:
                s = sgs.get({nome: cod}, start=start)
                frames[nome] = s.iloc[:, 0]
            except Exception:
                pass
        return pd.DataFrame(frames) if frames else pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)
def get_focus_data(indicadores=None):
    """Coleta expectativas Focus."""
    from bcb import Expectativas
    if indicadores is None:
        indicadores = ["IPCA", "Selic", "PIB Total", "Câmbio"]
    ano = datetime.now().year
    anos = [ano, ano + 1]

    try:
        em = Expectativas()
        ep = em.get_endpoint("ExpectativasMercadoAnuais")
        frames = []
        for ind in indicadores:
            for a in anos:
                try:
                    d = (ep.query()
                         .filter(ep.Indicador == ind)
                         .filter(ep.DataReferencia == str(a))
                         .filter(ep.baseCalculo == 0)
                         .select(ep.Indicador, ep.Data, ep.DataReferencia,
                                 ep.Media, ep.Mediana, ep.Minimo, ep.Maximo)
                         .orderby(ep.Data.desc())
                         .limit(52)
                         .collect())
                    if len(d) > 0:
                        frames.append(d)
                except Exception:
                    pass
        if frames:
            r = pd.concat(frames, ignore_index=True)
            r["Data"] = pd.to_datetime(r["Data"])
            return r
    except Exception:
        pass
    return pd.DataFrame()


def make_macro_chart(df, col_name, title, show_pct_change=False):
    """Cria gráfico Plotly interativo com último valor anotado."""
    serie = df[col_name].dropna()
    if len(serie) == 0:
        return None

    fig = go.Figure()

    # Série principal
    fig.add_trace(go.Scatter(
        x=serie.index, y=serie.values,
        mode="lines",
        name=col_name,
        line=dict(color="#2563eb", width=2.5),
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f}<extra></extra>",
    ))

    # Último valor como anotação
    last_date = serie.index[-1]
    last_val = serie.iloc[-1]
    fig.add_annotation(
        x=last_date, y=last_val,
        text=f"<b>{last_val:.2f}</b>",
        showarrow=True, arrowhead=2, arrowcolor="#2563eb",
        font=dict(size=13, color="#ffffff"),
        bgcolor="#2563eb", bordercolor="#2563eb", borderwidth=1,
        ax=40, ay=-30,
    )

    # Variação % se pedido
    if show_pct_change and len(serie) > 1:
        pct = serie.pct_change().dropna() * 100
        fig.add_trace(go.Bar(
            x=pct.index, y=pct.values,
            name="Var. %",
            marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in pct.values],
            opacity=0.35,
            yaxis="y2",
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f}%<extra></extra>",
        ))

    layout_kwargs = dict(
        title=dict(text=title, font=dict(size=16, color="#111827")),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=400,
        margin=dict(l=50, r=50, t=50, b=40),
        hovermode="x unified",
        xaxis=dict(gridcolor="#e5e7eb", showgrid=True, linecolor="#d1d5db"),
        yaxis=dict(gridcolor="#e5e7eb", showgrid=True, title=col_name, linecolor="#d1d5db"),
        legend=dict(orientation="h", y=1.1, font=dict(color="#374151")),
    )

    if show_pct_change:
        layout_kwargs["yaxis2"] = dict(
            title="Var. %", overlaying="y", side="right",
            showgrid=False, zeroline=True, zerolinecolor="#d1d5db",
        )

    fig.update_layout(**layout_kwargs)
    return fig


def make_focus_chart(df_focus, indicador):
    """Gráfico Focus para um indicador."""
    sub = df_focus[df_focus["Indicador"] == indicador].copy()
    if sub.empty:
        return None

    fig = go.Figure()
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea"]

    for i, ref in enumerate(sorted(sub["DataReferencia"].unique())):
        s = sub[sub["DataReferencia"] == ref].sort_values("Data")
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=s["Data"], y=s["Mediana"],
            mode="lines",
            name=f"Ref: {ref}",
            line=dict(color=color, width=2.5),
            hovertemplate="%{x|%d/%m/%Y}<br>Mediana: %{y:.2f}<extra></extra>",
        ))

        # Anotar último valor
        if len(s) > 0:
            fig.add_annotation(
                x=s["Data"].iloc[0], y=s["Mediana"].iloc[0],
                text=f"<b>{s['Mediana'].iloc[0]:.2f}</b>",
                showarrow=True, arrowhead=2, arrowcolor=color,
                font=dict(size=11, color="#ffffff"),
                bgcolor=color, bordercolor=color, borderwidth=1,
                ax=30, ay=-20,
            )

    fig.update_layout(
        title=dict(text=f"Focus: {indicador}", font=dict(size=16, color="#111827")),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=350,
        margin=dict(l=50, r=50, t=50, b=40),
        hovermode="x unified",
        xaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db"),
        yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db"),
        legend=dict(orientation="h", y=1.12, font=dict(color="#374151")),
    )
    return fig


# =============================================================================
# FUNÇÕES DE DADOS: NOTÍCIAS
# =============================================================================

RSS_FEEDS_SECTIONED = {
    "🇧🇷 Brasil": {
        "InfoMoney": {
            "Últimas":       "https://www.infomoney.com.br/feed/",
            "Mercados":      "https://www.infomoney.com.br/mercados/feed/",
            "Economia":      "https://www.infomoney.com.br/economia/feed/",
            "Onde Investir":  "https://www.infomoney.com.br/onde-investir/feed/",
            "Business":      "https://www.infomoney.com.br/business/feed/",
            "Finanças":      "https://www.infomoney.com.br/minhas-financas/feed/",
            "Mundo":         "https://www.infomoney.com.br/mundo/feed/",
            "Política":      "https://www.infomoney.com.br/brasil/feed/",
        },
        "Valor Econômico": {
            "Geral": "https://pox.globo.com/rss/valor/",
        },
        "Investing BR": {
            "Geral": "https://br.investing.com/rss/news.rss",
        },
        "Money Times": {
            "Mercados": "https://www.moneytimes.com.br/mercados/feed/",
        },
    },
    "🇺🇸 US / Global": {
        "CNBC": {
            "Top News":    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
            "Markets":     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
            "Investing":   "https://www.cnbc.com/id/10000664/device/rss/rss.html",
            "Economy":     "https://www.cnbc.com/id/20910258/device/rss/rss.html",
            "Technology":  "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "Real Estate": "https://www.cnbc.com/id/10000115/device/rss/rss.html",
        },
        "MarketWatch": {
            "Top Stories":   "https://feeds.marketwatch.com/marketwatch/topstories/",
            "Markets":       "https://feeds.marketwatch.com/marketwatch/marketpulse/",
        },
        "Yahoo Finance": {
            "Geral": "https://finance.yahoo.com/news/rssindex",
        },
        "Reuters": {
            "Business":  "https://feeds.reuters.com/reuters/businessNews",
        },
        "Investing.com": {
            "Geral": "https://www.investing.com/rss/news.rss",
        },
    },
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rss_feed(url, max_items=15):
    """Busca e parseia um feed RSS."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", entry.get("updated", ""))
            summary = entry.get("summary", "")
            # Limpar HTML do summary
            if summary:
                summary = BeautifulSoup(summary, "html.parser").get_text()[:200]
            items.append({
                "title": title,
                "link": link,
                "published": published,
                "summary": summary,
            })
        return items
    except Exception:
        return []


# =============================================================================
# UI: SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 📊 Investment Toolkit")
    st.markdown("---")

    page = st.radio(
        "Navegação",
        ["🎯 Screener de Ações", "📈 Painel Macro", "📐 Renda Fixa", "🌍 Mercado Hoje", "📋 Análise CVM", "🏢 CRE Lending", "📰 Notícias"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.75rem;'>"
        "Dados: Fundamentus, BCB, RSS<br>"
        "⚠️ Não é recomendação de investimento"
        "</p>",
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE: SCREENER
# =============================================================================

if page == "🎯 Screener de Ações":

    st.markdown("# 🎯 Screener de Ações Brasileiras")
    st.markdown("Fonte: fundamentus.com.br — dados atualizados a cada 1h")

    # Carregar dados
    with st.spinner("Carregando dados do Fundamentus..."):
        df_raw = get_fundamentus_data()

    if df_raw.empty:
        st.error("Erro ao carregar dados. Verifique sua conexão.")
        st.stop()

    st.success(f"✅ {len(df_raw)} ações carregadas")

    # ─── Tabs de estratégia ──────────────────────────────────────────────
    tab_magic, tab_multi, tab_value, tab_div, tab_quality, tab_custom = st.tabs([
        "🧙 Fórmula Mágica",
        "📐 Multi-Factor",
        "💎 Value",
        "💰 Dividendos",
        "🏆 Quality",
        "🔧 Custom",
    ])

    # ─── Filtros no sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Filtros")

        liq_min = st.number_input("Liquidez mín (R$)", value=500_000, step=100_000, format="%d")
        pl_range = st.slider("P/L", 0.0, 100.0, (1.0, 50.0), step=0.5)
        ev_range = st.slider("EV/EBIT", 0.0, 100.0, (1.0, 50.0), step=0.5)
        roic_min = st.slider("ROIC mín (%)", 0, 50, 5) / 100
        roe_min = st.slider("ROE mín (%)", 0, 50, 0) / 100
        dy_min = st.slider("Div. Yield mín (%)", 0.0, 20.0, 0.0, step=0.5) / 100
        div_pat_max = st.slider("Dív/Patrim máx", 0.0, 10.0, 3.0, step=0.5)
        top_n = st.slider("Top N resultados", 10, 50, 30)

    base_filters = {
        "liq_min": liq_min,
        "pl_min": pl_range[0], "pl_max": pl_range[1],
        "ev_ebit_min": ev_range[0], "ev_ebit_max": ev_range[1],
        "roic_min": roic_min,
        "roe_min": roe_min,
        "dy_min": dy_min,
        "div_pat_max": div_pat_max,
    }

    # Colunas para exibição
    display_cols_map = {
        "Papel": "Ticker", "Cotação": "Cotação", "P/L": "P/L", "P/VP": "P/VP",
        "EV/EBIT": "EV/EBIT", "ROIC": "ROIC", "ROE": "ROE",
        "Div.Yield": "DY", "Mrg Ebit": "Mrg EBIT", "Liq.2meses": "Liquidez",
    }

    def show_screener(df_result, top_n, tab_key="default"):
        """Exibe resultados do screener formatados."""
        if len(df_result) == 0:
            st.warning("Nenhuma ação passou nos filtros. Relaxe os parâmetros.")
            return

        display = df_result.head(top_n).copy()

        # Identificar colunas existentes
        cols_show = ["Rank"]
        for orig in display_cols_map:
            c = _col(display, orig)
            if c:
                cols_show.append(c)

        cols_exist = [c for c in cols_show if c in display.columns]
        show_df = display[cols_exist].copy()

        # Formatar
        col_roic = _col(show_df, "ROIC")
        col_roe = _col(show_df, "ROE")
        col_dy = _col(show_df, "Div.Yield")
        col_mrg = _col(show_df, "Mrg Ebit")
        col_liq = _col(show_df, "Liq.2meses")

        for c in [col_roic, col_roe, col_dy, col_mrg]:
            if c and c in show_df.columns:
                show_df[c] = show_df[c].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
                )
        if col_liq and col_liq in show_df.columns:
            show_df[col_liq] = show_df[col_liq].apply(
                lambda x: f"{x/1e6:.1f}M" if pd.notna(x) and x >= 1e6
                else f"{x/1e3:.0f}k" if pd.notna(x) else "—"
            )

        # Métricas resumo
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            med_roic = display[_col(display, "ROIC")].median() if _col(display, "ROIC") else 0
            st.metric("ROIC Mediano", f"{med_roic*100:.1f}%")
        with c2:
            med_pl = display[_col(display, "P/L")].median() if _col(display, "P/L") else 0
            st.metric("P/L Mediano", f"{med_pl:.1f}")
        with c3:
            med_dy = display[_col(display, "Div.Yield")].median() if _col(display, "Div.Yield") else 0
            st.metric("DY Mediano", f"{med_dy*100:.1f}%")
        with c4:
            st.metric("Ações", f"{len(df_result)}")

        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            height=min(len(show_df) * 38 + 40, 800),
        )

        # Download
        csv = display.to_csv(index=False, sep=";")
        st.download_button("📥 Baixar CSV", csv, "screener_resultado.csv", "text/csv", key=f"dl_{tab_key}")

    # ─── Tabs ────────────────────────────────────────────────────────────

    with tab_magic:
        st.markdown("#### Fórmula Mágica de Joel Greenblatt")
        st.caption("Ranking por Earnings Yield (1/EV_EBIT) + ROIC. Menor score = melhor.")
        result = apply_screener(df_raw, base_filters, "magic_formula")
        show_screener(result, top_n, "magic")

    with tab_multi:
        st.markdown("#### Multi-Factor Scoring")
        st.caption("Combina 6 fatores: EV/EBIT, ROIC, P/L, DY, Margem EBIT, Dív/Patrim.")
        result = apply_screener(df_raw, base_filters, "multi_factor")
        show_screener(result, top_n, "multi")

    with tab_value:
        st.markdown("#### Value Screening")
        st.caption("P/L baixo + P/VP baixo + EV/EBIT baixo.")
        result = apply_screener(df_raw, base_filters, "value")
        show_screener(result, top_n, "value")

    with tab_div:
        st.markdown("#### Screening por Dividendos")
        st.caption("Dividend Yield alto + ROIC alto (sustentabilidade do payout).")
        filters_div = {**base_filters, "dy_min": max(0.01, base_filters.get("dy_min", 0))}
        result = apply_screener(df_raw, filters_div, "dividendos")
        show_screener(result, top_n, "div")

    with tab_quality:
        st.markdown("#### Quality Screening")
        st.caption("ROIC alto + ROE alto + Margem EBIT alta + Dívida baixa.")
        result = apply_screener(df_raw, base_filters, "quality")
        show_screener(result, top_n, "quality")

    with tab_custom:
        st.markdown("#### 🔧 Filtro Customizado")
        st.caption("Ajuste todos os filtros no sidebar e veja o resultado sem scoring pré-definido.")

        result = df_raw.copy()
        col_liq = _col(result, "Liq.2meses")
        col_pl = _col(result, "P/L")
        col_ev = _col(result, "EV/EBIT")
        col_roic = _col(result, "ROIC")

        if col_liq: result = result[result[col_liq] >= liq_min]
        if col_pl: result = result[(result[col_pl] >= pl_range[0]) & (result[col_pl] <= pl_range[1])]
        if col_ev: result = result[(result[col_ev] >= ev_range[0]) & (result[col_ev] <= ev_range[1])]
        if col_roic: result = result[result[col_roic] >= roic_min]

        # Sorting dinâmico
        sort_col = st.selectbox("Ordenar por", ["ROIC", "P/L", "EV/EBIT", "Div.Yield", "ROE", "Liq.2meses"])
        sort_asc = st.checkbox("Ordem crescente", value=False)
        c = _col(result, sort_col)
        if c:
            result = result.sort_values(c, ascending=sort_asc)
        result["Rank"] = range(1, len(result) + 1)
        show_screener(result, top_n, "custom")


# =============================================================================
# PAGE: MACRO
# =============================================================================

elif page == "📈 Painel Macro":

    st.markdown("# 📈 Painel Macroeconômico Brasil")
    st.markdown("Fonte: Banco Central do Brasil (SGS + Focus)")

    # Período
    with st.sidebar:
        st.markdown("### 📅 Período")
        start_year = st.slider("Ano inicial", 2005, 2025, 2018)
        show_pct = st.checkbox("Mostrar variação %", value=False)

    start_date = f"{start_year}-01-01"

    # ─── Séries macro ────────────────────────────────────────────────────

    series_config = {
        # Juros
        "SELIC Meta (%)": 432,
        "CDI (%)": 4389,
        # Inflação
        "IPCA Mensal (%)": 433,
        "IPCA 12m (%)": 13522,
        "IGP-M Mensal (%)": 189,
        "INPC Mensal (%)": 188,
        # Câmbio
        "USD/BRL": 1,
        "EUR/BRL": 21619,
        # Atividade
        "IBC-Br": 24364,
        "Produção Industrial (%)": 21859,
        # Fiscal
        "Dívida/PIB (%)": 13762,
        "Resultado Primário/PIB (%)": 4505,
        # Emprego
        "Desemprego (%)": 24369,
        # Crédito
        "Crédito/PIB (%)": 20622,
        "Inadimplência (%)": 20714,
    }

    with st.spinner("Carregando dados do Banco Central..."):
        df_macro = get_macro_series(series_config, start_date)

    if not df_macro.empty:
        # Métricas do último ponto
        st.markdown("### Últimos valores")
        priority_row1 = ["SELIC Meta (%)", "IPCA 12m (%)", "USD/BRL", "Desemprego (%)"]
        priority_row2 = ["CDI (%)", "IGP-M Mensal (%)", "EUR/BRL", "Dívida/PIB (%)"]

        cols1 = st.columns(4)
        for i, nome in enumerate(priority_row1):
            if nome in df_macro.columns:
                serie = df_macro[nome].dropna()
                if len(serie) > 0:
                    val = serie.iloc[-1]
                    data = serie.index[-1].strftime("%d/%m/%Y")
                    delta = None
                    if len(serie) > 1:
                        prev = serie.iloc[-2]
                        if prev != 0:
                            delta = f"{((val - prev) / abs(prev)) * 100:+.2f}%"
                    with cols1[i]:
                        st.metric(nome, f"{val:.2f}", delta=delta, help=f"Último dado: {data}")

        cols2 = st.columns(4)
        for i, nome in enumerate(priority_row2):
            if nome in df_macro.columns:
                serie = df_macro[nome].dropna()
                if len(serie) > 0:
                    val = serie.iloc[-1]
                    data = serie.index[-1].strftime("%d/%m/%Y")
                    delta = None
                    if len(serie) > 1:
                        prev = serie.iloc[-2]
                        if prev != 0:
                            delta = f"{((val - prev) / abs(prev)) * 100:+.2f}%"
                    with cols2[i]:
                        st.metric(nome, f"{val:.2f}", delta=delta, help=f"Último dado: {data}")

        st.markdown("---")

        # Gráficos interativos
        st.markdown("### Séries históricas")

        chart_groups = {
            "Juros": ["SELIC Meta (%)", "CDI (%)"],
            "Inflação": ["IPCA 12m (%)", "IPCA Mensal (%)", "IGP-M Mensal (%)", "INPC Mensal (%)"],
            "Câmbio": ["USD/BRL", "EUR/BRL"],
            "Atividade": ["IBC-Br", "Produção Industrial (%)"],
            "Fiscal": ["Dívida/PIB (%)", "Resultado Primário/PIB (%)"],
            "Emprego": ["Desemprego (%)"],
            "Crédito": ["Crédito/PIB (%)", "Inadimplência (%)"],
        }

        for group_name, series_names in chart_groups.items():
            available = [s for s in series_names if s in df_macro.columns]
            if not available:
                continue

            for s in available:
                fig = make_macro_chart(df_macro, s, f"{group_name}: {s}", show_pct_change=show_pct)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # Tabela com variações
        st.markdown("### Variações recentes")
        var_data = []
        for col in df_macro.columns:
            serie = df_macro[col].dropna()
            if len(serie) < 2:
                continue
            last = serie.iloc[-1]
            prev = serie.iloc[-2]
            var_abs = last - prev
            var_pct = ((last - prev) / abs(prev) * 100) if prev != 0 else 0
            # YoY se tiver dados suficientes
            yoy = None
            idx_12m = serie.index.searchsorted(serie.index[-1] - pd.DateOffset(months=12))
            if idx_12m < len(serie):
                val_12m = serie.iloc[idx_12m]
                if val_12m != 0:
                    yoy = ((last - val_12m) / abs(val_12m)) * 100

            var_data.append({
                "Indicador": col,
                "Último": f"{last:.2f}",
                "Data": serie.index[-1].strftime("%d/%m/%Y"),
                "Var. Período": f"{var_abs:+.2f}",
                "Var. %": f"{var_pct:+.2f}%",
                "Var. 12m %": f"{yoy:+.2f}%" if yoy else "—",
            })

        if var_data:
            st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

    # ─── Focus ───────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### Expectativas de Mercado (Focus)")

    with st.spinner("Carregando Focus..."):
        df_focus = get_focus_data()

    if not df_focus.empty:
        # Medianas mais recentes
        ultimo_dia = df_focus["Data"].max()
        recente = df_focus[df_focus["Data"] == ultimo_dia]

        cols_focus = st.columns(4)
        for i, (_, row) in enumerate(recente.iterrows()):
            with cols_focus[i % 4]:
                st.metric(
                    f"{row['Indicador']} ({row['DataReferencia']})",
                    f"{row['Mediana']:.2f}",
                )

        # Gráficos Focus
        for ind in df_focus["Indicador"].unique():
            fig = make_focus_chart(df_focus, ind)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não foi possível carregar dados do Focus.")


# =============================================================================
# PAGE: RENDA FIXA (Curva DI + Inflação Implícita)
# =============================================================================

elif page == "📐 Renda Fixa":

    st.markdown("# 📐 Renda Fixa — Curvas de Juros")
    st.markdown("ETTJ Prefixada, IPCA e Inflação Implícita — Fonte: ANBIMA / B3")
    st.caption("⚠️ As taxas podem divergir ~10-15bps da tabela oficial ANBIMA. O pyettj retorna dados em dias corridos (B3), enquanto a ANBIMA publica em dias úteis. A conversão é aproximada. Para valores exatos, consulte anbima.com.br/informacoes/est-termo.")

    # ─── Imports específicos ─────────────────────────────────────────────

    try:
        from pyettj import ettj as ettj_mod
        PYETTJ_OK = True
    except ImportError:
        PYETTJ_OK = False

    # ─── Data fetchers ───────────────────────────────────────────────────

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_ettj_curve(data_str, curva="PRE"):
        """Busca curva ETTJ da ANBIMA/B3 via pyettj."""
        if not PYETTJ_OK:
            return pd.DataFrame()
        try:
            df = ettj_mod.get_ettj(data_str, curva=curva)
            return df
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_di_futures():
        """Busca cotações ao vivo dos contratos DI1 via API da B3."""
        try:
            url = "https://cotacao.b3.com.br/mds/api/v1/DerivativeQuotation/DI1"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            rows = []
            for item in data.get("Scty", []):
                ticker = item.get("symb", "")
                qtn = item.get("SctyQtn", {})
                asset = item.get("asset", {}).get("AsstSummry", {})
                buy = item.get("buyOffer", {})
                sell = item.get("sellOffer", {})

                maturity = asset.get("mtrtyCode", "")
                adj_price = qtn.get("prvsDayAdjstmntPric", 0)
                bottom = qtn.get("bottomLmtPric", 0)
                top = qtn.get("topLmtPric", 0)
                bid = buy.get("price", 0)
                ask = sell.get("price", 0)
                open_int = asset.get("opnCtrcts", 0)

                rows.append({
                    "Contrato": ticker,
                    "Vencimento": maturity,
                    "Ajuste Ant.": adj_price,
                    "Bid": bid,
                    "Ask": ask,
                    "Lim. Inf.": bottom,
                    "Lim. Sup.": top,
                    "Contratos Abertos": open_int,
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values("Vencimento").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=7200, show_spinner=False)
    def get_us_real_yields():
        """Busca yields reais dos US TIPS via FRED."""
        try:
            import pandas_datareader.data as web
            series = {
                "US 5Y Real": "DFII5",
                "US 7Y Real": "DFII7",
                "US 10Y Real": "DFII10",
                "US 20Y Real": "DFII20",
                "US 30Y Real": "DFII30",
                "US 10Y Nominal": "DGS10",
                "US 2Y Nominal": "DGS2",
                "US 5Y Nominal": "DGS5",
                "US 30Y Nominal": "DGS30",
            }
            from datetime import datetime as dt
            frames = {}
            for name, code in series.items():
                try:
                    d = web.DataReader(code, "fred", dt(2015, 1, 1))
                    frames[name] = d.iloc[:, 0]
                except Exception:
                    pass
            return pd.DataFrame(frames) if frames else pd.DataFrame()
        except ImportError:
            return pd.DataFrame()

    # ─── Helper: formatar data para pyettj ───────────────────────────────

    def fmt_date_ettj(dt_obj):
        """Converte datetime para formato dd/mm/yyyy."""
        return dt_obj.strftime("%d/%m/%Y")

    # ─── Calendário ANBIMA para conversão DC → DU ────────────────────────

    @st.cache_data(ttl=86400, show_spinner=False)
    def get_anbima_calendar():
        """Cria calendário de dias úteis ANBIMA via bizdays."""
        try:
            from bizdays import Calendar
            try:
                # Tentar baixar feriados da ANBIMA
                holidays = pd.read_excel(
                    "https://www.anbima.com.br/feriados/arqs/feriados_nacionais.xls",
                    skipfooter=9
                )["Data"].tolist()
            except Exception:
                # Fallback: feriados fixos mais comuns
                holidays = []

            cal = Calendar(holidays, ["Saturday", "Sunday"], name="ANBIMA")
            return cal
        except ImportError:
            return None

    cal_anbima = get_anbima_calendar()

    def dc_to_du(dias_corridos, ref_date):
        """Converte dias corridos para dias úteis a partir de uma data de referência."""
        if cal_anbima is None:
            # Fallback: aproximação DC * 252/365
            return int(round(dias_corridos * 252 / 365))
        try:
            from datetime import date as dt_date
            if hasattr(ref_date, 'date'):
                ref = ref_date.date() if not isinstance(ref_date, dt_date) else ref_date
            else:
                ref = ref_date
            target = ref + timedelta(days=int(dias_corridos))
            return cal_anbima.bizdays(ref, target)
        except Exception:
            return int(round(dias_corridos * 252 / 365))

    def normalize_ettj(df, name="Taxa", ref_date=None):
        """
        Normaliza colunas do pyettj para nomes padrão.
        Converte Dias Corridos → Dias Úteis usando calendário ANBIMA.
        """
        if df.empty or len(df.columns) < 2:
            return pd.DataFrame()
        result = df[[df.columns[0], df.columns[1]]].copy()
        result.columns = ["DC", name]
        result["DC"] = pd.to_numeric(result["DC"], errors="coerce")
        result[name] = pd.to_numeric(result[name], errors="coerce")
        result = result.dropna()

        # Converter DC → DU
        if ref_date is not None:
            result["DU"] = result["DC"].apply(lambda x: dc_to_du(x, ref_date))
        else:
            # Fallback: aproximação
            result["DU"] = (result["DC"] * 252 / 365).round().astype(int)

        return result

    MAX_YEARS = 12  # Limitar curva a 12 anos (onde é líquida)

    # ─── Sidebar ─────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("### 📅 Datas Curvas")
        today = datetime.now()

        # Forçar último dia útil ANTERIOR (ANBIMA publica D-1)
        ref_date = today - timedelta(days=1)
        while ref_date.weekday() >= 5:  # pular fim de semana
            ref_date = ref_date - timedelta(days=1)

        dt_atual = st.date_input("Data atual", value=ref_date, key="rf_dt1")
        dt_comp = st.date_input("Comparar com", value=ref_date - timedelta(days=7), key="rf_dt2")
        dt_comp2 = st.date_input("Comparar com (2)", value=ref_date - timedelta(days=30), key="rf_dt3")
        max_anos = st.slider("Prazo máximo (anos)", 2, 30, MAX_YEARS, key="rf_max_anos")

    # ─── Tabs ────────────────────────────────────────────────────────────

    tab_pre, tab_ipca, tab_breakeven, tab_di, tab_us = st.tabs([
        "📈 Curva PRE",
        "📊 Curva IPCA",
        "🔥 Inflação Implícita",
        "💹 DI Futuro (Live)",
        "🇺🇸 US Yields",
    ])

    if not PYETTJ_OK:
        st.error("Biblioteca `pyettj` não instalada. Rode: `pip install pyettj bizdays html5lib`")

    def plot_curve(curva_tipo, title, dates_list, max_anos_val):
        """Plota curva ETTJ para múltiplas datas."""
        fig = go.Figure()
        colors = ["#2563eb", "#ea580c", "#16a34a"]
        all_dfs = {}

        for i, (label, dt_val) in enumerate(dates_list):
            df_raw = get_ettj_curve(fmt_date_ettj(dt_val), curva_tipo)
            if df_raw.empty:
                continue
            df = normalize_ettj(df_raw, "Taxa", ref_date=dt_val)
            if df.empty:
                continue
            df["Anos"] = df["DU"] / 252
            df = df[df["Anos"] <= max_anos_val]  # LIMITAR
            all_dfs[label] = df

            fig.add_trace(go.Scatter(
                x=df["Anos"], y=df["Taxa"],
                mode="lines+markers",
                name=f"{label} ({dt_val.strftime('%d/%m')})",
                line=dict(color=colors[i], width=2.5),
                marker=dict(size=3),
                hovertemplate="Prazo: %{x:.1f}a<br>Taxa: %{y:.2f}%<extra></extra>",
            ))

        if fig.data:
            fig.update_layout(
                title=dict(text=title, font=dict(size=16, color="#111827")),
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff", height=500,
                xaxis=dict(title="Prazo (anos)", gridcolor="#e5e7eb", range=[0, max_anos_val]),
                yaxis=dict(title="Taxa (% a.a.)", gridcolor="#e5e7eb"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

        return all_dfs

    # ─── Tab: Curva PRE ──────────────────────────────────────────────────

    with tab_pre:
        st.markdown("### Curva Prefixada (ETTJ PRE)")
        st.caption("Fonte: ANBIMA — taxas zero-cupom de títulos prefixados")

        if PYETTJ_OK:
            dates_to_fetch = [
                ("Atual", dt_atual),
                ("Semana anterior", dt_comp),
                ("Mês anterior", dt_comp2),
            ]
            dfs_pre = plot_curve("PRE", "ETTJ Prefixada", dates_to_fetch, max_anos)

            if len(dfs_pre) >= 2:
                # Tabela de variações
                st.markdown("#### Variação entre datas")
                keys = list(dfs_pre.keys())
                df_a = dfs_pre[keys[0]].set_index("DU")["Taxa"]
                df_b = dfs_pre[keys[1]].set_index("DU")["Taxa"]
                common = df_a.index.intersection(df_b.index)
                if len(common) > 0:
                    var_df = pd.DataFrame({
                        "Prazo (anos)": common / 252,
                        f"{keys[0]} (%)": df_a.loc[common].values,
                        f"{keys[1]} (%)": df_b.loc[common].values,
                        "Var (bps)": (df_a.loc[common].values - df_b.loc[common].values) * 100,
                    })
                    # Filtrar vértices relevantes
                    vertices_du = [21, 42, 63, 126, 252, 504, 756, 1260, 1890, 2520]
                    var_display = var_df[var_df["Prazo (anos)"].apply(
                        lambda x: any(abs(x - v/252) < 0.05 for v in vertices_du)
                    )].copy()
                    if var_display.empty:
                        var_display = var_df.iloc[::max(1, len(var_df)//15)]
                    var_display["Prazo (anos)"] = var_display["Prazo (anos)"].apply(lambda x: f"{x:.1f}")
                    var_display[f"{keys[0]} (%)"] = var_display[f"{keys[0]} (%)"].apply(lambda x: f"{x:.2f}")
                    var_display[f"{keys[1]} (%)"] = var_display[f"{keys[1]} (%)"].apply(lambda x: f"{x:.2f}")
                    var_display["Var (bps)"] = var_display["Var (bps)"].apply(lambda x: f"{x:+.0f}")
                    st.dataframe(var_display, use_container_width=True, hide_index=True)

            if not dfs_pre:
                st.warning("Não foi possível carregar a curva PRE. Ajuste a data para um dia útil.")

    # ─── Tab: Curva IPCA ─────────────────────────────────────────────────

    with tab_ipca:
        st.markdown("### Curva Juros Reais (ETTJ IPCA)")
        st.caption("Fonte: ANBIMA — taxas de NTN-B (juros reais + IPCA)")

        if PYETTJ_OK:
            dates_to_fetch = [
                ("Atual", dt_atual),
                ("Semana anterior", dt_comp),
                ("Mês anterior", dt_comp2),
            ]
            dfs_ipca = plot_curve("DIC", "ETTJ IPCA (Juros Reais)", dates_to_fetch, max_anos)
            if not dfs_ipca:
                dfs_ipca = plot_curve("DOC", "ETTJ IPCA (Juros Reais)", dates_to_fetch, max_anos)
            if not dfs_ipca:
                st.warning("Curva IPCA não disponível. Tente ajustar a data.")

    # ─── Tab: Inflação Implícita ─────────────────────────────────────────

    with tab_breakeven:
        st.markdown("### Inflação Implícita (Breakeven)")
        st.caption("PRE - IPCA = expectativa de inflação embutida na curva")

        if PYETTJ_OK:
            df_pre_raw = get_ettj_curve(fmt_date_ettj(dt_atual), "PRE")
            df_ipca_raw = get_ettj_curve(fmt_date_ettj(dt_atual), "DIC")
            if df_ipca_raw.empty:
                df_ipca_raw = get_ettj_curve(fmt_date_ettj(dt_atual), "DOC")

            df_pre_n = normalize_ettj(df_pre_raw, "PRE", ref_date=dt_atual)
            df_ipca_n = normalize_ettj(df_ipca_raw, "IPCA", ref_date=dt_atual)

            if not df_pre_n.empty and not df_ipca_n.empty:
                # Merge por DU
                merged = df_pre_n.merge(df_ipca_n, on="DU", how="inner")
                merged["Anos"] = merged["DU"] / 252
                merged = merged[merged["Anos"] <= max_anos]
                merged["Inflação Implícita"] = ((1 + merged["PRE"]/100) / (1 + merged["IPCA"]/100) - 1) * 100

                if len(merged) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=merged["Anos"], y=merged["PRE"],
                        mode="lines", name="PRE (Nominal)",
                        line=dict(color="#2563eb", width=2.5),
                    ))
                    fig.add_trace(go.Scatter(
                        x=merged["Anos"], y=merged["IPCA"],
                        mode="lines", name="IPCA (Real)",
                        line=dict(color="#16a34a", width=2.5),
                    ))
                    fig.add_trace(go.Scatter(
                        x=merged["Anos"], y=merged["Inflação Implícita"],
                        mode="lines+markers", name="Inflação Implícita",
                        line=dict(color="#dc2626", width=3, dash="dash"),
                        marker=dict(size=5),
                        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                    ))
                    fig.update_layout(
                        title=dict(text=f"Inflação Implícita — {dt_atual.strftime('%d/%m/%Y')}",
                                   font=dict(size=16, color="#111827")),
                        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#ffffff", height=500,
                        xaxis=dict(title="Prazo (anos)", gridcolor="#e5e7eb", range=[0, max_anos]),
                        yaxis=dict(title="Taxa (% a.a.)", gridcolor="#e5e7eb"),
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.12),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Vértices principais
                    st.markdown("#### Vértices Principais")
                    vertices = [1, 2, 3, 5, 7, 10]
                    vert_data = []
                    for v in vertices:
                        if v > max_anos:
                            continue
                        closest_idx = (merged["Anos"] - v).abs().idxmin()
                        row = merged.loc[closest_idx]
                        vert_data.append({
                            "Prazo": f"{v}A",
                            "PRE (%)": f"{row['PRE']:.2f}",
                            "IPCA (%)": f"{row['IPCA']:.2f}",
                            "Inflação Impl. (%)": f"{row['Inflação Implícita']:.2f}",
                        })
                    if vert_data:
                        st.dataframe(pd.DataFrame(vert_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("Sem vértices em comum entre PRE e IPCA para esta data.")
            else:
                st.warning("Não foi possível calcular — faltam dados PRE ou IPCA para a data selecionada.")
                st.caption("Selecione o último dia útil (ANBIMA não publica no dia atual, fins de semana ou feriados).")

    # ─── Tab: DI Futuro Live ─────────────────────────────────────────────

    with tab_di:
        st.markdown("### Contratos DI Futuro — Cotações")
        st.caption("Fonte: API B3 (cotação.b3.com.br)")

        with st.spinner("Carregando DI futures..."):
            df_di = get_di_futures()

        if not df_di.empty:
            st.success(f"✅ {len(df_di)} contratos carregados")

            # Chart: curva DI implícita
            df_plot = df_di[df_di["Ajuste Ant."] > 0].copy()
            if not df_plot.empty and "Vencimento" in df_plot.columns:
                df_plot["Venc_dt"] = pd.to_datetime(df_plot["Vencimento"], errors="coerce")
                df_plot = df_plot.dropna(subset=["Venc_dt"]).sort_values("Venc_dt")
                df_plot["Anos"] = (df_plot["Venc_dt"] - pd.Timestamp.now()).dt.days / 365

                fig_di = go.Figure()
                fig_di.add_trace(go.Scatter(
                    x=df_plot["Anos"], y=df_plot["Ajuste Ant."],
                    mode="lines+markers", name="Ajuste Anterior",
                    line=dict(color="#2563eb", width=2.5), marker=dict(size=5),
                    hovertemplate="%{text}<br>Taxa: %{y:.3f}%<br>Prazo: %{x:.1f}a<extra></extra>",
                    text=df_plot["Contrato"],
                ))
                if "Bid" in df_plot.columns:
                    fig_di.add_trace(go.Scatter(
                        x=df_plot["Anos"], y=df_plot["Bid"],
                        mode="lines", name="Bid",
                        line=dict(color="#16a34a", width=1, dash="dot"), opacity=0.7,
                    ))
                    fig_di.add_trace(go.Scatter(
                        x=df_plot["Anos"], y=df_plot["Ask"],
                        mode="lines", name="Ask",
                        line=dict(color="#dc2626", width=1, dash="dot"), opacity=0.7,
                    ))
                fig_di.update_layout(
                    title=dict(text="Curva DI Futuro (B3 Live)", font=dict(size=16, color="#111827")),
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff", height=450,
                    xaxis=dict(title="Prazo (anos)", gridcolor="#e5e7eb"),
                    yaxis=dict(title="Taxa (% a.a.)", gridcolor="#e5e7eb"),
                    hovermode="x unified", legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_di, use_container_width=True)

            # Tabela
            st.dataframe(df_di, use_container_width=True, hide_index=True, height=500)
        else:
            st.warning("API da B3 indisponível. Pode estar fora do horário de pregão.")
            st.caption("A API retorna dados durante o horário de negociação (10h-18h).")

    # ─── Tab: US Yields ──────────────────────────────────────────────────

    with tab_us:
        st.markdown("### Curva de Juros EUA")
        st.caption("Treasury Yields (nominais e reais/TIPS) — Fonte: FRED")

        with st.spinner("Carregando US yields..."):
            df_us = get_us_real_yields()

        if not df_us.empty:
            # Último ponto
            st.markdown("#### Últimos valores")
            cols_m = st.columns(4)
            priority = ["US 10Y Nominal", "US 2Y Nominal", "US 10Y Real", "US 5Y Real"]
            for i, k in enumerate(priority):
                if k in df_us.columns:
                    s = df_us[k].dropna()
                    if len(s) > 0:
                        with cols_m[i]:
                            st.metric(k, f"{s.iloc[-1]:.2f}%")

            # Gráfico nominal
            fig_nom = go.Figure()
            nom_cols = [c for c in df_us.columns if "Nominal" in c]
            colors_n = ["#2563eb", "#60a5fa", "#93c5fd", "#bfdbfe"]
            for i, c in enumerate(nom_cols):
                s = df_us[c].dropna()
                fig_nom.add_trace(go.Scatter(
                    x=s.index, y=s.values, mode="lines",
                    name=c, line=dict(color=colors_n[i % len(colors_n)], width=2),
                ))
            fig_nom.update_layout(
                title=dict(text="US Treasury Yields (Nominal)", font=dict(size=15, color="#111827")),
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff", height=400,
                xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="%"),
                hovermode="x unified", legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_nom, use_container_width=True)

            # Gráfico real
            fig_real = go.Figure()
            real_cols = [c for c in df_us.columns if "Real" in c]
            colors_r = ["#dc2626", "#f97316", "#eab308", "#84cc16", "#22c55e"]
            for i, c in enumerate(real_cols):
                s = df_us[c].dropna()
                fig_real.add_trace(go.Scatter(
                    x=s.index, y=s.values, mode="lines",
                    name=c, line=dict(color=colors_r[i % len(colors_r)], width=2),
                ))
            fig_real.add_hline(y=0, line_dash="dash", line_color="#6b7280", opacity=0.5)
            fig_real.update_layout(
                title=dict(text="US TIPS Yields (Real)", font=dict(size=15, color="#111827")),
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff", height=400,
                xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="%"),
                hovermode="x unified", legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_real, use_container_width=True)

            # US Breakeven
            if "US 10Y Nominal" in df_us.columns and "US 10Y Real" in df_us.columns:
                be = (df_us["US 10Y Nominal"] - df_us["US 10Y Real"]).dropna()
                fig_be = go.Figure()
                fig_be.add_trace(go.Scatter(
                    x=be.index, y=be.values, mode="lines",
                    name="US 10Y Breakeven",
                    line=dict(color="#dc2626", width=2.5),
                    fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                ))
                last_be = be.iloc[-1]
                fig_be.add_annotation(
                    x=be.index[-1], y=last_be,
                    text=f"<b>{last_be:.2f}%</b>", showarrow=True,
                    arrowcolor="#dc2626", font=dict(color="#fff", size=12),
                    bgcolor="#dc2626", ax=40, ay=-25,
                )
                fig_be.update_layout(
                    title=dict(text="US 10Y Breakeven Inflation", font=dict(size=15, color="#111827")),
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff", height=350,
                    xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="%"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_be, use_container_width=True)
        else:
            st.warning("Não foi possível carregar dados do FRED.")
            st.caption("Instale: `pip install pandas-datareader`")


# =============================================================================
# PAGE: MERCADO HOJE
# =============================================================================

elif page == "🌍 Mercado Hoje":

    st.markdown("# 🌍 Mercado Hoje")
    st.markdown("Preços e variações — atualizado a cada 2h")

    import yfinance as yf
    from datetime import timedelta

    # ─── Watchlists ──────────────────────────────────────────────────────

    WATCHLISTS = {
        "🌐 Índices Globais": {
            "IBOV": "^BVSP",
            "S&P 500": "^GSPC",
            "Nasdaq": "^IXIC",
            "Dow Jones": "^DJI",
            "FTSE 100": "^FTSE",
            "DAX": "^GDAXI",
            "Nikkei 225": "^N225",
            "KOSPI": "^KS11",
            "Shanghai": "000001.SS",
            "Hang Seng": "^HSI",
            "Euro Stoxx 50": "^STOXX50E",
            "MSCI EM ETF": "EEM",
        },
        "🇧🇷 Ações Brasil": {
            "PETR4": "PETR4.SA", "VALE3": "VALE3.SA", "ITUB4": "ITUB4.SA",
            "BBDC4": "BBDC4.SA", "ABEV3": "ABEV3.SA", "WEGE3": "WEGE3.SA",
            "B3SA3": "B3SA3.SA", "RENT3": "RENT3.SA", "LREN3": "LREN3.SA",
            "MGLU3": "MGLU3.SA", "BBAS3": "BBAS3.SA", "SUZB3": "SUZB3.SA",
            "JBSS3": "JBSS3.SA", "GGBR4": "GGBR4.SA", "CSNA3": "CSNA3.SA",
            "CPLE6": "CPLE6.SA", "ELET3": "ELET3.SA", "EQTL3": "EQTL3.SA",
            "RADL3": "RADL3.SA", "TOTS3": "TOTS3.SA",
        },
        "🇺🇸 Ações US": {
            "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
            "Amazon": "AMZN", "Nvidia": "NVDA", "Meta": "META",
            "Tesla": "TSLA", "JPMorgan": "JPM", "Berkshire": "BRK-B",
            "Visa": "V", "UnitedHealth": "UNH", "Johnson&Johnson": "JNJ",
            "Walmart": "WMT", "Mastercard": "MA", "Netflix": "NFLX",
            "AMD": "AMD", "Broadcom": "AVGO", "Costco": "COST",
        },
        "₿ Crypto": {
            "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
            "BNB": "BNB-USD", "XRP": "XRP-USD", "Cardano": "ADA-USD",
            "Avalanche": "AVAX-USD", "Polkadot": "DOT-USD", "Chainlink": "LINK-USD",
            "Polygon": "MATIC-USD",
        },
        "💱 Moedas & Commodities": {
            "USD/BRL": "BRL=X", "EUR/BRL": "EURBRL=X", "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
            "Ouro": "GC=F", "Prata": "SI=F", "Petróleo WTI": "CL=F",
            "Petróleo Brent": "BZ=F", "Minério Ferro": "TIO=F",
            "DXY (Dólar Index)": "DX-Y.NYB",
        },
    }

    @st.cache_data(ttl=7200, show_spinner=False)
    def get_market_data(tickers_dict):
        """Busca preços e calcula variações para uma watchlist."""
        today = datetime.now()
        start_3y = today - timedelta(days=3 * 365 + 30)

        # Download all tickers at once
        symbols = list(tickers_dict.values())
        names = list(tickers_dict.keys())

        try:
            data = yf.download(symbols, start=start_3y.strftime("%Y-%m-%d"),
                               progress=False, auto_adjust=True, threads=True)
        except Exception as e:
            return pd.DataFrame(), str(e)

        if data.empty:
            return pd.DataFrame(), "Sem dados retornados"

        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data

        # If single ticker, wrap in DataFrame
        if isinstance(close, pd.Series):
            close = close.to_frame(columns=[symbols[0]])

        results = []
        for name, symbol in tickers_dict.items():
            if symbol not in close.columns:
                continue

            prices = close[symbol].dropna()
            if len(prices) < 2:
                continue

            last_price = prices.iloc[-1]
            last_date = prices.index[-1]

            def pct_change(days):
                target_date = last_date - timedelta(days=days)
                mask = prices.index <= target_date
                if mask.any():
                    ref = prices[mask].iloc[-1]
                    return ((last_price - ref) / ref) * 100
                return None

            # YTD
            ytd_start = prices[prices.index >= f"{today.year}-01-01"]
            ytd_pct = None
            if len(ytd_start) > 0:
                ytd_ref = ytd_start.iloc[0]
                ytd_pct = ((last_price - ytd_ref) / ytd_ref) * 100

            results.append({
                "Ativo": name,
                "Preço": last_price,
                "1D (%)": pct_change(1),
                "1M (%)": pct_change(30),
                "YTD (%)": ytd_pct,
                "1A (%)": pct_change(365),
                "3A (%)": pct_change(3 * 365),
            })

        return pd.DataFrame(results), ""

    def style_pct(val):
        """Formata e colore variações percentuais."""
        if pd.isna(val) or val is None:
            return "—"
        color = "#16a34a" if val >= 0 else "#dc2626"
        arrow = "▲" if val >= 0 else "▼"
        return f":{color}[{arrow} {val:+.2f}%]"

    def format_price(val):
        """Formata preço."""
        if pd.isna(val):
            return "—"
        if val >= 1000:
            return f"{val:,.0f}"
        elif val >= 1:
            return f"{val:,.2f}"
        else:
            return f"{val:,.4f}"

    def show_watchlist(name, tickers):
        """Exibe uma watchlist com métricas e tabela."""
        with st.spinner(f"Carregando {name}..."):
            df, err = get_market_data(tickers)

        if df.empty:
            st.warning(f"Não foi possível carregar dados. {err}")
            return

        # Formatar para exibição
        display = df.copy()

        # Formatar preço
        display["Preço"] = display["Preço"].apply(format_price)

        # Formatar percentuais com cores via HTML
        pct_cols = ["1D (%)", "1M (%)", "YTD (%)", "1A (%)", "3A (%)"]
        for col in pct_cols:
            display[col] = display[col].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) and x is not None else "—"
            )

        # Color styling function
        def color_cell(val):
            if val == "—":
                return "color: #9ca3af"
            try:
                num = float(val.replace("%", "").replace("+", ""))
                if num > 0:
                    return "color: #16a34a; font-weight: 600"
                elif num < 0:
                    return "color: #dc2626; font-weight: 600"
            except (ValueError, AttributeError):
                pass
            return ""

        # Use map (pandas>=2.1) or applymap (older)
        try:
            styled = display.style.map(color_cell, subset=pct_cols)
        except AttributeError:
            styled = display.style.applymap(color_cell, subset=pct_cols)
        styled = styled.set_properties(**{"text-align": "right"}, subset=["Preço"] + pct_cols)
        styled = styled.set_properties(**{"text-align": "left", "font-weight": "600"}, subset=["Ativo"])

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=min(len(display) * 38 + 40, 700),
        )

        # Top movers
        df_sorted = df.dropna(subset=["1D (%)"])
        if len(df_sorted) > 0:
            c1, c2 = st.columns(2)
            top = df_sorted.nlargest(3, "1D (%)")
            bottom = df_sorted.nsmallest(3, "1D (%)")
            with c1:
                st.markdown("**🟢 Maiores altas (1D)**")
                for _, r in top.iterrows():
                    st.markdown(f"- **{r['Ativo']}**: {r['1D (%)']:+.2f}%")
            with c2:
                st.markdown("**🔴 Maiores quedas (1D)**")
                for _, r in bottom.iterrows():
                    st.markdown(f"- **{r['Ativo']}**: {r['1D (%)']:+.2f}%")

    # ─── Custom tickers ──────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("### ➕ Ativo customizado")
        custom_ticker = st.text_input("Ticker Yahoo (ex: AAPL, PETR4.SA)", key="custom_tick")

    # ─── Render tabs ─────────────────────────────────────────────────────

    tab_names = list(WATCHLISTS.keys()) + ["📊 IBOV Composição", "🔍 Gráfico de Preço", "💸 Fluxo Estrangeiro"]
    tabs = st.tabs(tab_names)

    # ─── Watchlist tabs ──────────────────────────────────────────────────

    for tab, (wl_name, tickers) in zip(tabs[:len(WATCHLISTS)], WATCHLISTS.items()):
        with tab:
            active_tickers = dict(tickers)
            if custom_ticker:
                active_tickers[custom_ticker.upper()] = custom_ticker
            show_watchlist(wl_name, active_tickers)

    # ─── Tab IBOV Composição ─────────────────────────────────────────────

    with tabs[len(WATCHLISTS)]:
        IBOV_SETORES = {
            "Financeiro": ["ITUB4", "BBDC4", "BBAS3", "SANB11", "BPAC11", "B3SA3", "BBSE3"],
            "Petróleo & Gás": ["PETR4", "PETR3", "PRIO3", "BRAV3", "CSAN3", "UGPA3", "VBBR3"],
            "Mineração & Siderurgia": ["VALE3", "CSNA3", "GGBR4", "GOAU4", "CMIN3"],
            "Utilities": ["ELET3", "ELET6", "EQTL3", "ENGI11", "CPFE3", "CMIG4", "CPLE6", "SBSP3", "ENEV3"],
            "Consumo & Varejo": ["ABEV3", "LREN3", "MGLU3", "PETZ3", "ARZZ3", "SOMA3", "NTCO3", "ASAI3"],
            "Saúde": ["RDOR3", "HAPV3", "RADL3", "FLRY3"],
            "Indústria": ["WEGE3", "EMBR3", "SUZB3", "KLBN11"],
            "Telecom & Tech": ["VIVT3", "TOTS3", "CASH3"],
            "Imobiliário": ["MULT3", "CYRE3", "MRVE3", "EZTC3"],
            "Alimentos & Agro": ["JBSS3", "BRFS3", "BEEF3", "SMTO3", "SLCE3"],
        }

        @st.cache_data(ttl=7200, show_spinner=False)
        def get_ibov_sector_weights():
            all_tickers = []
            ticker_sector = {}
            for setor, tickers in IBOV_SETORES.items():
                for t in tickers:
                    all_tickers.append(t + ".SA")
                    ticker_sector[t + ".SA"] = setor
            try:
                data = yf.download(all_tickers, period="1d", progress=False, auto_adjust=True)
                if data.empty:
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame()
            sector_cap = {}
            for t in all_tickers:
                try:
                    info = yf.Ticker(t).fast_info
                    mcap = getattr(info, "market_cap", 0) or 0
                    setor = ticker_sector.get(t, "Outros")
                    sector_cap[setor] = sector_cap.get(setor, 0) + mcap
                except Exception:
                    pass
            if sector_cap:
                total = sum(sector_cap.values())
                df = pd.DataFrame([
                    {"Setor": k, "Market Cap (R$ bi)": v / 1e9, "Peso (%)": (v / total) * 100}
                    for k, v in sorted(sector_cap.items(), key=lambda x: -x[1])
                ])
                return df
            return pd.DataFrame()

        with st.spinner("Calculando composição setorial..."):
            df_ibov = get_ibov_sector_weights()

        if not df_ibov.empty:
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=df_ibov["Setor"], values=df_ibov["Peso (%)"],
                    hole=0.4, textinfo="label+percent",
                    marker=dict(colors=["#2563eb", "#ea580c", "#16a34a", "#9333ea",
                                        "#dc2626", "#eab308", "#06b6d4", "#ec4899",
                                        "#84cc16", "#f97316"]),
                )])
                fig_pie.update_layout(
                    title=dict(text="IBOV — Concentração por Setor", font=dict(size=15, color="#111827")),
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", height=450,
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.dataframe(df_ibov, use_container_width=True, hide_index=True)
        else:
            st.caption("⚠️ Não foi possível calcular composição setorial")

    # ─── Tab Gráfico de Preço ────────────────────────────────────────────

    with tabs[len(WATCHLISTS) + 1]:
        st.markdown("### Gráfico de Preço Individual")

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            chart_ticker = st.text_input(
                "Ticker (ex: PETR4.SA, AAPL, BTC-USD, USDBRL=X)",
                value="PETR4.SA", key="price_chart_ticker",
            )
        with c2:
            period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1A": "1y", "2A": "2y", "5A": "5y", "10A": "10y", "Máx": "max"}
            period_label = st.selectbox("Período", list(period_map.keys()), index=3, key="price_period")
        with c3:
            chart_type = st.selectbox("Tipo", ["Linha", "Candlestick"], key="price_chart_type")

        if chart_ticker:
            @st.cache_data(ttl=3600, show_spinner=False)
            def get_price_data(ticker, period):
                try:
                    import yfinance as yf
                    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                    return data
                except Exception:
                    return pd.DataFrame()

            with st.spinner(f"Carregando {chart_ticker}..."):
                df_price = get_price_data(chart_ticker, period_map[period_label])

            if not df_price.empty:
                close = df_price["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()

                if len(close) > 1:
                    last = close.iloc[-1]
                    first = close.iloc[0]
                    var_pct = ((last - first) / first) * 100
                    high = close.max()
                    low = close.min()

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.metric("Último", f"{last:,.2f}", delta=f"{var_pct:+.2f}% no período")
                    with mc2:
                        st.metric("Máxima", f"{high:,.2f}")
                    with mc3:
                        st.metric("Mínima", f"{low:,.2f}")
                    with mc4:
                        st.metric("Variação", f"{var_pct:+.2f}%")

                fig_price = go.Figure()

                if chart_type == "Candlestick" and "Open" in df_price.columns:
                    open_col = df_price["Open"].iloc[:, 0] if isinstance(df_price["Open"], pd.DataFrame) else df_price["Open"]
                    high_col = df_price["High"].iloc[:, 0] if isinstance(df_price["High"], pd.DataFrame) else df_price["High"]
                    low_col = df_price["Low"].iloc[:, 0] if isinstance(df_price["Low"], pd.DataFrame) else df_price["Low"]
                    fig_price.add_trace(go.Candlestick(
                        x=df_price.index, open=open_col, high=high_col,
                        low=low_col, close=close, name=chart_ticker,
                    ))
                else:
                    fig_price.add_trace(go.Scatter(
                        x=close.index, y=close.values, mode="lines",
                        name=chart_ticker, line=dict(color="#2563eb", width=2),
                        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
                    ))
                    fig_price.add_annotation(
                        x=close.index[-1], y=close.iloc[-1],
                        text=f"<b>{close.iloc[-1]:,.2f}</b>",
                        showarrow=True, arrowhead=2, arrowcolor="#2563eb",
                        font=dict(size=12, color="#fff"), bgcolor="#2563eb",
                        ax=40, ay=-25,
                    )

                fig_price.update_layout(
                    title=dict(text=f"{chart_ticker} — {period_label}",
                               font=dict(size=16, color="#111827")),
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff", height=500,
                    xaxis=dict(gridcolor="#e5e7eb", rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor="#e5e7eb"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_price, use_container_width=True)

                if "Volume" in df_price.columns:
                    vol = df_price["Volume"]
                    if isinstance(vol, pd.DataFrame):
                        vol = vol.iloc[:, 0]
                    vol = vol.dropna()
                    if vol.sum() > 0:
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(
                            x=vol.index, y=vol.values,
                            marker_color="rgba(37,99,235,0.3)", name="Volume",
                        ))
                        fig_vol.update_layout(
                            title=dict(text="Volume", font=dict(size=13, color="#6b7280")),
                            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#ffffff", height=200,
                            margin=dict(l=50, r=50, t=30, b=20),
                            xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
            else:
                st.warning(f"Ticker '{chart_ticker}' não encontrado. Verifique o formato (BR: adicione .SA)")

    # ─── Tab Fluxo Estrangeiro ───────────────────────────────────────────

    with tabs[len(WATCHLISTS) + 2]:
        st.markdown("### Fluxo de Capital Estrangeiro B3")

        @st.cache_data(ttl=7200, show_spinner=False)
        def get_foreign_flows():
            try:
                from bcb import sgs
                df = sgs.get({"Fluxo Cambial Financeiro (US$ mi)": 22023}, start="2020-01-01")
                return df
            except Exception:
                return pd.DataFrame()

        df_flows = get_foreign_flows()
        if not df_flows.empty:
            col_name = df_flows.columns[0]
            serie = df_flows[col_name].dropna()
            serie_cum = serie.cumsum()

            fig_flow = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                      subplot_titles=["Fluxo Diário (US$ mi)", "Acumulado (US$ mi)"])
            fig_flow.add_trace(go.Bar(
                x=serie.index, y=serie.values,
                marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in serie.values],
                name="Diário", showlegend=False,
            ), row=1, col=1)
            fig_flow.add_trace(go.Scatter(
                x=serie_cum.index, y=serie_cum.values,
                mode="lines", line=dict(color="#2563eb", width=2),
                name="Acumulado", showlegend=False,
            ), row=2, col=1)
            fig_flow.add_hline(y=0, line_dash="dash", line_color="#6b7280", opacity=0.5, row=2, col=1)
            fig_flow.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff", height=600,
                xaxis2=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
                yaxis2=dict(gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig_flow, use_container_width=True)
        else:
            st.caption("⚠️ Fluxo cambial indisponível")

    # Download all
    st.markdown("---")
    if st.button("📥 Exportar tudo para CSV", key="export_mkt"):
        all_frames = []
        for wl_name, tickers in WATCHLISTS.items():
            df, _ = get_market_data(tickers)
            if not df.empty:
                df.insert(0, "Categoria", wl_name)
                all_frames.append(df)
        if all_frames:
            full = pd.concat(all_frames, ignore_index=True)
            csv = full.to_csv(index=False, sep=";")
            st.download_button("💾 Baixar", csv, "mercado_hoje.csv", "text/csv", key="dl_mkt")


# =============================================================================
# PAGE: ANÁLISE CVM
# =============================================================================

elif page == "📋 Análise CVM":

    st.markdown("# 📋 Análise de Empresas — Dados CVM")
    st.markdown("DFP (anual) e ITR (trimestral) direto do Portal de Dados Abertos da CVM")

    import zipfile, io
    from pathlib import Path

    CVM_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
    CVM_CACHE = Path("cvm_cache")
    CVM_ENCODING = "ISO-8859-1"

    CONTAS_CHAVE = {
        "3.01": "Receita Líquida", "3.02": "CPV", "3.03": "Resultado Bruto",
        "3.04": "Despesas Operacionais", "3.05": "EBIT",
        "3.06": "Resultado Financeiro", "3.06.01": "Receitas Financeiras",
        "3.06.02": "Despesas Financeiras", "3.07": "EBT",
        "3.08": "IR/CSLL", "3.09": "Lucro Líquido Op. Continuadas",
        "3.11": "Lucro/Prejuízo Consolidado",
        "1": "Ativo Total", "1.01": "Ativo Circulante",
        "1.01.01": "Caixa e Equivalentes", "1.01.02": "Aplicações Fin. CP",
        "1.01.03": "Contas a Receber", "1.01.04": "Estoques",
        "1.02": "Ativo Não Circulante", "1.02.03": "Imobilizado", "1.02.04": "Intangível",
        "2": "Passivo Total", "2.01": "Passivo Circulante",
        "2.01.04": "Empréstimos CP", "2.02": "Passivo Não Circulante",
        "2.02.01": "Empréstimos LP", "2.03": "Patrimônio Líquido",
        "6.01": "FCO", "6.02": "Caixa Investimentos", "6.03": "Caixa Financiamentos",
        "6.05": "Variação de Caixa",
    }

    @st.cache_data(ttl=86400, show_spinner=False)
    def cvm_download(doc_type, year):
        url = f"{CVM_BASE_URL}/{doc_type}/DADOS/{doc_type.lower()}_cia_aberta_{year}.zip"
        CVM_CACHE.mkdir(exist_ok=True)
        cache = CVM_CACHE / f"{doc_type.lower()}_{year}.zip"
        if cache.exists():
            return cache.read_bytes()
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        if r.status_code != 200:
            return None
        cache.write_bytes(r.content)
        return r.content

    def cvm_read_csv(zb, filename):
        with zipfile.ZipFile(io.BytesIO(zb)) as zf:
            with zf.open(filename) as f:
                return pd.read_csv(f, sep=";", encoding=CVM_ENCODING,
                                   dtype={"CD_CVM": str, "CNPJ_CIA": str, "CD_CONTA": str},
                                   on_bad_lines="skip")

    def cvm_load(doc_type, year, file_type, consolidado=True):
        zb = cvm_download(doc_type, year)
        if zb is None:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(zb)) as zf:
            names = zf.namelist()
        tc = "con" if consolidado else "ind"
        prefix = f"{doc_type.lower()}_cia_aberta_{file_type}_{tc}_{year}"
        target = next((f for f in names if prefix.lower() in f.lower()), None)
        if not target:
            target = next((f for f in names if file_type.lower() in f.lower() and tc in f.lower()), None)
        if not target:
            return pd.DataFrame()
        df = cvm_read_csv(zb, target)
        if "VL_CONTA" in df.columns:
            df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")
        for c in ["DT_REFER", "DT_INI_EXERC", "DT_FIM_EXERC"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        if "ESCALA_MOEDA" in df.columns:
            mask = df["ESCALA_MOEDA"] == "MIL"
            df.loc[mask, "VL_CONTA"] = df.loc[mask, "VL_CONTA"] * 1000
        return df

    def cvm_empresa(df, query):
        if query.isdigit():
            r = df[df["CD_CVM"] == query]
            if len(r) > 0: return r
        if "DENOM_CIA" in df.columns:
            r = df[df["DENOM_CIA"].str.contains(query, case=False, na=False)]
            if len(r) > 0: return r
        return pd.DataFrame()

    def cvm_indicadores(nome, year, doc_type="DFP"):
        indicadores = {}
        for ft in ["DRE", "BPA", "BPP", "DFC_MI"]:
            try:
                df = cvm_load(doc_type, year, ft)
                if df.empty: continue
                df = cvm_empresa(df, nome)
                if "ORDEM_EXERC" in df.columns:
                    df = df[df["ORDEM_EXERC"] == "ÚLTIMO"]
                if "VERSAO" in df.columns:
                    idx = df.groupby(["CD_CVM", "DT_REFER", "CD_CONTA"])["VERSAO"].idxmax()
                    df = df.loc[idx]
                df = df[df["CD_CONTA"].isin(CONTAS_CHAVE.keys())].copy()
                df["CONTA_DESCR"] = df["CD_CONTA"].map(CONTAS_CHAVE)
                for _, row in df.iterrows():
                    indicadores[row["CD_CONTA"]] = {
                        "Código": row["CD_CONTA"],
                        "Conta": row["CONTA_DESCR"],
                        "Valor": row["VL_CONTA"],
                        "Demo": ft,
                    }
            except Exception:
                pass
        return pd.DataFrame(indicadores.values()) if indicadores else pd.DataFrame()

    def cvm_metricas(ind):
        def v(cd):
            r = ind[ind["Código"] == cd]
            return r.iloc[0]["Valor"] if len(r) > 0 else None
        m = {}
        rec = v("3.01"); ebit = v("3.05"); lucro = v("3.11") or v("3.09")
        rb = v("3.03"); pl = v("2.03"); at = v("1")
        ac = v("1.01"); pc = v("2.01"); caixa = v("1.01.01"); aplic = v("1.01.02")
        emp_cp = v("2.01.04"); emp_lp = v("2.02.01"); fco = v("6.01")

        if rec and rec != 0:
            m["Receita Líquida (R$ mi)"] = rec / 1e6
            if rb: m["Margem Bruta (%)"] = (rb / rec) * 100
            if ebit: m["Margem EBIT (%)"] = (ebit / rec) * 100
            if lucro: m["Margem Líquida (%)"] = (lucro / rec) * 100
        if ebit: m["EBIT (R$ mi)"] = ebit / 1e6
        if lucro: m["Lucro Líquido (R$ mi)"] = lucro / 1e6
        if at: m["Ativo Total (R$ mi)"] = at / 1e6
        if pl: m["Patrimônio Líquido (R$ mi)"] = pl / 1e6

        db = None
        if emp_cp is not None and emp_lp is not None:
            db = abs(emp_cp) + abs(emp_lp)
            m["Dívida Bruta (R$ mi)"] = db / 1e6
        cx = (caixa or 0) + (aplic or 0)
        m["Caixa + Aplic. (R$ mi)"] = cx / 1e6
        if db is not None:
            m["Dívida Líquida (R$ mi)"] = (db - cx) / 1e6
        if ac and pc:
            m["Capital de Giro (R$ mi)"] = (ac - abs(pc)) / 1e6
            if pc != 0: m["Liquidez Corrente"] = ac / abs(pc)
        if lucro and pl and pl != 0: m["ROE (%)"] = (lucro / pl) * 100
        if lucro and at and at != 0: m["ROA (%)"] = (lucro / at) * 100
        if ebit and pl and db is not None:
            ci = pl + db
            if ci != 0: m["ROIC (%)"] = (ebit / ci) * 100
        if db is not None and pl and pl != 0: m["Dívida Bruta / PL"] = db / pl
        if fco: m["FCO (R$ mi)"] = fco / 1e6
        return m

    @st.cache_data(ttl=86400, show_spinner=False)
    def cvm_listar_empresas(year):
        df = cvm_load("DFP", year, "DRE")
        if df.empty: return pd.DataFrame()
        return df[["CD_CVM", "DENOM_CIA"]].drop_duplicates().sort_values("DENOM_CIA").reset_index(drop=True)

    # ─── Sidebar controls ────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("### 🔎 Empresa")
        empresa_input = st.text_input("Nome da empresa (parcial)", value="PETROBRAS", key="cvm_emp")
        doc_type = st.selectbox("Tipo", ["DFP", "ITR"], key="cvm_doc")
        ano_atual = datetime.now().year
        anos_disp = list(range(ano_atual, 2019, -1))
        ano_sel = st.selectbox("Ano", anos_disp, key="cvm_ano")
        st.markdown("### 📊 Série Histórica")
        anos_hist = st.multiselect("Anos", anos_disp, default=anos_disp[:5], key="cvm_hist_anos")

    # ─── Tabs ────────────────────────────────────────────────────────────

    tab_indicadores, tab_serie, tab_comparacao, tab_raw = st.tabs([
        "📊 Indicadores",
        "📈 Série Histórica",
        "⚖️ Comparação",
        "🗂️ Dados Brutos",
    ])

    with tab_indicadores:
        st.markdown(f"### {empresa_input} — {doc_type} {ano_sel}")

        with st.spinner(f"Carregando {doc_type} {ano_sel}..."):
            ind = cvm_indicadores(empresa_input, ano_sel, doc_type)

        if ind.empty:
            st.warning(f"Empresa '{empresa_input}' não encontrada no {doc_type} {ano_sel}.")
            st.caption("Tente o nome como aparece na CVM (ex: 'PETROBRAS', 'VALE', 'WEG')")
        else:
            metricas = cvm_metricas(ind)

            # Cards de métricas principais
            row1 = st.columns(4)
            keys_row1 = ["Receita Líquida (R$ mi)", "EBIT (R$ mi)", "Lucro Líquido (R$ mi)", "FCO (R$ mi)"]
            for i, k in enumerate(keys_row1):
                with row1[i]:
                    v = metricas.get(k)
                    st.metric(k.replace(" (R$ mi)", ""), f"R$ {v:,.0f} mi" if v else "—")

            row2 = st.columns(4)
            keys_row2 = ["Margem EBIT (%)", "ROE (%)", "ROIC (%)", "Dívida Bruta / PL"]
            for i, k in enumerate(keys_row2):
                with row2[i]:
                    v = metricas.get(k)
                    if v is not None:
                        fmt = f"{v:.1f}%" if "%" in k else f"{v:.2f}x"
                        st.metric(k.replace(" (%)", ""), fmt)
                    else:
                        st.metric(k, "—")

            row3 = st.columns(4)
            keys_row3 = ["Dívida Líquida (R$ mi)", "Caixa + Aplic. (R$ mi)", "Liquidez Corrente", "Capital de Giro (R$ mi)"]
            for i, k in enumerate(keys_row3):
                with row3[i]:
                    v = metricas.get(k)
                    if v is not None:
                        if "R$ mi" in k:
                            st.metric(k.replace(" (R$ mi)", ""), f"R$ {v:,.0f} mi")
                        else:
                            st.metric(k, f"{v:.2f}")
                    else:
                        st.metric(k, "—")

            # Tabela completa
            st.markdown("#### Todas as métricas")
            met_df = pd.DataFrame(list(metricas.items()), columns=["Métrica", "Valor"])
            met_df["Valor"] = met_df.apply(
                lambda r: f"{r['Valor']:.1f}%" if "%" in r["Métrica"]
                else f"{r['Valor']:,.0f}" if "R$ mi" in r["Métrica"]
                else f"{r['Valor']:.2f}", axis=1
            )
            st.dataframe(met_df, use_container_width=True, hide_index=True)

            # Indicadores brutos CVM
            with st.expander("Ver contas CVM detalhadas"):
                display_ind = ind.copy()
                display_ind["Valor (R$ mil)"] = display_ind["Valor"].apply(
                    lambda x: f"{x/1e3:,.0f}" if pd.notna(x) else "—"
                )
                st.dataframe(display_ind[["Código", "Conta", "Valor (R$ mil)", "Demo"]],
                             use_container_width=True, hide_index=True)

    with tab_serie:
        st.markdown(f"### Série Histórica — {empresa_input}")

        if not anos_hist:
            st.info("Selecione os anos no sidebar.")
        else:
            all_met = {}
            progress = st.progress(0)
            for i, ano in enumerate(sorted(anos_hist)):
                try:
                    ind = cvm_indicadores(empresa_input, ano, doc_type)
                    if not ind.empty:
                        all_met[ano] = cvm_metricas(ind)
                except Exception:
                    pass
                progress.progress((i + 1) / len(anos_hist))
            progress.empty()

            if all_met:
                serie_df = pd.DataFrame(all_met)
                serie_df.index.name = "Métrica"

                # Tabela formatada
                display_serie = serie_df.copy()
                for col in display_serie.columns:
                    display_serie[col] = display_serie[col].apply(
                        lambda x: f"{x:,.1f}" if pd.notna(x) else "—"
                    )
                st.dataframe(display_serie, use_container_width=True)

                # Helper: add Y-axis padding to charts
                def add_y_padding(fig):
                    """Adds 15% padding above max value to prevent text cutoff."""
                    all_y = []
                    for trace in fig.data:
                        if hasattr(trace, 'y') and trace.y is not None:
                            all_y.extend([v for v in trace.y if v is not None])
                    if all_y:
                        y_min = min(all_y)
                        y_max = max(all_y)
                        rng = y_max - y_min if y_max != y_min else abs(y_max) * 0.2
                        pad = rng * 0.2
                        fig.update_yaxes(range=[y_min - pad * 0.3, y_max + pad])

                # ─── Gráficos de LINHA: Margens ─────────────────────────────
                st.markdown("#### Evolução de Margens")
                margin_metrics = ["Margem Bruta (%)", "Margem EBIT (%)", "Margem Líquida (%)"]
                avail_margins = [m for m in margin_metrics if m in serie_df.index]

                if avail_margins:
                    fig_margins = go.Figure()
                    m_colors = ["#2563eb", "#ea580c", "#16a34a"]
                    for i, metric in enumerate(avail_margins):
                        row = serie_df.loc[metric].dropna()
                        if len(row) > 0:
                            fig_margins.add_trace(go.Scatter(
                                x=[str(y) for y in row.index], y=row.values,
                                mode="lines+markers+text", name=metric.replace(" (%)", ""),
                                line=dict(color=m_colors[i], width=2.5),
                                marker=dict(size=8),
                                text=[f"{v:.1f}%" for v in row.values],
                                textposition="top center", textfont=dict(size=10),
                            ))
                    add_y_padding(fig_margins)
                    fig_margins.update_layout(
                        title=dict(text="Margens (%)", font=dict(size=15, color="#111827")),
                        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#ffffff", height=420,
                        xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="%"),
                        hovermode="x unified", legend=dict(orientation="h", y=1.15),
                    )
                    st.plotly_chart(fig_margins, use_container_width=True)
                    st.caption("Margem Bruta = (Receita − CPV) / Receita  •  "
                               "Margem EBIT = EBIT / Receita  •  "
                               "Margem Líquida = Lucro Líquido / Receita")

                # ─── Gráficos de LINHA: Retornos ────────────────────────────
                st.markdown("#### Evolução de Retornos")
                return_metrics = ["ROE (%)", "ROIC (%)", "ROA (%)"]
                avail_returns = [m for m in return_metrics if m in serie_df.index]

                if avail_returns:
                    fig_returns = go.Figure()
                    r_colors = ["#9333ea", "#dc2626", "#eab308"]
                    for i, metric in enumerate(avail_returns):
                        row = serie_df.loc[metric].dropna()
                        if len(row) > 0:
                            fig_returns.add_trace(go.Scatter(
                                x=[str(y) for y in row.index], y=row.values,
                                mode="lines+markers+text", name=metric.replace(" (%)", ""),
                                line=dict(color=r_colors[i], width=2.5),
                                marker=dict(size=8),
                                text=[f"{v:.1f}%" for v in row.values],
                                textposition="top center", textfont=dict(size=10),
                            ))
                    add_y_padding(fig_returns)
                    fig_returns.update_layout(
                        title=dict(text="Retornos (%)", font=dict(size=15, color="#111827")),
                        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#ffffff", height=420,
                        xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="%"),
                        hovermode="x unified", legend=dict(orientation="h", y=1.15),
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                    st.caption("ROE = Lucro Líquido / Patrimônio Líquido  •  "
                               "ROIC = EBIT / (PL + Dívida Bruta)  •  "
                               "ROA = Lucro Líquido / Ativo Total")

                # ─── Gráfico de LINHA: Endividamento ────────────────────────
                st.markdown("#### Evolução do Endividamento")
                debt_metrics = ["Dívida Bruta (R$ mi)", "Dívida Líquida (R$ mi)",
                                "Caixa + Aplic. (R$ mi)", "Dívida Bruta / PL"]
                avail_debt = [m for m in debt_metrics if m in serie_df.index]

                if avail_debt:
                    abs_debt = [m for m in avail_debt if "R$ mi" in m]
                    ratio_debt = [m for m in avail_debt if "R$ mi" not in m]

                    if abs_debt:
                        fig_debt = go.Figure()
                        debt_colors = ["#dc2626", "#ea580c", "#16a34a"]
                        for i, metric in enumerate(abs_debt):
                            row = serie_df.loc[metric].dropna()
                            if len(row) > 0:
                                fig_debt.add_trace(go.Scatter(
                                    x=[str(y) for y in row.index], y=row.values,
                                    mode="lines+markers+text", name=metric.replace(" (R$ mi)", ""),
                                    line=dict(color=debt_colors[i % len(debt_colors)], width=2.5),
                                    marker=dict(size=8),
                                    text=[f"{v:,.0f}" for v in row.values],
                                    textposition="top center", textfont=dict(size=9),
                                ))
                        fig_debt.add_hline(y=0, line_dash="dash", line_color="#6b7280", opacity=0.5)
                        add_y_padding(fig_debt)
                        fig_debt.update_layout(
                            title=dict(text="Endividamento (R$ mi)", font=dict(size=15, color="#111827")),
                            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#ffffff", height=450,
                            xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="R$ milhões"),
                            hovermode="x unified", legend=dict(orientation="h", y=1.15),
                        )
                        st.plotly_chart(fig_debt, use_container_width=True)
                        st.caption("Dívida Bruta = Empréstimos CP + LP  •  "
                                   "Dívida Líquida = Dívida Bruta − Caixa − Aplicações Financeiras")

                    if ratio_debt:
                        fig_ratio = go.Figure()
                        for metric in ratio_debt:
                            row = serie_df.loc[metric].dropna()
                            if len(row) > 0:
                                fig_ratio.add_trace(go.Bar(
                                    x=[str(y) for y in row.index], y=row.values,
                                    name=metric, marker_color="#ea580c",
                                    text=[f"{v:.2f}x" for v in row.values], textposition="outside",
                                ))
                        add_y_padding(fig_ratio)
                        fig_ratio.update_layout(
                            title=dict(text="Alavancagem — Dívida Bruta / Patrimônio Líquido",
                                       font=dict(size=15, color="#111827")),
                            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#ffffff", height=380,
                            xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="x"),
                        )
                        st.plotly_chart(fig_ratio, use_container_width=True)
                        st.caption("Alavancagem = Dívida Bruta (Empréstimos CP + LP) / Patrimônio Líquido.  "
                                   "Valores > 1.0x indicam que a empresa deve mais do que seu patrimônio próprio.")

                # Gráficos de BARRA para valores absolutos (receita, EBIT, lucro, FCO)
                st.markdown("#### Receita, Lucro e Caixa")
                bar_metrics = ["Receita Líquida (R$ mi)", "EBIT (R$ mi)",
                               "Lucro Líquido (R$ mi)", "FCO (R$ mi)"]
                avail_bar = [m for m in bar_metrics if m in serie_df.index]
                sel_bar = st.multiselect("Métricas (barras)", avail_bar,
                                         default=avail_bar[:3], key="cvm_bar_sel")

                for metric in sel_bar:
                    row = serie_df.loc[metric].dropna()
                    if len(row) == 0:
                        continue
                    fig = go.Figure()
                    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in row.values]
                    fig.add_trace(go.Bar(
                        x=[str(y) for y in row.index], y=row.values,
                        marker_color=colors, text=[f"{v:,.0f}" for v in row.values],
                        textposition="outside",
                    ))
                    add_y_padding(fig)
                    fig.update_layout(
                        title=dict(text=metric, font=dict(size=15, color="#111827")),
                        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#ffffff", height=350,
                        xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = serie_df.to_csv(sep=";")
                st.download_button("📥 Baixar série histórica", csv,
                                    f"cvm_{empresa_input}_{doc_type}.csv", "text/csv", key="dl_cvm_serie")
            else:
                st.warning("Nenhum dado encontrado para os anos selecionados.")

    with tab_comparacao:
        st.markdown("### Comparação entre empresas")

        empresas_comp = st.text_input(
            "Empresas (separadas por vírgula)",
            value="PETROBRAS, VALE, WEG",
            key="cvm_comp_input",
        )
        nomes = [e.strip() for e in empresas_comp.split(",") if e.strip()]

        sub_tab_snap, sub_tab_evo = st.tabs(["📊 Snapshot (1 ano)", "📈 Evolução Multi-Ano"])

        with sub_tab_snap:
            ano_comp = st.selectbox("Ano", anos_disp, key="cvm_comp_ano")

            if st.button("Comparar", key="cvm_comp_btn"):
                comp_data = {}
                progress = st.progress(0)
                for i, nome in enumerate(nomes):
                    try:
                        ind = cvm_indicadores(nome, ano_comp, doc_type)
                        if not ind.empty:
                            comp_data[nome] = cvm_metricas(ind)
                    except Exception:
                        st.caption(f"⚠️ {nome}: não encontrado")
                    progress.progress((i + 1) / len(nomes))
                progress.empty()

                if comp_data:
                    comp_df = pd.DataFrame(comp_data)
                    comp_df.index.name = "Métrica"

                    display_comp = comp_df.copy()
                    for col in display_comp.columns:
                        display_comp[col] = display_comp[col].apply(
                            lambda x: f"{x:,.1f}" if pd.notna(x) else "—"
                        )
                    st.dataframe(display_comp, use_container_width=True)

                    comp_metrics = ["Margem Bruta (%)", "Margem EBIT (%)", "Margem Líquida (%)",
                                    "ROE (%)", "ROIC (%)", "Dívida Bruta / PL", "Liquidez Corrente"]
                    available = [m for m in comp_metrics if m in comp_df.index]

                    if available:
                        metric_sel = st.selectbox("Métrica para gráfico", available, key="cvm_comp_metric")
                        row = comp_df.loc[metric_sel].dropna()
                        fig = go.Figure()
                        bar_colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626"]
                        fig.add_trace(go.Bar(
                            x=list(row.index), y=list(row.values),
                            marker_color=bar_colors[:len(row)],
                            text=[f"{v:.1f}" for v in row.values], textposition="outside",
                        ))
                        # Linha de média
                        avg = row.mean()
                        fig.add_hline(y=avg, line_dash="dash", line_color="#6b7280",
                                      annotation_text=f"Média: {avg:.1f}", annotation_position="top left")
                        fig.update_layout(
                            title=dict(text=f"{metric_sel} — {ano_comp}", font=dict(size=15, color="#111827")),
                            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="#ffffff", height=400,
                            yaxis=dict(gridcolor="#e5e7eb"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    csv = comp_df.to_csv(sep=";")
                    st.download_button("📥 Baixar comparação", csv,
                                        f"cvm_comparacao_{ano_comp}.csv", "text/csv", key="dl_cvm_comp")
                else:
                    st.warning("Nenhuma empresa encontrada.")

        with sub_tab_evo:
            st.markdown("#### Evolução comparativa ao longo dos anos")
            st.caption("Compara ROE, ROIC, margens de múltiplas empresas ao longo do tempo")

            anos_evo = st.multiselect("Anos", anos_disp, default=anos_disp[:5], key="cvm_evo_anos")
            metric_evo = st.selectbox("Métrica", [
                "ROE (%)", "ROIC (%)", "Margem EBIT (%)", "Margem Líquida (%)",
                "Margem Bruta (%)", "Dívida Bruta / PL", "Receita Líquida (R$ mi)",
                "Lucro Líquido (R$ mi)", "Dívida Líquida (R$ mi)", "FCO (R$ mi)",
            ], key="cvm_evo_metric")

            if st.button("Gerar evolução", key="cvm_evo_btn") and nomes and anos_evo:
                evo_data = {}  # {empresa: {ano: valor}}
                progress = st.progress(0)
                total = len(nomes) * len(anos_evo)
                count = 0

                for nome in nomes:
                    evo_data[nome] = {}
                    for ano in sorted(anos_evo):
                        try:
                            ind = cvm_indicadores(nome, ano, doc_type)
                            if not ind.empty:
                                met = cvm_metricas(ind)
                                if metric_evo in met:
                                    evo_data[nome][ano] = met[metric_evo]
                        except Exception:
                            pass
                        count += 1
                        progress.progress(count / total)
                progress.empty()

                # Plotar
                fig_evo = go.Figure()
                evo_colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626",
                              "#eab308", "#06b6d4", "#ec4899"]

                has_data = False
                all_values = []
                for i, (nome, values) in enumerate(evo_data.items()):
                    if not values:
                        continue
                    has_data = True
                    anos_sorted = sorted(values.keys())
                    vals = [values[a] for a in anos_sorted]
                    all_values.extend(vals)
                    fig_evo.add_trace(go.Scatter(
                        x=[str(a) for a in anos_sorted], y=vals,
                        mode="lines+markers+text", name=nome,
                        line=dict(color=evo_colors[i % len(evo_colors)], width=2.5),
                        marker=dict(size=8),
                        text=[f"{v:.1f}" if "%" in metric_evo or "/" in metric_evo
                              else f"{v:,.0f}" for v in vals],
                        textposition="top center", textfont=dict(size=9),
                    ))

                if has_data:
                    # Média das empresas selecionadas
                    avg_by_year = {}
                    for nome, values in evo_data.items():
                        for ano, val in values.items():
                            avg_by_year.setdefault(ano, []).append(val)
                    avg_anos = sorted(avg_by_year.keys())
                    avg_vals = [sum(avg_by_year[a]) / len(avg_by_year[a]) for a in avg_anos]
                    fig_evo.add_trace(go.Scatter(
                        x=[str(a) for a in avg_anos], y=avg_vals,
                        mode="lines", name="Média (selecionadas)",
                        line=dict(color="#6b7280", width=2, dash="dash"),
                    ))

                    fig_evo.update_layout(
                        title=dict(text=f"{metric_evo} — Evolução Comparativa",
                                   font=dict(size=16, color="#111827")),
                        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#ffffff", height=500,
                        xaxis=dict(gridcolor="#e5e7eb"),
                        yaxis=dict(gridcolor="#e5e7eb",
                                   title="%" if "%" in metric_evo else "x" if "/" in metric_evo else "R$ mi"),
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.15),
                    )
                    st.plotly_chart(fig_evo, use_container_width=True)
                    st.caption("⚠️ A linha tracejada 'Média (selecionadas)' é a média aritmética das empresas que você digitou acima, "
                               "não a média do setor. Para comparação setorial, inclua as principais empresas do setor.")

                    # Tabela
                    evo_table = pd.DataFrame(evo_data).T
                    evo_table.index.name = "Empresa"
                    display_evo = evo_table.copy()
                    for col in display_evo.columns:
                        display_evo[col] = display_evo[col].apply(
                            lambda x: f"{x:,.1f}" if pd.notna(x) else "—"
                        )
                    st.dataframe(display_evo, use_container_width=True)
                else:
                    st.warning("Nenhum dado encontrado.")

    with tab_raw:
        st.markdown("### Dados brutos da CVM")

        col1, col2, col3 = st.columns(3)
        with col1:
            raw_type = st.selectbox("Demonstrativo", ["DRE", "BPA", "BPP", "DFC_MI", "DVA"], key="cvm_raw_type")
        with col2:
            raw_year = st.selectbox("Ano", anos_disp, key="cvm_raw_year")
        with col3:
            raw_cons = st.checkbox("Consolidado", value=True, key="cvm_raw_cons")

        if st.button("Carregar", key="cvm_raw_btn"):
            with st.spinner("Baixando..."):
                df_raw = cvm_load(doc_type, raw_year, raw_type, raw_cons)

            if df_raw.empty:
                st.warning("Arquivo não encontrado.")
            else:
                st.success(f"✅ {len(df_raw)} linhas carregadas")

                # Filtro por empresa
                raw_emp = st.text_input("Filtrar por empresa (opcional)", key="cvm_raw_emp")
                if raw_emp:
                    df_raw = cvm_empresa(df_raw, raw_emp)

                # Filtro por conta
                raw_conta = st.text_input("Filtrar por código de conta (ex: 3.01)", key="cvm_raw_conta")
                if raw_conta:
                    df_raw = df_raw[df_raw["CD_CONTA"] == raw_conta]

                st.dataframe(df_raw.head(500), use_container_width=True, hide_index=True)

                csv = df_raw.to_csv(index=False, sep=";")
                st.download_button("📥 Baixar dados brutos", csv,
                                    f"cvm_{raw_type}_{raw_year}.csv", "text/csv", key="dl_cvm_raw")

        # Listar empresas
        with st.expander("📋 Listar todas as empresas disponíveis"):
            emp_year = st.selectbox("Ano", anos_disp, key="cvm_emp_list_year")
            if st.button("Listar", key="cvm_emp_list_btn"):
                with st.spinner("Carregando..."):
                    empresas_list = cvm_listar_empresas(emp_year)
                if not empresas_list.empty:
                    st.success(f"✅ {len(empresas_list)} empresas")
                    st.dataframe(empresas_list, use_container_width=True, hide_index=True, height=400)
                else:
                    st.warning("Dados não disponíveis para este ano.")


# =============================================================================
# PAGE: CRE LENDING
# =============================================================================

elif page == "🏢 CRE Lending":

    st.markdown("# 🏢 CRE Lending Dashboard")
    st.markdown("Dados macro e de crédito imobiliário essenciais para gestores e analistas")

    # ─── FRED Data (via fredapi or manual URL) ───────────────────────────

    FRED_CRE_SERIES = {
        # Delinquency & Credit Quality
        "CRE Delinquency Rate (%)": "DRCRELEXFACBS",
        "Residential Mortgage Delinquency (%)": "DRSFRMACBS",
        "All Loans Delinquency (%)": "DRALACBS",
        "CRE Charge-Off Rate (%)": "CORERELEXFACBS",

        # Lending Volume & Rates
        "CRE Loans Outstanding ($B)": "CREACBM027NBOG",
        "30Y Mortgage Rate (%)": "MORTGAGE30US",
        "10Y Treasury (%)": "DGS10",
        "2Y Treasury (%)": "DGS2",
        "Fed Funds Rate (%)": "FEDFUNDS",
        "BAA Corporate Spread (%)": "BAA10Y",

        # Real Estate Market
        "Case-Shiller US Home Price Index": "CSUSHPINSA",
        "Housing Starts (thousands)": "HOUST",
        "Building Permits (thousands)": "PERMIT",
        "CPI Shelter (%)": "CUSR0000SAH1",

        # Economic Context
        "Real GDP Growth (%)": "A191RL1Q225SBEA",
        "Unemployment Rate (%)": "UNRATE",
        "CPI All Items (%)": "CPIAUCSL",
    }

    @st.cache_data(ttl=7200, show_spinner=False)
    def get_fred_series(series_dict, start="2010-01-01"):
        """Fetch FRED series. Tries pandas_datareader first, then CSV fallback."""
        frames = {}
        errors = []

        # Method 1: pandas_datareader (most reliable)
        try:
            import pandas_datareader.data as web
            from datetime import datetime as dt
            start_dt = dt.strptime(start, "%Y-%m-%d")
            for name, code in series_dict.items():
                try:
                    df = web.DataReader(code, "fred", start_dt)
                    frames[name] = df.iloc[:, 0]
                except Exception as e:
                    errors.append(f"{name}: {e}")
            if frames:
                return pd.DataFrame(frames), errors
        except ImportError:
            errors.append("pandas_datareader não instalado, tentando CSV...")

        # Method 2: FRED CSV (fallback)
        for name, code in series_dict.items():
            if name in frames:
                continue
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}"
                headers = {"User-Agent": "Mozilla/5.0"}
                df = pd.read_csv(url, parse_dates=[0], index_col=0, storage_options={"User-Agent": "Mozilla/5.0"} if hasattr(pd, 'read_csv') else {})
                df.columns = [name]
                df[name] = pd.to_numeric(df[name], errors="coerce")
                frames[name] = df[name].dropna()
            except Exception as e:
                errors.append(f"{name} (CSV): {e}")

        # Method 3: FRED text files (last resort)
        for name, code in series_dict.items():
            if name in frames:
                continue
            try:
                url = f"https://fred.stlouisfed.org/data/{code}.txt"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                lines = resp.text.strip().split("\n")
                # Find where data starts (after header lines)
                data_start = 0
                for i, line in enumerate(lines):
                    if line.strip() and line[0].isdigit():
                        data_start = i
                        break
                data_lines = lines[data_start:]
                records = []
                for line in data_lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            date = pd.to_datetime(parts[0])
                            val = float(parts[1]) if parts[1] != "." else float("nan")
                            records.append((date, val))
                        except (ValueError, IndexError):
                            pass
                if records:
                    s = pd.Series(
                        [r[1] for r in records],
                        index=pd.DatetimeIndex([r[0] for r in records]),
                        name=name,
                    )
                    s = s[s.index >= start]
                    frames[name] = s.dropna()
            except Exception as e:
                errors.append(f"{name} (TXT): {e}")

        return pd.DataFrame(frames) if frames else pd.DataFrame(), errors

    with st.sidebar:
        st.markdown("### 📅 Período CRE")
        cre_start = st.slider("Ano inicial", 2005, 2025, 2015, key="cre_start")

    # Tabs internas
    tab_credit, tab_rates, tab_market, tab_resources = st.tabs([
        "📊 Crédito & Delinquência",
        "📈 Taxas & Spreads",
        "🏠 Mercado Imobiliário",
        "📚 Recursos & Referências",
    ])

    with st.spinner("Carregando dados do FRED..."):
        result = get_fred_series(FRED_CRE_SERIES, f"{cre_start}-01-01")
        if isinstance(result, tuple):
            df_cre, cre_errors = result
        else:
            df_cre, cre_errors = result, []

    if df_cre.empty:
        st.error("Não foi possível carregar dados do FRED.")
        st.caption("Instale pandas-datareader: `pip install pandas-datareader`")
        if cre_errors:
            with st.expander("Ver erros"):
                for e in cre_errors[:10]:
                    st.text(e)
    else:
        st.success(f"✅ {len(df_cre.columns)} séries carregadas do FRED")

    def cre_chart(col, title, color="#2563eb"):
        if col not in df_cre.columns:
            return None
        serie = df_cre[col].dropna()
        if len(serie) == 0:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.values, mode="lines",
            name=col, line=dict(color=color, width=2.5),
            hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra></extra>",
        ))
        last = serie.iloc[-1]
        fig.add_annotation(
            x=serie.index[-1], y=last,
            text=f"<b>{last:.2f}</b>", showarrow=True, arrowhead=2,
            arrowcolor=color, font=dict(size=12, color="#fff"),
            bgcolor=color, bordercolor=color, ax=40, ay=-25,
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=15, color="#111827")),
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff", height=380,
            margin=dict(l=50, r=50, t=50, b=40), hovermode="x unified",
            xaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db"),
            yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db"),
        )
        return fig

    with tab_credit:
        st.markdown("### Qualidade de Crédito CRE")

        if df_cre.empty:
            st.info("⏳ Dados não carregados. Instale: `pip install pandas-datareader`")
        else:
            # Métricas
            metrics_credit = [
                ("CRE Delinquency Rate (%)", "CRE Delinq."),
                ("Residential Mortgage Delinquency (%)", "Resid. Delinq."),
                ("All Loans Delinquency (%)", "All Loans Delinq."),
            ]
            cols_m = st.columns(len(metrics_credit))
            for i, (col_name, label) in enumerate(metrics_credit):
                if col_name in df_cre.columns:
                    s = df_cre[col_name].dropna()
                    if len(s) > 1:
                        val, prev = s.iloc[-1], s.iloc[-2]
                        delta = f"{val - prev:+.2f} pp"
                        with cols_m[i]:
                            st.metric(label, f"{val:.2f}%", delta=delta)

            for col_name in ["CRE Delinquency Rate (%)", "CRE Charge-Off Rate (%)", "CRE Loans Outstanding ($B)"]:
                fig = cre_chart(col_name, col_name, "#dc2626" if "Delinq" in col_name or "Charge" in col_name else "#2563eb")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with tab_rates:
        st.markdown("### Taxas de Juros & Spreads")

        if df_cre.empty:
            st.info("⏳ Dados não carregados. Instale: `pip install pandas-datareader`")
        else:
            metrics_rates = [
                ("30Y Mortgage Rate (%)", "Mortgage 30Y"),
                ("10Y Treasury (%)", "UST 10Y"),
                ("Fed Funds Rate (%)", "Fed Funds"),
                ("BAA Corporate Spread (%)", "BAA Spread"),
            ]
            cols_r = st.columns(len(metrics_rates))
            for i, (col_name, label) in enumerate(metrics_rates):
                if col_name in df_cre.columns:
                    s = df_cre[col_name].dropna()
                    if len(s) > 0:
                        with cols_r[i]:
                            st.metric(label, f"{s.iloc[-1]:.2f}%")

            # Yield curve proxy
            if "10Y Treasury (%)" in df_cre.columns and "2Y Treasury (%)" in df_cre.columns:
                spread = (df_cre["10Y Treasury (%)"] - df_cre["2Y Treasury (%)"]).dropna()
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=spread.index, y=spread.values, mode="lines",
                    fill="tozeroy", line=dict(color="#2563eb", width=2),
                    fillcolor="rgba(37,99,235,0.1)",
                    hovertemplate="%{x|%b %Y}<br>Spread: %{y:.2f}%<extra></extra>",
                ))
                fig_spread.add_hline(y=0, line_dash="dash", line_color="#dc2626", opacity=0.7)
                last_sp = spread.iloc[-1]
                fig_spread.add_annotation(
                    x=spread.index[-1], y=last_sp,
                    text=f"<b>{last_sp:.2f}%</b>", showarrow=True, arrowhead=2,
                    arrowcolor="#2563eb", font=dict(size=12, color="#fff"),
                    bgcolor="#2563eb", ax=40, ay=-25,
                )
                fig_spread.update_layout(
                    title=dict(text="Yield Curve Spread (10Y - 2Y)", font=dict(size=15, color="#111827")),
                    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff", height=380,
                    margin=dict(l=50, r=50, t=50, b=40),
                    xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", title="Spread (%)"),
                )
                st.plotly_chart(fig_spread, use_container_width=True)

            for col_name in ["30Y Mortgage Rate (%)", "10Y Treasury (%)", "BAA Corporate Spread (%)"]:
                fig = cre_chart(col_name, col_name, "#ea580c" if "Spread" in col_name else "#2563eb")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with tab_market:
        st.markdown("### Indicadores do Mercado Imobiliário")

        if df_cre.empty:
            st.info("⏳ Dados não carregados. Instale: `pip install pandas-datareader`")
        else:
            for col_name in ["Case-Shiller US Home Price Index", "Housing Starts (thousands)", "Building Permits (thousands)", "CPI Shelter (%)"]:
                fig = cre_chart(col_name, col_name, "#16a34a")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    with tab_resources:
        st.markdown("### 📚 Recursos para CRE Lending")

        st.markdown("""
        #### 🔗 Fontes de Dados Gratuitas

        **Macro & Crédito (séries FRED usadas acima)**
        - [FRED - CRE Series](https://fred.stlouisfed.org/tags/series?t=commercial%3Breal+estate) — 277 séries de commercial real estate
        - [Fed Charge-Off & Delinquency Rates](https://www.federalreserve.gov/releases/chargeoff/) — dados trimestrais por tipo de loan
        - [MBA CREF Research](https://www.mba.org/news-and-research/research-and-economics/commercial-multifamily-research) — delinquency reports por capital source

        **Mercado & Transações**
        - [NCREIF](https://www.ncreif.org/) — property index, cap rates por tipo de ativo
        - [Real Capital Analytics / MSCI](https://www.msci.com/real-capital-analytics) — transaction volume, pricing trends
        - [CoStar / LoopNet](https://www.costar.com/) — listagens e analytics (parcialmente grátis)
        - [Reonomy](https://www.reonomy.com/) — ownership, debt, property data

        **CMBS & Structured**
        - [CREFC](https://www.crefc.org/) — CMBS market standards, Investor Reporting Package
        - [Trepp](https://www.trepp.com/) — CMBS delinquency, surveillance (reports grátis)
        - [KBRA](https://www.kbra.com/) — ratings e research de CMBS

        **Research & Análise**
        - [Green Street](https://www.greenstreet.com/) — setor analysis, cap rate forecasts
        - [CBRE Research](https://www.cbre.com/insights) — market outlook, cap rate surveys
        - [JLL Research](https://www.jll.com/en/trends-and-insights) — office, industrial, retail trends
        - [Cushman & Wakefield](https://www.cushmanwakefield.com/en/insights) — market reports

        ---

        #### 📰 News Feeds Essenciais para CRE

        - [Commercial Observer](https://commercialobserver.com/) — deals, lending, NYC/national
        - [The Real Deal](https://therealdeal.com/) — transactions, development, finance
        - [Bisnow](https://www.bisnow.com/) — CRE news por mercado e setor
        - [CRE Daily](https://www.credaily.com/) — newsletter diária CRE
        - [GlobeSt](https://www.globest.com/) — CRE finance, investment, development
        - [Mortgage Bankers Association](https://www.mba.org/news-and-research/newsroom) — policy, origination data

        ---

        #### 🐍 Repositórios & Tools

        - [`fredapi`](https://github.com/mortada/fredapi) — Python wrapper para FRED API (precisa de key gratuita)
        - [`pandas-datareader`](https://github.com/pydata/pandas-datareader) — pull direto do FRED, World Bank, etc.
        - [`OpenBB`](https://github.com/OpenBB-finance/OpenBB) — inclui módulos de economia e real estate
        - [ATTOM API](https://www.attomdata.com/) — property data API (freemium)
        - [Zillow API / ZTRAX](https://www.zillow.com/research/) — research data downloads grátis

        ---

        #### 📖 Leituras Chave

        - *"Real Estate Finance and Investments"* — Brueggeman & Fisher (textbook padrão)
        - *"Commercial Real Estate Analysis and Investments"* — Geltner, Miller et al.
        - *"The Handbook of Commercial Mortgage-Backed Securities"* — Fabozzi & Jacob
        - Fed Financial Stability Reports — seção de CRE sempre relevante
        - FDIC Quarterly Banking Profile — exposição bancária a CRE
        """)

    # ─── CRE News ────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 📰 Notícias CRE")

    CRE_RSS = {
        "Commercial Observer": "https://commercialobserver.com/feed/",
        "Multi-Housing News": "https://www.multihousingnews.com/feed/",
        "Bisnow National": "https://www.bisnow.com/feed/national/news",
        "HousingWire": "https://www.housingwire.com/feed/",
    }

    cre_news_tabs = st.tabs(list(CRE_RSS.keys()))
    for tab, (source, url) in zip(cre_news_tabs, CRE_RSS.items()):
        with tab:
            items = fetch_rss_feed(url, max_items=12)
            if items:
                for item in items:
                    title = item["title"]
                    link = item["link"]
                    pub = item["published"][:25] if item["published"] else ""
                    summ = item["summary"][:150] + "..." if len(item["summary"]) > 150 else item["summary"]
                    st.markdown(
                        f'<div class="news-card">'
                        f'<a href="{link}" target="_blank" class="news-link">{title}</a>'
                        f'<p class="news-date">{pub}</p>'
                        f'<p class="news-summary">{summ}</p></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(f"⚠️ Feed indisponível")


# =============================================================================
# PAGE: NOTÍCIAS
# =============================================================================

elif page == "📰 Notícias":

    st.markdown("# 📰 Notícias do Mercado")
    st.markdown("Feeds RSS atualizados a cada 15 min")

    def render_news_items(items):
        """Renderiza lista de notícias."""
        if not items:
            st.caption("⚠️ Feed indisponível ou vazio")
            return
        for item in items:
            title = item["title"]
            link = item["link"]
            pub = item["published"][:25] if item["published"] else ""
            summ = item["summary"][:150] + "..." if len(item["summary"]) > 150 else item["summary"]
            st.markdown(
                f'<div class="news-card">'
                f'<a href="{link}" target="_blank" class="news-link">{title}</a>'
                f'<p class="news-date">{pub}</p>'
                f'<p class="news-summary">{summ}</p></div>',
                unsafe_allow_html=True,
            )

    for region, sources in RSS_FEEDS_SECTIONED.items():
        st.markdown(f"## {region}")

        # Tabs por fonte
        source_tabs = st.tabs(list(sources.keys()))

        for src_tab, (source_name, sections) in zip(source_tabs, sources.items()):
            with src_tab:
                if len(sections) == 1:
                    # Fonte sem seções — mostra direto
                    url = list(sections.values())[0]
                    items = fetch_rss_feed(url, max_items=15)
                    render_news_items(items)
                else:
                    # Fonte com seções — tabs internas
                    section_tabs = st.tabs(list(sections.keys()))
                    for sec_tab, (sec_name, sec_url) in zip(section_tabs, sections.items()):
                        with sec_tab:
                            items = fetch_rss_feed(sec_url, max_items=12)
                            render_news_items(items)

        st.markdown("---")

    if st.button("🔄 Atualizar notícias"):
        st.cache_data.clear()
        st.rerun()
