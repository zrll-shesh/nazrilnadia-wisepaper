import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(
    page_title="IPM Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #F7F5F0; }

section[data-testid="stSidebar"] { background-color: #0F1419 !important; }
section[data-testid="stSidebar"] * { color: #C8BFB0 !important; }
section[data-testid="stSidebar"] .stRadio > label { display: none; }
section[data-testid="stSidebar"] .stRadio label span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: #8A8070 !important;
    letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) span {
    color: #F5EFE6 !important;
    font-weight: 600 !important;
}

.brand { font-family: 'DM Serif Display', serif; font-size: 1.3rem;
         color: #F5EFE6; line-height: 1.2; }
.brand-sub { font-family: 'DM Mono', monospace; font-size: 0.6rem;
             color: #4A4440; letter-spacing: 0.14em; text-transform: uppercase;
             margin-top: 0.2rem; margin-bottom: 1.4rem; }
.sdiv { border-top: 1px solid #1C2026; margin: 0.9rem 0; }
.slabel { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: #3A3830;
          letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.4rem;
          margin-top: 0.8rem; }
.smrow { display: flex; justify-content: space-between; padding: 0.28rem 0;
         border-bottom: 1px solid #161A1E; }
.smlabel { font-size: 0.68rem; color: #5A5448; font-family: 'DM Sans', sans-serif; }
.smval { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #BCA882; font-weight: 500; }

.ph { padding: 1.6rem 0 0.9rem 0; border-bottom: 2px solid #0F1419; margin-bottom: 1.8rem; }
.pt { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #0F1419;
      letter-spacing: -0.02em; line-height: 1.1; }
.pt em { font-style: italic; color: #8B7355; }
.pc { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #9B8E7E;
      letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.4rem; }

.kcard { background: #fff; border: 1px solid #E5DDD0; border-radius: 2px;
         padding: 1.1rem 1.3rem; position: relative; overflow: hidden; }
.kcard::before { content: ''; position: absolute; top: 0; left: 0;
                 width: 3px; height: 100%; background: #0F1419; }
.klbl { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: #9B8E7E;
        letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.35rem; }
.kval { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #0F1419;
        line-height: 1; letter-spacing: -0.02em; }
.ksub { font-size: 0.7rem; color: #9B8E7E; margin-top: 0.25rem; }
.kdpos { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #2D6A4F; margin-top: 0.15rem; }
.kdneg { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #9B1D20; margin-top: 0.15rem; }

.sec { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: #9B8E7E;
       letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.5rem;
       margin-top: 1.6rem; padding-bottom: 0.35rem; border-bottom: 1px solid #E5DDD0; }

.prow { display: flex; justify-content: space-between; align-items: center;
        padding: 0.4rem 0; border-bottom: 1px solid #F0E8DE; }
.prow:last-child { border-bottom: none; }
.pname { font-size: 0.8rem; color: #2A2420; font-weight: 500; font-family: 'DM Sans', sans-serif; }
.pval  { font-family: 'DM Mono', monospace; font-size: 0.73rem; color: #6B5F50; }

.ib  { border-left: 3px solid #0F1419; padding: 0.75rem 1rem; background: #fff;
       margin-bottom: 0.55rem; border-radius: 0 2px 2px 0; }
.ibg { border-left-color: #2D6A4F !important; }
.ibr { border-left-color: #9B1D20 !important; }
.iba { border-left-color: #92580A !important; }
.ibt { font-family: 'DM Mono', monospace; font-size: 0.58rem; letter-spacing: 0.1em;
       text-transform: uppercase; color: #6B5F50; margin-bottom: 0.25rem; }
.ibb { font-family: 'DM Sans', sans-serif; font-size: 0.8rem; color: #2A2420; line-height: 1.55; }

.strow { background: #F0EBE3; padding: 0.6rem 0.9rem; border-radius: 2px;
         margin-bottom: 0.35rem; display: flex; justify-content: space-between; }
.stlbl { font-size: 0.77rem; color: #4A4440; font-family: 'DM Sans', sans-serif; }
.stval { font-family: 'DM Mono', monospace; font-size: 0.77rem; color: #0F1419; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all():
    df_cl  = pd.read_csv('hasil_clustering.csv')
    df_fc  = pd.read_csv('hasil_forecast.csv')
    df_fe  = pd.read_csv('hasil_fe_coef.csv', index_col=0)
    df_sh  = pd.read_csv('hasil_shap.csv')
    df_gr  = pd.read_csv('hasil_growth.csv')
    df_raw = pd.read_csv('data_bps_provinsi.csv')
    df_raw.columns = [
        'provinsi','tahun','aps_1315','aps_1618','aps_1924',
        'tpt_feb','tpt_agt','tpak_feb','tpak_agt',
        'gk_maret','gk_sept','miskin_maret','miskin_sept',
        'pct_miskin_maret','pct_miskin_sept','ipm','rls','hls'
    ]
    for c in df_raw.columns[2:]:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
    with open('meta_analisis.json') as f:
        meta = json.load(f)
    return df_cl, df_fc, df_fe, df_sh, df_gr, df_raw, meta

df_cl, df_fc, df_fe, df_sh, df_gr, df_raw, meta = load_all()

FEATURES = meta['features']
FL       = meta['feature_labels']
VL       = meta['var_labels']
RAW_COLORS = meta['cluster_colors']          # {'0': '#hex', '1': '#hex', ...}
CL_COLORS  = {int(k): v for k, v in RAW_COLORS.items()}

PROV_COORDS = {
    'ACEH':(4.7,96.7),'SUMATERA UTARA':(2.1,99.3),'SUMATERA BARAT':(-0.8,100.4),
    'RIAU':(0.3,101.5),'JAMBI':(-1.6,103.6),'SUMATERA SELATAN':(-3.3,104.2),
    'BENGKULU':(-3.8,102.3),'LAMPUNG':(-4.6,105.6),'KEP. BANGKA BELITUNG':(-2.7,106.8),
    'KEP. RIAU':(3.9,108.1),'DKI JAKARTA':(-6.2,106.8),'JAWA BARAT':(-7.1,107.6),
    'JAWA TENGAH':(-7.2,110.4),'DI YOGYAKARTA':(-7.8,110.4),'JAWA TIMUR':(-7.5,112.2),
    'BANTEN':(-6.4,106.1),'BALI':(-8.4,115.2),'NUSA TENGGARA BARAT':(-8.7,117.4),
    'NUSA TENGGARA TIMUR':(-8.7,121.1),'KALIMANTAN BARAT':(-0.1,110.5),
    'KALIMANTAN TENGAH':(-1.7,113.9),'KALIMANTAN SELATAN':(-3.1,115.3),
    'KALIMANTAN TIMUR':(0.5,116.4),'KALIMANTAN UTARA':(3.1,116.0),
    'SULAWESI UTARA':(1.5,124.8),'SULAWESI TENGAH':(-1.4,121.4),
    'SULAWESI SELATAN':(-3.7,120.0),'SULAWESI TENGGARA':(-4.1,122.5),
    'GORONTALO':(0.7,122.5),'SULAWESI BARAT':(-2.8,119.2),
    'MALUKU':(-3.2,130.1),'MALUKU UTARA':(1.6,127.9),
    'PAPUA BARAT':(-1.3,133.2),'PAPUA BARAT DAYA':(-1.9,132.0),
    'PAPUA':(-4.3,138.1),'PAPUA SELATAN':(-7.2,139.6),
    'PAPUA TENGAH':(-3.6,135.5),'PAPUA PEGUNUNGAN':(-4.1,139.7)
}


def cl_color(cl_id):
    return CL_COLORS.get(int(cl_id), '#9B8E7E')


def base_layout(**extra):
    d = dict(
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
        font=dict(family='DM Sans, sans-serif', color='#2A2420', size=11),
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(showgrid=True, gridcolor='#F0E8DE', linecolor='#E5DDD0',
                   tickfont=dict(size=10), title_font=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor='#F0E8DE', linecolor='#E5DDD0',
                   tickfont=dict(size=10), title_font=dict(size=11)),
        hoverlabel=dict(bgcolor='#0F1419', font_color='#F5EFE6',
                        font_family='DM Mono, monospace', font_size=11,
                        bordercolor='#0F1419')
    )
    d.update(extra)
    return d


def header(title_html, caption):
    st.markdown(
        f'<div class="ph"><div class="pt">{title_html}</div>'
        f'<div class="pc">{caption}</div></div>',
        unsafe_allow_html=True
    )


def kpi(label, value, sub=None, delta=None):
    delta_html = ''
    if delta is not None:
        cls  = 'kdpos' if delta >= 0 else 'kdneg'
        sign = '+' if delta >= 0 else ''
        delta_html = f'<div class="{cls}">{sign}{delta:.2f}</div>'
    sub_html = f'<div class="ksub">{sub}</div>' if sub else ''
    st.markdown(
        f'<div class="kcard"><div class="klbl">{label}</div>'
        f'<div class="kval">{value}</div>{sub_html}{delta_html}</div>',
        unsafe_allow_html=True
    )


def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)


def ib(title, body, variant=''):
    cls = {'green': 'ibg', 'red': 'ibr', 'amber': 'iba'}.get(variant, '')
    st.markdown(
        f'<div class="ib {cls}"><div class="ibt">{title}</div>'
        f'<div class="ibb">{body}</div></div>',
        unsafe_allow_html=True
    )


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Analisis IPM<br>Provinsi Indonesia</div>'
                '<div class="brand-sub">2021 – 2027 &nbsp;|&nbsp; 38 Provinsi</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)

    page = st.radio("nav", [
        "Ringkasan Eksekutif", "Clustering", "Peta Spasial",
        "Regresi Panel", "SHAP Analysis", "Forecasting",
        "Paradoks & Underdog", "Insight Gabungan"
    ], label_visibility="collapsed")

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Performa Model</div>', unsafe_allow_html=True)
    for lbl, val in [
        ("XGB R2 (CV)",      f"{meta['xgb_cv_r2_mean']:.4f}"),
        ("XGB MAE (CV)",     f"{meta['xgb_cv_mae_mean']:.4f}"),
        ("FE R2 Within",     f"{meta['fe_r2_within']:.4f}"),
        ("Silhouette GMM",   f"{meta['sil_score']:.4f}"),
        ("Davies-Bouldin",   f"{meta['db_score']:.4f}"),
        ("K Cluster",        str(meta['optimal_k'])),
    ]:
        st.markdown(
            f'<div class="smrow"><span class="smlabel">{lbl}</span>'
            f'<span class="smval">{val}</span></div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════
if page == "Ringkasan Eksekutif":
    header("Indeks Pembangunan Manusia<br><em>Provinsi Indonesia</em>",
           "BPS 2021–2024 | Proyeksi 2025–2027 | UMAP+GMM | XGBoost | SHAP | Fixed Effect")

    avg24    = df_cl['ipm'].mean()
    max24    = df_cl['ipm'].max()
    min24    = df_cl['ipm'].min()
    avg27    = df_fc[df_fc['tahun'] == 2027]['ipm_forecast'].mean()
    pmax     = df_cl.loc[df_cl['ipm'].idxmax(), 'provinsi'].title()
    pmin     = df_cl.loc[df_cl['ipm'].idxmin(), 'provinsi'].title()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Rerata IPM 2024",  f"{avg24:.2f}")
    with c2: kpi("IPM Tertinggi",    f"{max24:.2f}", sub=pmax)
    with c3: kpi("IPM Terendah",     f"{min24:.2f}", sub=pmin)
    with c4: kpi("Kesenjangan",      f"{max24-min24:.2f}", sub="maks minus min")
    with c5: kpi("Proyeksi 2027",    f"{avg27:.2f}", delta=avg27 - avg24)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        sec("Distribusi IPM 2024 — Seluruh Provinsi")
        ds = df_cl.sort_values('ipm', ascending=True)
        fig = go.Figure(go.Bar(
            x=ds['ipm'], y=ds['provinsi'].str.title(), orientation='h',
            marker=dict(color=[cl_color(c) for c in ds['cluster_id']], line=dict(width=0)),
            text=ds['ipm'].round(2), textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=9, color='#6B5F50'),
            hovertemplate='<b>%{y}</b><br>IPM: %{x:.2f}<extra></extra>'
        ))
        fig.update_layout(**base_layout(height=650, xaxis_range=[58, 87],
                                        xaxis_title='IPM 2024',
                                        margin=dict(l=0, r=60, t=10, b=10)))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sec("Tren IPM Nasional 2021–2024")
        tren = df_raw.groupby('tahun')['ipm'].agg(['mean','min','max']).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tren['tahun'], y=tren['max'], name='Maks',
                                  line=dict(color='#C8B89A', width=1.5, dash='dot')))
        fig2.add_trace(go.Scatter(x=tren['tahun'], y=tren['mean'], name='Rata-rata',
                                  line=dict(color='#0F1419', width=2.5),
                                  mode='lines+markers', marker=dict(size=7, color='#0F1419')))
        fig2.add_trace(go.Scatter(x=tren['tahun'], y=tren['min'], name='Min',
                                  line=dict(color='#9B1D20', width=1.5, dash='dot')))
        fig2.update_layout(**base_layout(height=200,
                                         legend=dict(orientation='h', y=-0.35, font_size=10),
                                         margin=dict(l=0, r=0, t=10, b=50)))
        st.plotly_chart(fig2, use_container_width=True)

        sec("Komposisi Cluster 2024")
        cc = df_cl['cluster_label'].value_counts().reset_index()
        cc.columns = ['cluster', 'n']
        fig3 = go.Figure(go.Pie(
            labels=cc['cluster'], values=cc['n'], hole=0.52,
            marker=dict(colors=[cl_color(i) for i in range(len(cc))],
                        line=dict(color='#F7F5F0', width=3)),
            textfont=dict(family='DM Mono, monospace', size=10),
            hovertemplate='%{label}<br>%{value} provinsi<extra></extra>'
        ))
        fig3.update_layout(**base_layout(height=200, showlegend=True,
                                          legend=dict(orientation='h', y=-0.25, font_size=9),
                                          margin=dict(l=0, r=0, t=10, b=50)))
        st.plotly_chart(fig3, use_container_width=True)

        sec("Variabel Signifikan (FEM)")
        for v in meta['sig_vars']:
            if v not in df_fe.index:
                continue
            row  = df_fe.loc[v]
            coef = float(row['coef'])
            pval = float(row['pvalue'])
            star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
            var  = 'green' if coef > 0 else 'red'
            ib(f"{VL.get(v, v)}  {star}",
               f"Koef {coef:+.4f} — {'efek positif' if coef>0 else 'efek negatif'} terhadap IPM",
               var)


# ══════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    header("Clustering <em>UMAP + GMM</em>",
           "Gaussian Mixture Model pada 2D UMAP Embedding | Cross-Section 2024")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("K Optimal",        str(meta['optimal_k']), sub="via BIC minimum")
    with c2: kpi("Silhouette",       f"{meta['sil_score']:.4f}", sub="makin tinggi makin baik")
    with c3: kpi("Davies-Bouldin",   f"{meta['db_score']:.4f}", sub="makin rendah makin baik")
    with c4: kpi("Calinski-Harabasz",f"{meta['ch_score']:.0f}", sub="makin tinggi makin baik")

    st.markdown("<br>", unsafe_allow_html=True)
    la, lb = st.columns([3, 2])

    with la:
        sec("UMAP Embedding + Label Cluster")
        fig = go.Figure()
        for cl_id in sorted(df_cl['cluster_id'].unique()):
            sub = df_cl[df_cl['cluster_id'] == cl_id]
            lbl = sub['cluster_label'].iloc[0]
            fig.add_trace(go.Scatter(
                x=sub['umap_1'], y=sub['umap_2'],
                mode='markers+text', name=lbl,
                marker=dict(size=13, color=cl_color(cl_id),
                            line=dict(width=1.5, color='#F7F5F0')),
                text=sub['provinsi'].str.title().str[:10],
                textposition='top center',
                textfont=dict(family='DM Sans, sans-serif', size=8, color='#4A4440'),
                hovertemplate=(
                    '<b>' + sub['provinsi'].str.title() + '</b><br>'
                    'IPM: ' + sub['ipm'].round(2).astype(str) + '<br>'
                    'Conf: ' + sub['cluster_confidence'].round(4).astype(str) +
                    '<extra></extra>'
                )
            ))
        fig.update_layout(**base_layout(
            height=460, xaxis_title='UMAP Dimensi 1', yaxis_title='UMAP Dimensi 2',
            legend=dict(orientation='h', y=-0.12, font_size=10)
        ))
        st.plotly_chart(fig, use_container_width=True)

        sec("Confidence Probabilistik GMM per Provinsi")
        dc = df_cl.sort_values('cluster_confidence')
        figc = go.Figure(go.Bar(
            x=dc['cluster_confidence'], y=dc['provinsi'].str.title(), orientation='h',
            marker=dict(color=[cl_color(c) for c in dc['cluster_id']], line=dict(width=0)),
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.4f}<extra></extra>'
        ))
        figc.add_vline(x=0.9, line_color='#9B8E7E', line_dash='dash', line_width=1)
        figc.update_layout(**base_layout(
            height=480, xaxis_range=[0.5, 1.02],
            xaxis_title='Probabilitas Keanggotaan',
            margin=dict(l=0, r=10, t=10, b=10)
        ))
        st.plotly_chart(figc, use_container_width=True)

    with lb:
        sec("Profil Rata-rata per Cluster")
        prof = df_cl.groupby('cluster_label')[
            ['ipm', 'pct_miskin_maret', 'tpt_feb', 'tpak_feb', 'aps_1618', 'aps_1924']
        ].mean().round(2)
        prof.columns = ['IPM', '% Miskin', 'TPT Feb', 'TPAK Feb', 'APS 16-18', 'APS 19-24']
        st.dataframe(prof, use_container_width=True, height=180)

        sec("Anggota per Cluster")
        for cl_id in sorted(df_cl['cluster_id'].unique()):
            sub    = df_cl[df_cl['cluster_id'] == cl_id].sort_values('ipm', ascending=False)
            lbl    = sub['cluster_label'].iloc[0]
            color  = cl_color(cl_id)
            st.markdown(
                f'<div style="background:#fff; border:1px solid #E5DDD0; '
                f'border-top:3px solid {color}; padding:0.9rem 1rem; '
                f'margin-bottom:0.7rem; border-radius:0 0 2px 2px;">'
                f'<div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">'
                f'<span style="font-family:DM Mono,monospace; font-size:0.62rem; '
                f'letter-spacing:0.1em; text-transform:uppercase; color:#6B5F50;">{lbl}</span>'
                f'<span style="font-family:DM Mono,monospace; font-size:0.62rem; color:#9B8E7E;">'
                f'n={len(sub)} | IPM {sub["ipm"].mean():.2f}</span></div>',
                unsafe_allow_html=True
            )
            for _, row in sub.iterrows():
                st.markdown(
                    f'<div class="prow"><span class="pname">{row["provinsi"].title()}</span>'
                    f'<span class="pval">{row["ipm"]:.2f}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
elif page == "Peta Spasial":
    header("Distribusi <em>Spasial</em>",
           "Sebaran Cluster dan Nilai IPM di Seluruh Wilayah Indonesia")

    mode = st.radio("mode", ["Cluster GMM", "Nilai IPM 2024", "Proyeksi IPM 2027"],
                    horizontal=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    map_df = df_cl.copy()
    map_df['lat'] = map_df['provinsi'].map(lambda p: PROV_COORDS.get(p, (None, None))[0])
    map_df['lon'] = map_df['provinsi'].map(lambda p: PROV_COORDS.get(p, (None, None))[1])
    map_df = map_df.dropna(subset=['lat', 'lon'])

    if mode == "Cluster GMM":
        fig = px.scatter_mapbox(
            map_df, lat='lat', lon='lon', color='cluster_label',
            size=[16]*len(map_df), hover_name='provinsi',
            hover_data={'ipm':':.2f','cluster_confidence':':.4f',
                        'lat':False,'lon':False,'cluster_label':False},
            color_discrete_sequence=[cl_color(i) for i in range(meta['optimal_k'])],
            mapbox_style='carto-positron', zoom=3.8,
            center={'lat':-2.5,'lon':118}, height=560
        )
    elif mode == "Nilai IPM 2024":
        fig = px.scatter_mapbox(
            map_df, lat='lat', lon='lon', color='ipm', size='ipm', size_max=18,
            hover_name='provinsi',
            hover_data={'ipm':':.2f','cluster_label':True,'lat':False,'lon':False},
            color_continuous_scale=[[0,'#9B1D20'],[0.5,'#C8B89A'],[1,'#2D6A4F']],
            range_color=[62, 84], mapbox_style='carto-positron', zoom=3.8,
            center={'lat':-2.5,'lon':118}, height=560, labels={'ipm':'IPM 2024'}
        )
    else:
        fc27 = df_fc[df_fc['tahun'] == 2027].set_index('provinsi')['ipm_forecast']
        map_df['ipm_2027'] = map_df['provinsi'].map(fc27)
        fig = px.scatter_mapbox(
            map_df.dropna(subset=['ipm_2027']),
            lat='lat', lon='lon', color='ipm_2027', size='ipm_2027', size_max=18,
            hover_name='provinsi',
            hover_data={'ipm_2027':':.2f','ipm':':.2f','lat':False,'lon':False},
            color_continuous_scale=[[0,'#9B1D20'],[0.5,'#C8B89A'],[1,'#2D6A4F']],
            range_color=[62, 84], mapbox_style='carto-positron', zoom=3.8,
            center={'lat':-2.5,'lon':118}, height=560, labels={'ipm_2027':'IPM 2027'}
        )

    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='#F7F5F0',
                      legend=dict(font=dict(family='DM Sans', size=11)))
    st.plotly_chart(fig, use_container_width=True)

    sec("Data Lengkap")
    show = df_cl[['provinsi','cluster_label','ipm','pct_miskin_maret',
                  'tpt_feb','tpak_feb','aps_1618','cluster_confidence']].copy()
    show.columns = ['Provinsi','Cluster','IPM 2024','% Miskin','TPT Feb',
                    'TPAK Feb','APS 16-18','Confidence GMM']
    show = show.sort_values('IPM 2024', ascending=False).reset_index(drop=True)
    st.dataframe(show.round(4), use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════════════════
elif page == "Regresi Panel":
    header("Fixed Effect <em>Panel Regression</em>",
           "Determinan IPM | Cluster-Robust SE | Entity Effects | 151 Observasi")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("R2 Within",  f"{meta['fe_r2_within']:.4f}",  sub="variasi dalam provinsi")
    with c2: kpi("R2 Overall", f"{meta['fe_r2_overall']:.4f}", sub="variasi keseluruhan")
    with c3: kpi("Var. Signifikan", str(len(meta['sig_vars'])), sub="pada p < 0.05")
    with c4: kpi("N Observasi", "151", sub="38 prov × 2021-2024")

    st.markdown("<br>", unsafe_allow_html=True)
    la, lb = st.columns([3, 2])

    with la:
        sec("Koefisien Regresi + 95% Confidence Interval")
        fp = df_fe.copy()
        fp['label'] = [VL.get(v, v) for v in fp.index]
        fp['color'] = fp['significant'].map({True: '#0F1419', False: '#C8B89A'})
        fp = fp.sort_values('coef')

        fig = go.Figure(go.Bar(
            x=fp['coef'], y=fp['label'], orientation='h',
            marker=dict(color=fp['color'], line=dict(width=0)),
            error_x=dict(type='data', array=1.96*fp['std_err'],
                         visible=True, color='#9B8E7E', thickness=1.5),
            text=['  '+s if s and str(s) not in ('nan', 'None', '') else '' for s in fp['signif_star'].fillna('').astype(str)],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=11, color='#9B1D20'),
            hovertemplate='<b>%{y}</b><br>Koef: %{x:.4f}<extra></extra>'
        ))
        fig.add_vline(x=0, line_color='#0F1419', line_width=1.5, line_dash='dot')
        fig.update_layout(**base_layout(height=360, xaxis_title='Koefisien Regresi',
                                        margin=dict(l=0, r=60, t=10, b=10)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div style="font-family:DM Mono,monospace; font-size:0.6rem; color:#9B8E7E;">'
            'Hitam = signifikan p&lt;0.05 &nbsp;|&nbsp; Abu = tidak signifikan '
            '&nbsp;|&nbsp; *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05'
            '</div>', unsafe_allow_html=True
        )

    with lb:
        sec("Tabel Koefisien Lengkap")
        tbl = df_fe[['coef','std_err','pvalue','signif_star']].copy()
        tbl.index = [VL.get(v, v) for v in tbl.index]
        tbl.columns = ['Koefisien','Std Error','P-Value','Sig']
        tbl = tbl.round(4)
        st.dataframe(tbl, use_container_width=True, height=260)

        sec("Interpretasi Variabel Signifikan")
        if len(meta['sig_vars']) == 0:
            ib("Catatan", "Tidak ada variabel signifikan pada p < 0.05.")
        for v in meta['sig_vars']:
            if v not in df_fe.index:
                continue
            row  = df_fe.loc[v]
            coef = float(row['coef'])
            pval = float(row['pvalue'])
            star = '***' if pval<0.001 else '**' if pval<0.01 else '*'
            arah = "menaikkan" if coef > 0 else "menurunkan"
            ib(f"{VL.get(v, v)}  {star}",
               f"Koef = {coef:+.4f}. Kenaikan 1 satuan {arah} IPM sebesar "
               f"{abs(coef):.4f} poin (p = {pval:.4f}).",
               'green' if coef > 0 else 'red')

        sec("Catatan Metodologis")
        ib("Spesifikasi Model",
           "Fixed Effect Model mengendalikan heterogenitas tidak teramati yang bersifat "
           "time-invariant per provinsi. SE menggunakan cluster-robust (clustered by entity). "
           "rls dan hls dikeluarkan karena merupakan sub-indeks langsung pembentuk IPM.")


# ══════════════════════════════════════════════════════════════════════
elif page == "SHAP Analysis":
    header("SHAP <em>Feature Importance</em>",
           "SHapley Additive Explanations | Global + Per Cluster | XGBoost")

    shap_imp = df_sh[FEATURES].abs().mean().sort_values(ascending=False)
    total_s  = shap_imp.sum()

    c1, c2, c3 = st.columns(3)
    with c1: kpi("Fitur Dominan", FL[shap_imp.index[0]],
                 sub=f"mean |SHAP| = {shap_imp.iloc[0]:.4f}")
    with c2: kpi("CV R2", f"{meta['xgb_cv_r2_mean']:.4f}",
                 sub=f"std {meta['xgb_cv_r2_std']:.4f}")
    with c3: kpi("CV MAE", f"{meta['xgb_cv_mae_mean']:.4f}",
                 sub="TimeSeriesSplit 3-fold")

    st.markdown("<br>", unsafe_allow_html=True)
    la, lb = st.columns([2, 3])

    with la:
        sec("Global Feature Importance")
        fi_rev = shap_imp.index[::-1]
        bar_c  = ['#0F1419' if i >= len(shap_imp)-3 else
                  '#8B7355' if i >= len(shap_imp)-5 else '#C8B89A'
                  for i in range(len(shap_imp))]
        fig = go.Figure(go.Bar(
            x=shap_imp.values[::-1], y=[FL[f] for f in fi_rev],
            orientation='h',
            marker=dict(color=bar_c, line=dict(width=0)),
            text=[f'{v/total_s*100:.1f}%' for v in shap_imp.values[::-1]],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=9, color='#6B5F50'),
            hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
        ))
        fig.update_layout(**base_layout(height=300, xaxis_title='Mean |SHAP Value|',
                                        margin=dict(l=0, r=70, t=10, b=10)))
        st.plotly_chart(fig, use_container_width=True)

        sec("Ranking Kontribusi")
        for rank, (feat, val) in enumerate(shap_imp.items(), 1):
            tier   = "Dominan" if rank == 1 else "Penting" if rank <= 3 else "Pendukung"
            border = '#0F1419' if rank == 1 else '#8B7355' if rank <= 3 else '#C8B89A'
            st.markdown(
                f'<div style="display:flex; gap:0.7rem; padding:0.45rem 0; '
                f'border-bottom:1px solid #F0E8DE;">'
                f'<div style="font-family:DM Mono,monospace; font-size:0.65rem; '
                f'color:{border}; width:1rem; font-weight:600;">{rank}</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.78rem; color:#2A2420; font-weight:500;">{FL[feat]}</div>'
                f'<div style="font-family:DM Mono,monospace; font-size:0.62rem; color:#9B8E7E;">'
                f'{val:.4f} &nbsp;|&nbsp; {val/total_s*100:.1f}% &nbsp;|&nbsp; {tier}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    with lb:
        sec("SHAP Scatter — Nilai Fitur vs Dampak terhadap IPM")
        feat_sel = st.selectbox("feat", FEATURES, format_func=lambda f: FL.get(f, f),
                                label_visibility="collapsed")
        sv      = df_sh[feat_sel].values
        fv      = df_sh[feat_sel].values
        ipm_v   = df_raw.sort_values(['tahun','provinsi'])['ipm'].values
        ipm_v   = ipm_v[:len(sv)]

        fig_sc = go.Figure(go.Scatter(
            x=fv, y=sv, mode='markers',
            marker=dict(size=7, opacity=0.75, color=ipm_v,
                        colorscale=[[0,'#9B1D20'],[0.5,'#C8B89A'],[1,'#2D6A4F']],
                        showscale=True,
                        colorbar=dict(title=dict(text='IPM', side='right'),
                                      thickness=10, len=0.8,
                                      tickfont=dict(family='DM Mono,monospace', size=9)),
                        line=dict(width=0)),
            hovertemplate=f'{FL[feat_sel]}: %{{x:.2f}}<br>SHAP: %{{y:.4f}}<extra></extra>'
        ))
        fig_sc.add_hline(y=0, line_color='#9B8E7E', line_dash='dash', line_width=1)
        fig_sc.update_layout(**base_layout(
            height=290, xaxis_title=FL.get(feat_sel, feat_sel),
            yaxis_title='SHAP Value', margin=dict(l=0, r=10, t=10, b=10)
        ))
        st.plotly_chart(fig_sc, use_container_width=True)

        sec("SHAP Importance per Cluster")
        cols_cl = st.columns(meta['optimal_k'])
        for i, cl_id in enumerate(sorted(df_cl['cluster_id'].unique())):
            lbl        = df_cl[df_cl['cluster_id']==cl_id]['cluster_label'].iloc[0]
            prov_in_cl = df_cl[df_cl['cluster_id']==cl_id]['provinsi'].tolist()
            mask_cl    = df_sh['provinsi'].isin(prov_in_cl)
            sh_cl      = df_sh.loc[mask_cl, FEATURES].abs().mean().sort_values(ascending=True)
            with cols_cl[i]:
                fig_cl = go.Figure(go.Bar(
                    x=sh_cl.values, y=[FL[f] for f in sh_cl.index], orientation='h',
                    marker=dict(color=cl_color(cl_id), line=dict(width=0)),
                    hovertemplate='%{y}: %{x:.4f}<extra></extra>'
                ))
                fig_cl.update_layout(**base_layout(
                    height=280, xaxis_title='Mean |SHAP|',
                    title=dict(text=lbl, font=dict(family='DM Mono,monospace',
                               size=9, color='#6B5F50'), x=0),
                    margin=dict(l=0, r=10, t=30, b=10)
                ))
                st.plotly_chart(fig_cl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    header("Proyeksi IPM <em>2025 – 2027</em>",
           "XGBoost | OLS Trend + Slope Damping | TimeSeriesSplit CV")

    fc27   = df_fc[df_fc['tahun'] == 2027]
    avg24  = df_cl['ipm'].mean()
    avg27  = fc27['ipm_forecast'].mean()
    ipm24  = df_cl.set_index('provinsi')['ipm']

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Rerata 2027", f"{avg27:.2f}", delta=avg27 - avg24)
    with c2: kpi("Tertinggi 2027", f"{fc27['ipm_forecast'].max():.2f}",
                 sub=fc27.loc[fc27['ipm_forecast'].idxmax(),'provinsi'].title())
    with c3: kpi("Terendah 2027", f"{fc27['ipm_forecast'].min():.2f}",
                 sub=fc27.loc[fc27['ipm_forecast'].idxmin(),'provinsi'].title())
    with c4: kpi("CV R2", f"{meta['xgb_cv_r2_mean']:.4f}",
                 sub=f"RMSE {meta['xgb_cv_rmse_mean']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Tren Cluster", "Top & Bottom", "Detail Provinsi"])

    with tab1:
        sec("Tren IPM Historis dan Proyeksi per Cluster")
        fig = go.Figure()
        for cl_id in sorted(df_fc['cluster_id'].dropna().unique()):
            cl_id  = int(cl_id)
            lbl    = df_cl[df_cl['cluster_id']==cl_id]['cluster_label'].iloc[0]
            cl_d   = df_fc[df_fc['cluster_id']==cl_id].groupby('tahun')['ipm_forecast'].mean()
            hm, fm = cl_d.index <= 2024, cl_d.index >= 2024
            fig.add_trace(go.Scatter(
                x=cl_d.index[hm], y=cl_d.values[hm],
                name=lbl, line=dict(color=cl_color(cl_id), width=2.5),
                mode='lines+markers', marker=dict(size=7)
            ))
            fig.add_trace(go.Scatter(
                x=cl_d.index[fm], y=cl_d.values[fm],
                name=f'{lbl} (proj)', line=dict(color=cl_color(cl_id), width=2, dash='dot'),
                mode='lines+markers', marker=dict(size=6, symbol='diamond'), showlegend=True
            ))
        fig.add_vline(x=2024.5, line_color='#C8B89A', line_dash='dot', line_width=1.5)
        fig.update_layout(**base_layout(
            height=380, xaxis_title='Tahun', yaxis_title='IPM',
            legend=dict(orientation='h', y=-0.2, font_size=10)
        ))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ct, cb = st.columns(2)
        with ct:
            sec("10 Provinsi IPM Tertinggi — 2027")
            top10 = fc27.nlargest(10,'ipm_forecast').sort_values('ipm_forecast')
            top10 = top10.copy()
            top10['ipm_2024'] = top10['provinsi'].map(ipm24)
            fig_t = go.Figure(go.Bar(
                x=top10['ipm_forecast'], y=top10['provinsi'].str.title(),
                orientation='h',
                marker=dict(color=[cl_color(c) for c in top10['cluster_id']], line=dict(width=0)),
                text=top10['ipm_forecast'].round(2), textposition='outside',
                textfont=dict(family='DM Mono,monospace', size=9, color='#6B5F50'),
                hovertemplate='<b>%{y}</b><br>IPM 2027: %{x:.2f}<extra></extra>'
            ))
            fig_t.update_layout(**base_layout(height=340, xaxis_range=[62, 88],
                                               xaxis_title='IPM 2027',
                                               margin=dict(l=0,r=60,t=10,b=10)))
            st.plotly_chart(fig_t, use_container_width=True)

        with cb:
            sec("10 Provinsi IPM Terendah — 2027")
            bot10 = fc27.nsmallest(10,'ipm_forecast').sort_values('ipm_forecast', ascending=False)
            bot10 = bot10.copy()
            bot10['ipm_2024'] = bot10['provinsi'].map(ipm24)
            fig_b = go.Figure(go.Bar(
                x=bot10['ipm_forecast'], y=bot10['provinsi'].str.title(),
                orientation='h',
                marker=dict(color=[cl_color(c) for c in bot10['cluster_id']], line=dict(width=0)),
                text=bot10['ipm_forecast'].round(2), textposition='outside',
                textfont=dict(family='DM Mono,monospace', size=9, color='#6B5F50'),
                hovertemplate='<b>%{y}</b><br>IPM 2027: %{x:.2f}<extra></extra>'
            ))
            fig_b.update_layout(**base_layout(height=340, xaxis_range=[55, 78],
                                               xaxis_title='IPM 2027',
                                               margin=dict(l=0,r=60,t=10,b=10)))
            st.plotly_chart(fig_b, use_container_width=True)

    with tab3:
        prov_sel = st.selectbox("prov", sorted(df_raw['provinsi'].unique()),
                                format_func=str.title, label_visibility="collapsed")
        hist_p = df_raw[df_raw['provinsi']==prov_sel][['tahun','ipm']].copy()
        fc_p   = df_fc[(df_fc['provinsi']==prov_sel) & (df_fc['tahun']>2024)][['tahun','ipm_forecast']].copy()
        cl_p   = df_cl[df_cl['provinsi']==prov_sel].iloc[0]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=hist_p['tahun'], y=hist_p['ipm'], name='Historis',
            line=dict(color='#0F1419', width=2.5),
            mode='lines+markers', marker=dict(size=9, color='#0F1419')
        ))
        fig_p.add_trace(go.Scatter(
            x=fc_p['tahun'], y=fc_p['ipm_forecast'], name='Proyeksi',
            line=dict(color='#8B7355', width=2, dash='dot'),
            mode='lines+markers', marker=dict(size=8, symbol='diamond', color='#8B7355')
        ))
        fig_p.add_vline(x=2024.5, line_color='#C8B89A', line_dash='dot')
        fig_p.update_layout(**base_layout(
            height=300, xaxis_title='Tahun', yaxis_title='IPM',
            title=dict(text=prov_sel.title(),
                       font=dict(family='DM Serif Display,serif', size=15, color='#0F1419'),
                       x=0, xanchor='left'),
            legend=dict(orientation='h', y=-0.25)
        ))
        st.plotly_chart(fig_p, use_container_width=True)

        cc1, cc2, cc3 = st.columns(3)
        val27 = fc_p[fc_p['tahun']==2027]['ipm_forecast']
        v27   = float(val27.values[0]) if len(val27) > 0 else float('nan')
        with cc1: kpi("IPM 2024", f"{cl_p['ipm']:.2f}")
        with cc2: kpi("Proyeksi 2027", f"{v27:.2f}", delta=v27 - cl_p['ipm'])
        with cc3: kpi("Cluster", cl_p['cluster_label'])


# ══════════════════════════════════════════════════════════════════════
elif page == "Paradoks & Underdog":
    header("Analisis <em>Paradoks & Underdog</em>",
           "Identifikasi Provinsi dengan Pola Pertumbuhan Proyeksi Tidak Linier")

    mean_ipm   = df_gr['ipm_2024'].mean()
    mean_delta = df_gr['delta'].mean()
    paradox_p  = meta['paradox_provinces']
    underdog_p = meta['underdog_provinces']

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Rata-rata IPM 2024",  f"{mean_ipm:.2f}")
    with c2: kpi("Rata-rata Delta",     f"{mean_delta:.3f}", sub="2024 ke 2027")
    with c3: kpi("Provinsi Paradoks",   str(len(paradox_p)), sub="IPM tinggi, tumbuh lambat")
    with c4: kpi("Provinsi Underdog",   str(len(underdog_p)), sub="IPM rendah, tumbuh cepat")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Peta Kuadran: IPM 2024 vs Delta Proyeksi 2024 – 2027")

    df_gr2 = df_gr.set_index('provinsi').copy()

    def cat(prov):
        if prov in paradox_p:  return 'Paradoks'
        if prov in underdog_p: return 'Underdog'
        row = df_gr2.loc[prov]
        if row['ipm_2024'] >= mean_ipm and row['delta'] >= mean_delta: return 'Unggul'
        return 'Perlu Perhatian'

    df_gr2['kategori'] = df_gr2.index.map(cat)
    df_gr2 = df_gr2.reset_index()

    cmap = {'Paradoks':'#9B1D20','Underdog':'#2D6A4F',
            'Unggul':'#0F1419','Perlu Perhatian':'#C8B89A'}
    fig = px.scatter(
        df_gr2, x='ipm_2024', y='delta',
        color='kategori', color_discrete_map=cmap,
        text='provinsi',
        hover_name='provinsi',
        hover_data={'ipm_2024':':.2f','ipm_2027':':.2f','delta':':.3f',
                    'cluster_label':True,'kategori':False},
        height=500
    )
    fig.update_traces(
        textposition='top center',
        textfont=dict(family='DM Sans,sans-serif', size=8, color='#4A4440'),
        marker=dict(size=9, line=dict(width=1, color='#F7F5F0'))
    )
    fig.add_hline(y=mean_delta, line_color='#8B7355', line_dash='dash', line_width=1.5,
                  annotation_text=f'rata-rata delta {mean_delta:.3f}',
                  annotation_font=dict(family='DM Mono,monospace', size=9, color='#8B7355'))
    fig.add_vline(x=mean_ipm, line_color='#8B7355', line_dash='dash', line_width=1.5,
                  annotation_text=f'rata-rata IPM {mean_ipm:.2f}',
                  annotation_font=dict(family='DM Mono,monospace', size=9, color='#8B7355'))
    fig.update_layout(**base_layout(
        xaxis_title='IPM 2024', yaxis_title='Delta IPM (2027 - 2024)',
        legend=dict(orientation='h', y=-0.12, font_size=11)
    ))
    st.plotly_chart(fig, use_container_width=True)

    cp, cu = st.columns(2)
    with cp:
        sec(f"Provinsi Paradoks ({len(paradox_p)})")
        ib("Definisi",
           "IPM 2024 di atas rata-rata nasional, namun delta proyeksi 2024-2027 "
           "berada di bawah rata-rata. Mengindikasikan potensi stagnasi di provinsi "
           "yang sudah relatif maju.", 'red')
        df_par = df_gr.set_index('provinsi')
        df_par = df_par[df_par.index.isin(paradox_p)].sort_values('delta')
        for prov, row in df_par.iterrows():
            st.markdown(
                f'<div class="prow"><span class="pname">{prov.title()}</span>'
                f'<span style="display:flex; gap:0.8rem;">'
                f'<span class="pval">IPM {row["ipm_2024"]:.2f}</span>'
                f'<span style="font-family:DM Mono,monospace; font-size:0.73rem; color:#9B1D20;">'
                f'{row["delta"]:+.3f}</span></span></div>',
                unsafe_allow_html=True
            )

    with cu:
        sec(f"Provinsi Underdog ({len(underdog_p)})")
        ib("Definisi",
           "IPM 2024 di bawah rata-rata nasional, namun delta proyeksi 2024-2027 "
           "berada di atas rata-rata. Kandidat prioritas investasi dengan potensi "
           "catch-up yang tinggi.", 'green')
        df_und = df_gr.set_index('provinsi')
        df_und = df_und[df_und.index.isin(underdog_p)].sort_values('delta', ascending=False)
        for prov, row in df_und.iterrows():
            st.markdown(
                f'<div class="prow"><span class="pname">{prov.title()}</span>'
                f'<span style="display:flex; gap:0.8rem;">'
                f'<span class="pval">IPM {row["ipm_2024"]:.2f}</span>'
                f'<span style="font-family:DM Mono,monospace; font-size:0.73rem; color:#2D6A4F;">'
                f'{row["delta"]:+.3f}</span></span></div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════
elif page == "Insight Gabungan":
    header("Insight <em>Gabungan</em>",
           "Sintesis Komprehensif: Clustering | Regresi | SHAP | Forecasting")

    shap_imp = df_sh[FEATURES].abs().mean().sort_values(ascending=False)
    total_s  = shap_imp.sum()
    avg24    = df_cl['ipm'].mean()
    avg27    = df_fc[df_fc['tahun']==2027]['ipm_forecast'].mean()
    fc27     = df_fc[df_fc['tahun']==2027]
    sig_coef = df_fe[df_fe['significant'] == True].sort_values('coef', key=abs, ascending=False)

    # 1. Clustering
    sec("1. Temuan Clustering (UMAP + GMM)")
    for cl_id in sorted(df_cl['cluster_id'].unique()):
        sub    = df_cl[df_cl['cluster_id'] == cl_id]
        lbl    = sub['cluster_label'].iloc[0]
        color  = cl_color(cl_id)
        members = ', '.join(sub.sort_values('ipm', ascending=False)['provinsi'].str.title().tolist())
        st.markdown(
            f'<div style="background:#fff; border:1px solid #E5DDD0; '
            f'border-left:4px solid {color}; padding:0.9rem 1.1rem; '
            f'margin-bottom:0.55rem; border-radius:0 2px 2px 0;">'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">'
            f'<span style="font-family:DM Mono,monospace; font-size:0.62rem; '
            f'letter-spacing:0.1em; text-transform:uppercase; color:#6B5F50;">{lbl}</span>'
            f'<span style="font-family:DM Mono,monospace; font-size:0.62rem; color:#9B8E7E;">'
            f'n={len(sub)} &nbsp;|&nbsp; IPM {sub["ipm"].mean():.2f} &nbsp;|&nbsp; '
            f'% miskin {sub["pct_miskin_maret"].mean():.1f}%</span></div>'
            f'<div style="font-family:DM Sans,sans-serif; font-size:0.78rem; '
            f'color:#4A4440; line-height:1.6;">{members}</div></div>',
            unsafe_allow_html=True
        )

    # 2. FEM
    sec("2. Faktor Penentu IPM (Fixed Effect Model)")
    c1, c2 = st.columns([1, 2])
    with c1:
        for lbl, val in [("R2 Within",  f"{meta['fe_r2_within']:.4f}"),
                         ("R2 Overall", f"{meta['fe_r2_overall']:.4f}"),
                         ("N Obs",      "151")]:
            st.markdown(
                f'<div class="strow"><span class="stlbl">{lbl}</span>'
                f'<span class="stval">{val}</span></div>',
                unsafe_allow_html=True
            )
    with c2:
        if len(sig_coef) == 0:
            ib("Catatan", "Tidak ada variabel signifikan pada p < 0.05.")
        for v, row in sig_coef.iterrows():
            coef = float(row['coef']); pval = float(row['pvalue'])
            star = '***' if pval<0.001 else '**' if pval<0.01 else '*'
            arah = "menaikkan" if coef > 0 else "menurunkan"
            ib(f"{VL.get(v, v)}  {star}",
               f"Koef {coef:+.4f}. Kenaikan 1 satuan {arah} IPM "
               f"{abs(coef):.4f} poin (p={pval:.4f}).",
               'green' if coef > 0 else 'red')

    # 3. SHAP
    sec("3. Kontribusi Variabel (SHAP)")
    cols_sh = st.columns(len(FEATURES))
    for i, (feat, val) in enumerate(shap_imp.items()):
        with cols_sh[i]:
            tier   = "Dominan" if i==0 else "Penting" if i<3 else "Pendukung"
            border = '#0F1419' if i==0 else '#8B7355' if i<3 else '#C8B89A'
            st.markdown(
                f'<div style="border-top:3px solid {border}; padding:0.65rem 0.55rem; '
                f'background:#fff; border:1px solid #E5DDD0; border-top:3px solid {border};">'
                f'<div style="font-family:DM Mono,monospace; font-size:0.56rem; '
                f'color:#9B8E7E; text-transform:uppercase; letter-spacing:0.08em;">{tier}</div>'
                f'<div style="font-family:DM Sans,sans-serif; font-size:0.73rem; '
                f'color:#2A2420; margin:0.18rem 0; font-weight:500;">{FL[feat]}</div>'
                f'<div style="font-family:DM Mono,monospace; font-size:0.7rem; '
                f'color:#0F1419;">{val:.4f}</div>'
                f'<div style="font-family:DM Mono,monospace; font-size:0.6rem; '
                f'color:#9B8E7E;">{val/total_s*100:.1f}%</div></div>',
                unsafe_allow_html=True
            )

    # 4. Forecasting
    sec("4. Proyeksi IPM 2027")
    st.markdown(
        f'<div class="strow"><span class="stlbl">Rata-rata nasional 2024</span>'
        f'<span class="stval">{avg24:.2f}</span></div>'
        f'<div class="strow"><span class="stlbl">Proyeksi rata-rata 2027</span>'
        f'<span class="stval">{avg27:.2f} &nbsp; ({avg27-avg24:+.2f})</span></div>',
        unsafe_allow_html=True
    )
    top3 = fc27.nlargest(3, 'ipm_forecast')
    bot3 = fc27.nsmallest(3, 'ipm_forecast')
    ct3, cb3 = st.columns(2)
    with ct3:
        ib("Tertinggi 2027",
           '<br>'.join([f'{r["provinsi"].title()}: {r["ipm_forecast"]:.2f}'
                        for _, r in top3.iterrows()]), 'green')
    with cb3:
        ib("Terendah 2027",
           '<br>'.join([f'{r["provinsi"].title()}: {r["ipm_forecast"]:.2f}'
                        for _, r in bot3.iterrows()]), 'red')

    # 5. Paradoks & Underdog
    sec("5. Paradoks & Underdog")
    cp5, cu5 = st.columns(2)
    with cp5:
        ib(f"Paradoks — {len(meta['paradox_provinces'])} provinsi",
           f"IPM di atas rata-rata ({avg24:.2f}) namun laju pertumbuhan proyeksi di bawah "
           f"rata-rata. Risiko stagnasi struktural.<br><br>"
           + ', '.join([p.title() for p in meta['paradox_provinces']]), 'red')
    with cu5:
        ib(f"Underdog — {len(meta['underdog_provinces'])} provinsi",
           f"IPM di bawah rata-rata ({avg24:.2f}) namun laju pertumbuhan proyeksi di atas "
           f"rata-rata. Kandidat prioritas investasi.<br><br>"
           + ', '.join([p.title() for p in meta['underdog_provinces']]), 'green')

    # 6. Rekomendasi
    sec("6. Rekomendasi Kebijakan")
    ib("Prioritas Intervensi Variabel",
       f'Fokuskan pada {FL[shap_imp.index[0]]} dan {FL[shap_imp.index[1]]} '
       f'sebagai dua prediktor IPM terkuat berdasarkan SHAP '
       f'(kontribusi gabungan {(shap_imp.iloc[0]+shap_imp.iloc[1])/total_s*100:.1f}% dari total).')
    for v, row in sig_coef.iterrows():
        coef = float(row['coef']); pval = float(row['pvalue'])
        if coef < 0:
            ib("Reduksi Diprioritaskan",
               f'Penurunan {VL.get(v, v)} terbukti signifikan meningkatkan IPM '
               f'(koef={coef:+.4f}, p={pval:.4f}).', 'red')
        else:
            ib("Peningkatan Diprioritaskan",
               f'Peningkatan {VL.get(v, v)} terbukti signifikan menaikkan IPM '
               f'(koef={coef:+.4f}, p={pval:.4f}).', 'green')
    if meta['paradox_provinces']:
        ib("Evaluasi Struktural",
           f'{len(meta["paradox_provinces"])} provinsi paradoks perlu audit kebijakan mendalam '
           f'untuk mengidentifikasi hambatan yang memperlambat pertumbuhan IPM di tengah '
           f'kondisi awal yang relatif baik.', 'amber')
    if meta['underdog_provinces']:
        ib("Investasi Prioritas",
           f'{len(meta["underdog_provinces"])} provinsi underdog menunjukkan momentum positif. '
           f'Alokasi anggaran pembangunan yang lebih besar berpotensi mempercepat catch-up '
           f'terhadap rata-rata nasional.', 'green')