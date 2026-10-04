"""
Demo — Normalización de Láminas H&E
Tesis de Maestría — Jorge Benítez
Universidad Comunera (UCOM) — Paraguay

Algoritmos implementados:
    - Reinhard (tiatoolbox)
    - Vahadane (wsi-normalizer)
    - ACD — Zheng et al. (2019) — eta=0.4, gamma=0.6
    - SCAN — Salvi et al. (2020)
"""

import io
import time
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Normalización H&E — UCOM",
    page_icon="🔬",
    layout="wide",
)

# ── Paleta UCOM ──────────────────────────────────────────────────────────────
LILA        = "#6247aa"
LILA_DARK   = "#4b3280"
LILA_LIGHT  = "#ede9fa"
VERDE       = "#2e8b57"
VERDE_DARK  = "#236b43"
BG          = "#ffffff"
BG2         = "#f5f4fb"
BORDER      = "#ddd8f0"
FG          = "#1a1a2e"
MUTED       = "#5a5a7a"
RED_WARN    = "#c0392b"
ORANGE      = "#e67e22"

# ── Estilos globales ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Fondo general blanco */
    .stApp, .main, section.main > div {{
        background-color: {BG} !important;
    }}
    header[data-testid="stHeader"] {{
        background-color: {BG} !important;
        border-bottom: 1px solid {BORDER} !important;
    }}

    /* Texto oscuro legible sobre fondo blanco */
    p, span, label, div, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText {{
        color: {FG} !important;
    }}
    .stCaption, small {{
        color: {MUTED} !important;
        font-size: 0.84rem !important;
    }}
    hr {{
        border: none !important;
        border-top: 2px solid {LILA} !important;
        margin: 1.4rem 0 !important;
        opacity: 0.25 !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {LILA_LIGHT} !important;
        border: 2px dashed {LILA} !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }}
    [data-testid="stFileUploader"] * {{
        color: {FG} !important;
    }}
    [data-testid="stFileUploader"] button {{
        background: {LILA} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }}

    /* Botón primario — VERDE */
    .stButton > button[kind="primary"] {{
        background: {VERDE} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.5rem !important;
        transition: background 0.2s !important;
        box-shadow: 0 2px 8px rgba(46,139,87,0.25) !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {VERDE_DARK} !important;
    }}

    /* Botón de descarga individual */
    .stDownloadButton > button {{
        width: 100% !important;
        background: {VERDE} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        transition: background 0.2s !important;
    }}
    .stDownloadButton > button:hover {{
        background: {VERDE_DARK} !important;
    }}

    /* Spinner */
    .stSpinner > div {{
        border-top-color: {LILA} !important;
    }}

    /* Alert */
    .stAlert {{
        background: {LILA_LIGHT} !important;
        border-left: 4px solid {LILA} !important;
        color: {FG} !important;
    }}

    /* Imagen */
    [data-testid="stImage"] img {{
        border-radius: 8px !important;
        border: 1px solid {BORDER} !important;
    }}

    /* Tabla */
    table {{
        background: #ffffff !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        width: 100% !important;
    }}
    thead tr th {{
        background: {LILA} !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 10px 14px !important;
        font-size: 0.85rem !important;
    }}
    tbody tr td {{
        color: {FG} !important;
        padding: 8px 14px !important;
        border-bottom: 1px solid {BORDER} !important;
        font-size: 0.9rem !important;
    }}
    tbody tr:last-child td {{ border-bottom: none !important; }}
    tbody tr:nth-child(even) td {{ background: {LILA_LIGHT} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Funciones de métricas ────────────────────────────────────────────────────
def calcular_delta_e(img1: np.ndarray, img2: np.ndarray) -> float:
    lab1 = rgb2lab(img1 / 255.0)
    lab2 = rgb2lab(img2 / 255.0)
    med1 = np.median(lab1.reshape(-1, 3), axis=0)
    med2 = np.median(lab2.reshape(-1, 3), axis=0)
    return float(deltaE_ciede2000(med1.reshape(1, 1, 3), med2.reshape(1, 1, 3))[0, 0])

def calcular_ssim(source: np.ndarray, norm: np.ndarray) -> float:
    s_gray = np.mean(source, axis=2).astype(np.uint8)
    n_gray = np.mean(norm,   axis=2).astype(np.uint8)
    return float(structural_similarity(s_gray, n_gray, data_range=255))

def img_a_bytes(img_array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return buf.getvalue()

def fmt_tiempo(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.1f} seg"
    m = int(segundos // 60)
    s = segundos % 60
    return f"{m} min {s:.1f} seg"

def stripe(titulo: str) -> str:
    """Barra lila como encabezado de sección."""
    return f"""
    <div style="
        background: {LILA};
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 7px 16px;
        border-radius: 6px 6px 0 0;
        margin-bottom: 0;
    ">{titulo}</div>
    """


# ── Carga de normalizadores (caché) ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_reinhard():
    from tiatoolbox.tools import stainnorm
    return stainnorm.get_normalizer("Reinhard")

@st.cache_resource(show_spinner=False)
def cargar_acd():
    from acd_zheng import StainNormalizerACD
    return StainNormalizerACD(eta=0.4, gamma=0.6)

@st.cache_resource(show_spinner=False)
def cargar_scan():
    from scan_salvi import StainNormalizerSCAN
    return StainNormalizerSCAN()


# ── Encabezado ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: {LILA};
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 3px 14px rgba(98,71,170,0.18);
">
    <div style="
        background: white;
        border-radius: 50%;
        width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        font-size: 1.6rem;
    ">🔬</div>
    <div>
        <div style="color:white; font-size:1.45rem; font-weight:800; line-height:1.2;">
            Normalización de Tinciones H&amp;E
        </div>
        <div style="color:rgba(255,255,255,0.82); font-size:0.85rem; margin-top:3px;">
            Tesis de Maestría &nbsp;·&nbsp; Jorge Benítez &nbsp;·&nbsp;
            Universidad Comunera (UCOM) &nbsp;·&nbsp; Paraguay &nbsp;·&nbsp; 2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="
    background: {LILA_DARK};
    color: rgba(255,255,255,0.88);
    font-size: 0.78rem;
    text-align: center;
    padding: 5px 20px;
    border-radius: 6px;
    margin-bottom: 24px;
    letter-spacing: 0.05em;
">
    Algoritmos: Reinhard &nbsp;·&nbsp; Vahadane &nbsp;·&nbsp; ACD Zheng (η=0.4, γ=0.6) &nbsp;·&nbsp; SCAN Salvi
</div>
""", unsafe_allow_html=True)


# ── Sección 1: Cargar imágenes ────────────────────────────────────────────────
st.markdown(stripe("① Cargar imágenes"), unsafe_allow_html=True)
st.markdown(f"""
<div style="background:{LILA_LIGHT}; border:1px solid {BORDER}; border-top:none;
            border-radius:0 0 8px 8px; padding:18px 18px 6px;">
</div>
""", unsafe_allow_html=True)

col_izq, col_der = st.columns(2)

with col_izq:
    st.markdown(f"<div style='color:{LILA_DARK}; font-weight:700; font-size:0.95rem; margin-bottom:4px;'>🔬 Patch fuente</div>", unsafe_allow_html=True)
    st.caption("Imagen a normalizar (parche de lámina H&E)")
    upload_fuente = st.file_uploader(
        "fuente", type=["png", "jpg", "tif", "tiff"],
        key="fuente", label_visibility="collapsed"
    )
    if upload_fuente:
        img_prev = np.array(Image.open(upload_fuente).convert("RGB"))
        st.image(img_prev, caption="Patch fuente", use_container_width=True)

with col_der:
    st.markdown(f"<div style='color:{LILA_DARK}; font-weight:700; font-size:0.95rem; margin-bottom:4px;'>🎯 Patch referencia (target)</div>", unsafe_allow_html=True)
    st.caption("Imagen hacia la cual se normalizará el color")
    upload_target = st.file_uploader(
        "target", type=["png", "jpg", "tif", "tiff"],
        key="target", label_visibility="collapsed"
    )
    if upload_target:
        img_prev_t = np.array(Image.open(upload_target).convert("RGB"))
        st.image(img_prev_t, caption="Patch de referencia", use_container_width=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Botón normalizar ──────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    normalizar = st.button(
        "⚗️  Normalizar con los 4 algoritmos",
        use_container_width=True,
        type="primary",
    )

# ── Procesamiento ─────────────────────────────────────────────────────────────
if normalizar:

    if not upload_fuente or not upload_target:
        st.warning("⚠️  Cargá el patch fuente y el patch de referencia antes de normalizar.")
        st.stop()

    fuente_rgb = np.array(Image.open(upload_fuente).convert("RGB")).astype(np.uint8)
    target_rgb = np.array(Image.open(upload_target).convert("RGB")).astype(np.uint8)
    fuente_acd = fuente_rgb[np.newaxis, ...]
    target_acd = target_rgb[np.newaxis, ...]

    delta_e_baseline = calcular_delta_e(fuente_rgb, target_rgb)
    resultados   = {}
    tiempos_alg  = {}
    errores      = {}

    # ── Spinner ───────────────────────────────────────────────────────────────
    with st.spinner("⏳  Ejecutando algoritmos de normalización… por favor esperá"):

        t_inicio_total = time.time()

        # Reinhard
        try:
            norm_r = cargar_reinhard()
            t0 = time.time()
            norm_r.fit(target_rgb)
            img_r = norm_r.transform(fuente_rgb.copy())
            tiempos_alg["Reinhard"]  = time.time() - t0
            resultados["Reinhard"]   = img_r
        except Exception as e:
            errores["Reinhard"] = str(e)

        # Vahadane
        try:
            from wsi_normalizer import imread as wsi_imread, VahadaneNormalizer
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmpdir:
                ruta_f = os.path.join(tmpdir, "fuente.png")
                ruta_t = os.path.join(tmpdir, "target.png")
                Image.fromarray(fuente_rgb).save(ruta_f)
                Image.fromarray(target_rgb).save(ruta_t)
                norm_v = VahadaneNormalizer()
                t0 = time.time()
                norm_v.fit(wsi_imread(ruta_t))
                img_v = np.clip(norm_v.transform(wsi_imread(ruta_f)), 0, 255).astype(np.uint8)
                tiempos_alg["Vahadane"] = time.time() - t0
                resultados["Vahadane"]  = img_v
        except Exception as e:
            errores["Vahadane"] = str(e)

        # ACD (Zheng)
        try:
            norm_acd = cargar_acd()
            t0 = time.time()
            norm_acd.fit(target_acd)
            raw = norm_acd.transform(fuente_acd)
            img_acd = np.clip(raw[0], 0, 255).astype(np.uint8)
            tiempos_alg["ACD (Zheng)"] = time.time() - t0
            resultados["ACD (Zheng)"]  = img_acd
        except Exception as e:
            errores["ACD (Zheng)"] = str(e)

        # SCAN (Salvi)
        try:
            norm_scan = cargar_scan()
            t0 = time.time()
            norm_scan.fit(target_rgb)
            img_scan = norm_scan.transform(fuente_rgb.copy())
            tiempos_alg["SCAN (Salvi)"] = time.time() - t0
            resultados["SCAN (Salvi)"]  = img_scan
        except Exception as e:
            errores["SCAN (Salvi)"] = str(e)

        t_total = time.time() - t_inicio_total

    # ── Banner: tiempo total ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: #edf7f0;
        border: 1.5px solid #66bb8a;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 16px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size:1.4rem;">✅</span>
        <div>
            <span style="color:#1a5c35; font-weight:700; font-size:0.97rem;">
                Normalización completada
            </span>
            <span style="color:#2e7d50; font-size:0.88rem; margin-left:10px;">
                Tiempo total: <strong>{fmt_tiempo(t_total)}</strong>
                &nbsp;·&nbsp; {len(resultados)} algoritmos ejecutados con éxito
                {"&nbsp;·&nbsp; ⚠ " + str(len(errores)) + " con error" if errores else ""}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sección 2: Resultados ─────────────────────────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown(stripe("② Imágenes normalizadas"), unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BG2}; border:1px solid {BORDER}; border-top:none;
                border-radius:0 0 8px 8px; padding:10px 14px 4px;">
        <span style="color:{MUTED}; font-size:0.8rem;">
            ΔE CIEDE2000 baseline (sin normalizar): <strong style="color:{FG};">{delta_e_baseline:.4f}</strong>
            &nbsp;— diferencia de color fuente vs referencia antes de procesar
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    ALGORITMOS = ["Reinhard", "Vahadane", "ACD (Zheng)", "SCAN (Salvi)"]
    REFS = {
        "Reinhard":    "tiatoolbox · transferencia LAB",
        "Vahadane":    "wsi-normalizer · factorización NMF",
        "ACD (Zheng)": "acd_zheng · η=0.4, γ=0.6 (tejido peniano)",
        "SCAN (Salvi)":"scan_salvi · KMeans + proyección",
    }
    cols = st.columns(4)
    imagenes_zip = {"fuente_original.png": img_a_bytes(fuente_rgb)}

    for col, nombre in zip(cols, ALGORITMOS):
        with col:
            es_acd = nombre == "ACD (Zheng)"
            header_bg = LILA_DARK if es_acd else LILA
            star = " ⭐" if es_acd else ""

            # Header de tarjeta
            st.markdown(f"""
            <div style="
                background: {header_bg};
                border-radius: 8px 8px 0 0;
                padding: 8px 12px;
                text-align: center;
            ">
                <div style="color:white; font-weight:700; font-size:0.9rem;">{nombre}{star}</div>
                <div style="color:rgba(255,255,255,0.72); font-size:0.68rem;">{REFS[nombre]}</div>
            </div>
            """, unsafe_allow_html=True)

            if nombre in errores:
                st.markdown(f"""
                <div style="background:#fff0f0; border:1px solid #f8c8c8;
                            border-radius:0 0 8px 8px; padding:12px;
                            text-align:center; color:{RED_WARN}; font-size:0.8rem;">
                    ⚠ Error:<br>{errores[nombre]}
                </div>
                """, unsafe_allow_html=True)
                continue

            img_norm = resultados[nombre]
            delta_e  = calcular_delta_e(img_norm, target_rgb)
            ssim_val = calcular_ssim(fuente_rgb, img_norm)
            t_alg    = tiempos_alg.get(nombre, 0)

            # Imagen normalizada
            st.image(img_norm, use_container_width=True)

            # Colores de métricas
            mejora_de   = delta_e < delta_e_baseline
            color_de    = VERDE  if mejora_de else RED_WARN
            icono_de    = "↓" if mejora_de else "↑"
            color_ssim  = VERDE if ssim_val >= 0.90 else (ORANGE if ssim_val >= 0.75 else RED_WARN)

            # Card de métricas
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid {BORDER};
                border-radius: 0 0 8px 8px;
                padding: 10px;
                margin-top: -4px;
            ">
                <div style="display:flex; gap:4px; margin-bottom:8px;">
                    <div style="flex:1; background:{LILA_LIGHT}; border-radius:6px;
                                padding:8px 6px; text-align:center;">
                        <div style="color:{MUTED}; font-size:0.65rem; font-weight:700;
                                    text-transform:uppercase; letter-spacing:.04em;">
                            ΔE CIEDE2000
                        </div>
                        <div style="color:{color_de}; font-size:1.1rem; font-weight:700;
                                    font-variant-numeric:tabular-nums;">
                            {icono_de} {delta_e:.4f}
                        </div>
                        <div style="color:{MUTED}; font-size:0.63rem;">↓ menor es mejor</div>
                    </div>
                    <div style="flex:1; background:{LILA_LIGHT}; border-radius:6px;
                                padding:8px 6px; text-align:center;">
                        <div style="color:{MUTED}; font-size:0.65rem; font-weight:700;
                                    text-transform:uppercase; letter-spacing:.04em;">
                            SSIM
                        </div>
                        <div style="color:{color_ssim}; font-size:1.1rem; font-weight:700;
                                    font-variant-numeric:tabular-nums;">
                            {ssim_val:.4f}
                        </div>
                        <div style="color:{MUTED}; font-size:0.63rem;">↑ mayor es mejor</div>
                    </div>
                </div>
                <div style="text-align:center; color:{VERDE}; font-size:0.72rem; font-weight:600;">
                    ⏱ {fmt_tiempo(t_alg)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Botón descarga individual
            nombre_arch = (nombre.lower()
                           .replace(" ", "_")
                           .replace("(", "").replace(")", "")) + ".png"
            bytes_img = img_a_bytes(img_norm)
            imagenes_zip[nombre_arch] = bytes_img
            st.download_button(
                label="⬇ Descargar imagen",
                data=bytes_img,
                file_name=f"normalizado_{nombre_arch}",
                mime="image/png",
                key=f"dl_{nombre}",
                use_container_width=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Sección 3: Tabla comparativa ─────────────────────────────────────────
    st.markdown(stripe("③ Tabla comparativa de métricas"), unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:white; border:1px solid {BORDER}; border-top:none;
                border-radius:0 0 8px 8px; padding:4px 0;">
    </div>
    """, unsafe_allow_html=True)

    filas = [{"Algoritmo": "Baseline (sin normalizar)",
              "ΔE CIEDE2000": f"{delta_e_baseline:.4f}",
              "SSIM": "—", "Tiempo": "—"}]
    for nombre in ALGORITMOS:
        if nombre in resultados:
            img_n = resultados[nombre]
            filas.append({
                "Algoritmo":    nombre,
                "ΔE CIEDE2000": f"{calcular_delta_e(img_n, target_rgb):.4f}",
                "SSIM":         f"{calcular_ssim(fuente_rgb, img_n):.4f}",
                "Tiempo":       fmt_tiempo(tiempos_alg.get(nombre, 0)),
            })
        else:
            filas.append({"Algoritmo": nombre, "ΔE CIEDE2000": "ERROR",
                          "SSIM": "ERROR", "Tiempo": "—"})
    st.table(filas)

    # ── ZIP ───────────────────────────────────────────────────────────────────
    buf_zip = io.BytesIO()
    with zipfile.ZipFile(buf_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for arch, datos in imagenes_zip.items():
            zf.writestr(arch, datos)
    buf_zip.seek(0)

    _, zip_col, _ = st.columns([1, 2, 1])
    with zip_col:
        st.download_button(
            label="📦 Descargar todas las imágenes (.zip)",
            data=buf_zip.getvalue(),
            file_name="normalizadas_HE.zip",
            mime="application/zip",
            use_container_width=True,
        )

# ── Pie de página ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="
    border-top: 1px solid {BORDER};
    padding-top: 12px;
    text-align: center;
    color: {MUTED};
    font-size: 0.75rem;
">
    🔬 Tesis de Maestría · Jorge Benítez · Universidad Comunera (UCOM) · Paraguay · 2026<br>
    Algoritmos: Reinhard &nbsp;·&nbsp; Vahadane &nbsp;·&nbsp;
    ACD (Zheng et al. 2019) &nbsp;·&nbsp; SCAN (Salvi et al. 2020)
</div>
""", unsafe_allow_html=True)
