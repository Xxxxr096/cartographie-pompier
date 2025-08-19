# -*- coding: utf-8 -*-
"""
Carte Streamlit pour afficher :
- UT / CIS en mode agrégé
- EAP en mode individuel, avec légende (EAP + Code)

Points forts :
- Détection AUTOMATIQUE du format des coordonnées (norm=0..1, percent=0..100, px)
- Lecture CSV robuste (séparateur ; ou ,)
- Normalisation des libellés (accents, casse, tirets, espaces)
- Panneau Diagnostic (clés non trouvées) + Debug coordonnées
- zorder élevé pour que les points/labels apparaissent toujours au-dessus

Pré-requis : streamlit, matplotlib, pandas, numpy
"""

import os
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st

# -----------------------------------------------------------------------------
# Paths des fichiers (placés un niveau au-dessus du script)
# -----------------------------------------------------------------------------
IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "carte_j.jpeg")
)
UT_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ut.csv"))
CIS_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cis.csv"))
EAP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "eap_clean.csv")
)


# -----------------------------------------------------------------------------
# Utilitaires
# -----------------------------------------------------------------------------
def _normalise_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .apply(
            lambda x: unicodedata.normalize("NFKD", x)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("-", " ")
        .str.strip()
        .replace({"NAN": np.nan})
    )


def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path)
    return df


def _load_reference(path: str, key_cols) -> pd.DataFrame:
    """Charge un référentiel (UT/CIS) en détectant la colonne clé automatiquement."""
    if not os.path.exists(path):
        st.error(f"Référentiel manquant : {os.path.basename(path)}")
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )

    ref = _read_csv_robust(path)
    ref.columns = ref.columns.str.strip().str.lower()

    candidates = [key_cols] if isinstance(key_cols, str) else list(key_cols)
    key_col = next((c for c in candidates if c in ref.columns), None)
    if key_col is None:
        st.error(
            f"{os.path.basename(path)} ne contient aucune des colonnes clés attendues : {candidates}"
        )
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )

    needed = {"x_norm", "y_norm"}
    if not needed.issubset(set(ref.columns)):
        st.error(
            f"{os.path.basename(path)} doit contenir les colonnes : {key_col}, x_norm, y_norm (offset_x/offset_y optionnels)."
        )
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )

    ref["key_norm"] = _normalise_series(ref[key_col])
    ref["label"] = ref[key_col]
    for c in ("x_norm", "y_norm", "offset_x", "offset_y"):
        if c not in ref.columns:
            ref[c] = 0
        ref[c] = pd.to_numeric(ref[c], errors="coerce").fillna(0.0)
    return ref[["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]]


def _load_eap(path: str):
    if not os.path.exists(path):
        st.warning(
            "Fichier EAP introuvable – le mode 'EAP (individuel)' sera désactivé."
        )
        return pd.DataFrame(), None, None, None, None, None
    eap = _read_csv_robust(path)
    eap.columns = eap.columns.str.strip().str.lower()

    col_eap = next(
        (c for c in ["eap", "libelle_eap", "nom_eap"] if c in eap.columns), None
    )
    col_code = next(
        (c for c in ["code", "code_eap", "code_eap2"] if c in eap.columns), None
    )
    col_ut = next(
        (c for c in ["ut", "compagnie", "unite_territoriale"] if c in eap.columns), None
    )
    col_cis = next(
        (c for c in ["cis", "centre", "centre_cis", "nom_cis"] if c in eap.columns),
        None,
    )
    col_imc = next((c for c in ["imc", "imc_moyen"] if c in eap.columns), None)

    if col_ut:
        eap["ut_norm"] = _normalise_series(eap[col_ut])
    if col_cis:
        eap["cis_norm"] = _normalise_series(eap[col_cis])
    if col_eap is None:
        st.warning(
            "Colonne EAP absente (eap/libelle_eap/nom_eap). Les labels EAP seront 'EAP'."
        )
        eap["eap"] = "EAP"
        col_eap = "eap"
    if col_code is None:
        st.warning(
            "Colonne Code absente (code/code_eap). Les labels n'afficheront pas de code."
        )
        eap["code"] = ""
        col_code = "code"
    if col_imc:
        eap[col_imc] = pd.to_numeric(eap[col_imc], errors="coerce")

    return eap, col_eap, col_code, col_ut, col_cis, col_imc


def _infer_coord_scale(df: pd.DataFrame) -> str:
    """Détecte le format des coordonnées: 'norm'(0..1), 'percent'(0..100) ou 'px'."""
    if df is None or df.empty or not {"x_norm", "y_norm"}.issubset(df.columns):
        return "norm"
    x = pd.to_numeric(df["x_norm"], errors="coerce")
    y = pd.to_numeric(df["y_norm"], errors="coerce")
    q = np.nanmax([np.nanquantile(x, 0.98), np.nanquantile(y, 0.98)])
    if not np.isfinite(q):
        return "norm"
    if q <= 1.5:
        return "norm"
    if q <= 100:
        return "percent"
    return "px"


def _to_plot_xy(W: float, H: float, x_val: float, y_val: float, scale: str):
    if pd.isna(x_val) or pd.isna(y_val):
        return np.nan, np.nan
    if scale == "norm":
        X, Y = float(x_val) * W, float(y_val) * H
    elif scale == "percent":
        X, Y = (float(x_val) / 100.0) * W, (float(y_val) / 100.0) * H
    else:  # 'px'
        X, Y = float(x_val), float(y_val)
    # Clamp dans l'image
    X = max(0, min(W - 1, X))
    Y = max(0, min(H - 1, Y))
    return X, Y


# -----------------------------------------------------------------------------
# UI – Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("**Affichage sur la carte**")
mode_affichage = st.sidebar.radio(
    "Sélectionnez le mode d'affichage :",
    (
        "UT uniquement (agrégé)",
        "CIS uniquement (agrégé)",
        "UT + CIS (agrégé)",
        "EAP (individuel)",
    ),
    index=2,
)

# -----------------------------------------------------------------------------
# Chargements référentiels + EAP
# -----------------------------------------------------------------------------
ref_ut = _load_reference(
    UT_REF_PATH, ["ut", "compagnie", "unite_territoriale", "libelle_ut", "nom_ut"]
)
ref_cis = _load_reference(
    CIS_REF_PATH, ["cis", "centre", "centre_cis", "nom_cis", "nom", "label"]
)
eap_df, col_eap, col_code, col_ut, col_cis, col_imc = _load_eap(EAP_PATH)

# Base pour agrégations
base_df = eap_df.copy()
base_df.columns = base_df.columns.str.strip().str.lower()
base_df["ut_norm"] = _normalise_series(base_df[col_ut]) if col_ut else np.nan
base_df["cis_norm"] = _normalise_series(base_df[col_cis]) if col_cis else np.nan

# Jointures
join_ut = base_df.dropna(subset=["ut_norm"]).merge(
    ref_ut, left_on="ut_norm", right_on="key_norm", how="left"
)
join_cis = base_df.dropna(subset=["cis_norm"]).merge(
    ref_cis, left_on="cis_norm", right_on="key_norm", how="left"
)

# IMC
if col_imc:
    join_ut["imc"] = pd.to_numeric(join_ut[col_imc], errors="coerce")
    join_cis["imc"] = pd.to_numeric(join_cis[col_imc], errors="coerce")
else:
    join_ut["imc"] = np.nan
    join_cis["imc"] = np.nan

# Agrégations
agg_ut = join_ut.groupby(
    ["ut_norm", "x_norm", "y_norm", "offset_x", "offset_y"],
    dropna=False,
    as_index=False,
).agg(effectif=("ut_norm", "count"), imc_moyen=("imc", "mean"))
agg_cis = join_cis.groupby(
    ["cis_norm", "x_norm", "y_norm", "offset_x", "offset_y"],
    dropna=False,
    as_index=False,
).agg(effectif=("cis_norm", "count"), imc_moyen=("imc", "mean"))

agg_ut_ok = agg_ut.dropna(subset=["x_norm", "y_norm"]) if not agg_ut.empty else agg_ut
agg_cis_ok = (
    agg_cis.dropna(subset=["x_norm", "y_norm"]) if not agg_cis.empty else agg_cis
)

# -----------------------------------------------------------------------------
# Image de fond + détection d'échelle
# -----------------------------------------------------------------------------
if not os.path.exists(IMAGE_PATH):
    st.error(f"Image non trouvée : {IMAGE_PATH}")
    st.stop()

img = mpimg.imread(IMAGE_PATH)
H, W = img.shape[0], img.shape[1]

ut_scale = _infer_coord_scale(ref_ut)
cis_scale = _infer_coord_scale(ref_cis)

# -----------------------------------------------------------------------------
# Paramètres d'affichage
# -----------------------------------------------------------------------------
c1, c2 = st.columns([1, 1])
with c1:
    ms = st.slider("Taille des points", 4, 20, 10)
    fs = st.slider("Taille du texte", 6, 16, 9)
with c2:
    show_labels = st.checkbox("Afficher les labels", True)
    show_imc = st.checkbox("Afficher l'IMC moyen (si agrégé)", False)
    prefer_ref = st.selectbox(
        "Pour l'affichage EAP, positionner les points par :",
        ["UT", "CIS"],
        index=0,
        help="Choix utilisé uniquement en mode 'EAP (individuel)'.",
    )

# -----------------------------------------------------------------------------
# Tracé
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(img, zorder=0)
ax.axis("off")

# UT agrégé
if (
    mode_affichage in ("UT uniquement (agrégé)", "UT + CIS (agrégé)")
    and not agg_ut_ok.empty
):
    for _, r in agg_ut_ok.iterrows():
        x, y = _to_plot_xy(W, H, r["x_norm"], r["y_norm"], ut_scale)
        dx = float(r.get("offset_x", 0))
        dy = float(r.get("offset_y", 0))
        ax.plot(
            x,
            y,
            "o",
            markersize=ms,
            color="yellow",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=10,
        )
        if show_labels:
            label = f"{r['ut_norm']}\n{int(r['effectif'])} pers"
            if show_imc and pd.notna(r["imc_moyen"]):
                label += f"\nIMC {r['imc_moyen']:.1f}"
            ax.text(
                x + dx,
                y + dy,
                label,
                fontsize=fs,
                color="black",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
                zorder=20,
            )

# CIS agrégé
if (
    mode_affichage in ("CIS uniquement (agrégé)", "UT + CIS (agrégé)")
    and not agg_cis_ok.empty
):
    for _, r in agg_cis_ok.iterrows():
        x, y = _to_plot_xy(W, H, r["x_norm"], r["y_norm"], cis_scale)
        dx = float(r.get("offset_x", 0))
        dy = float(r.get("offset_y", 0))
        ax.plot(
            x,
            y,
            "o",
            markersize=ms,
            color="red",
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=12,
        )
        if show_labels:
            label = f"{r['cis_norm']}\n{int(r['effectif'])} pers"
            if show_imc and pd.notna(r["imc_moyen"]):
                label += f"\nIMC {r['imc_moyen']:.1f}"
            ax.text(
                x + dx,
                y + dy,
                label,
                fontsize=fs,
                color="white",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="black",
                    alpha=0.75,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                ),
                zorder=20,
            )

# EAP individuel
if mode_affichage == "EAP (individuel)":
    if eap_df.empty:
        st.warning("Pas de données EAP à afficher.")
    else:
        working = join_ut if prefer_ref == "UT" else join_cis
        if working.empty:
            st.warning("Aucune correspondance entre EAP et le référentiel choisi.")
        else:
            working_ok = working.dropna(subset=["x_norm", "y_norm"]).copy()
            if working_ok.empty:
                st.warning(
                    "Les lignes EAP n'ont pas de coordonnées dans le référentiel choisi."
                )
            else:
                for _, r in working_ok.iterrows():
                    x, y = _to_plot_xy(
                        W,
                        H,
                        r["x_norm"],
                        r["y_norm"],
                        ut_scale if prefer_ref == "UT" else cis_scale,
                    )
                    dx = float(r.get("offset_x", 0))
                    dy = float(r.get("offset_y", 0))
                    ax.plot(
                        x,
                        y,
                        "o",
                        markersize=ms,
                        color="red",
                        markeredgecolor="white",
                        markeredgewidth=1.2,
                        zorder=12,
                    )
                    if show_labels:
                        eap_val = str(r.get(col_eap, "")).strip() if col_eap else "EAP"
                        code_val = str(r.get(col_code, "")).strip() if col_code else ""
                        label = (
                            eap_val if not code_val else f"{eap_val}\nCode {code_val}"
                        )
                        if show_imc and col_imc and pd.notna(r.get("imc", np.nan)):
                            label += f"\nIMC {float(r['imc']):.1f}"
                        ax.text(
                            x + dx,
                            y + dy,
                            label,
                            fontsize=fs,
                            color="white",
                            ha="center",
                            va="center",
                            bbox=dict(
                                facecolor="black",
                                alpha=0.75,
                                edgecolor="none",
                                boxstyle="round,pad=0.2",
                            ),
                            zorder=20,
                        )

st.subheader("🗺️ Carte — UT (jaune), CIS (rouge) et EAP (individuel)")

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
with st.expander("Diagnostic clés UT/CIS"):
    st.write("**UT connus dans ut.csv**:", len(ref_ut))
    st.write("**CIS connus dans cis.csv**:", len(ref_cis))
    if not base_df.empty:
        ut_missing = sorted(
            set(base_df["ut_norm"].dropna().unique())
            - set(ref_ut["key_norm"].dropna().unique())
        )
        cis_missing = sorted(
            set(base_df["cis_norm"].dropna().unique())
            - set(ref_cis["key_norm"].dropna().unique())
        )
        if ut_missing:
            st.info(f"UT non trouvés (exemple) : {ut_missing[:10]}")
        if cis_missing:
            st.info(f"CIS non trouvés (exemple) : {cis_missing[:10]}")

# Debug coordonnées (échantillon)
if (
    mode_affichage in ("CIS uniquement (agrégé)", "UT + CIS (agrégé)")
    and not agg_cis_ok.empty
):
    xs, ys = [], []
    for _, r in agg_cis_ok.head(20).iterrows():
        X, Y = _to_plot_xy(W, H, r["x_norm"], r["y_norm"], cis_scale)
        xs.append(X)
        ys.append(Y)
    if xs and ys:
        st.caption(
            f"Debug CIS → scale={cis_scale} | X:[{min(xs):.1f},{max(xs):.1f}] Y:[{min(ys):.1f},{max(ys):.1f}] | HxW=({H}x{W})"
        )

plotted_ut = (
    len(agg_ut_ok)
    if mode_affichage in ("UT uniquement (agrégé)", "UT + CIS (agrégé)")
    else 0
)
plotted_cis = (
    len(agg_cis_ok)
    if mode_affichage in ("CIS uniquement (agrégé)", "UT + CIS (agrégé)")
    else 0
)
st.caption(f"Points UT tracés: {plotted_ut} | Points CIS tracés: {plotted_cis}")

st.pyplot(plt.gcf(), use_container_width=True)
