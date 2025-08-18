import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.image as mpimg
import unicodedata

st.set_page_config(page_title="Analyse SPP + Carte UT/CIS", layout="wide")

# =========================
# 🔧 Utilitaires ROBUSTES
# =========================


# ✅ Lecture CSV robuste: essaie plusieurs encodages & séparateurs
def read_csv_smart(path, prefer_sep=None, prefer_encoding="utf-8-sig"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    seps = (
        [prefer_sep] + [",", ";", "\t", "|", None]
        if prefer_sep
        else [",", ";", "\t", "|", None]
    )
    encs = [prefer_encoding, "utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                # drop éventuel index sauvé
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(
        f"Impossible de lire {path} avec encodages={encs} seps={seps}. Dernière erreur: {last_err}"
    )


# ✅ Normalisation forte (pour clés de jointure): enlève BOM/espaces, supprime accents, MAJUSCULE
def normalize_key(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    s = s.apply(
        lambda x: "".join(
            ch
            for ch in unicodedata.normalize("NFKD", x)
            if not unicodedata.combining(ch)
        )
    )
    return (
        s.str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("-", " ")
        .str.strip()
        .replace({"NAN": np.nan})
    )


# ✅ Conversion "xx,yy" -> float
def comma_float(s):
    return pd.to_numeric(
        pd.Series(s, dtype="object").astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


# --- Chargement des données ---
@st.cache_data()
def load_data():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "spp_final.csv")
    )

    # ✅ lecture robuste
    df = read_csv_smart(data_path)

    # Standardiser les noms de colonnes : minuscules, sans espace
    df.columns = df.columns.str.strip().str.lower()

    # Colonnes à convertir de 'xx,yy' → float (si présentes)
    cols_to_fix = [
        "poids",
        "taille",
        "imc",
        "resul ll",
        "tension artérielle systol",
        "tension artérielle diastol",
        "périmètre abdominal",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = comma_float(df[col])

    # Nettoyage taille & dérivés
    if "taille" in df.columns:
        df.loc[(df["taille"] <= 100) | (df["taille"] > 250), "taille"] = np.nan
        df["taille"] = df["taille"] / 100.0

    if "resul ll" in df.columns:
        df.loc[df["resul ll"] > 20, "resul ll"] = (
            df["resul ll"] / 10
        )  # correction 93 -> 9.3
        df["luc_leger_arrondi"] = df["resul ll"].round().astype("Int64")
        # remettre niv ll=0 si resul ll=0, si 'niv ll' existe
        if "niv ll" in df.columns:
            df.loc[df["resul ll"].fillna(0) == 0, "niv ll"] = 0

    return df


def niveau_to_couleur(niveau):
    if pd.isna(niveau):
        return "Inconnu"
    if niveau >= 3:
        return "Vert"
    elif niveau == 2:
        return "Orange"
    elif niveau == 1:
        return "Rouge"
    else:
        return "Inconnu"


def score_to_couleur(score):
    if pd.isna(score):
        return "Inconnu"
    if score >= 2.7:
        return "Vert"
    elif score >= 1.5:
        return "Orange"
    else:
        return "Rouge"


def age_to_categorie(age):
    if pd.isna(age):
        return "Inconnu"
    elif age < 30:
        return "16-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age <= 57:
        return "50-57"
    else:
        return "58+"


df = load_data()
df.columns = df.columns.str.strip().str.lower()

# =========================
# Pré-traitements
# =========================

# Tension artérielle
if "tension artérielle systol" in df.columns:
    df["tension artérielle systol"] = comma_float(df["tension artérielle systol"])
    df.loc[df["tension artérielle systol"] > 250, "tension artérielle systol"] /= 10
if "tension artérielle diastol" in df.columns:
    df["tension artérielle diastol"] = comma_float(df["tension artérielle diastol"])
    df.loc[df["tension artérielle diastol"] > 150, "tension artérielle diastol"] /= 10

# Luc Léger & VO2
palier_to_vitesse = {
    0: 8.0,
    1: 8.5,
    2: 9.0,
    3: 9.5,
    4: 10.0,
    5: 10.5,
    6: 11.0,
    7: 11.5,
    8: 12.0,
    9: 12.5,
    10: 13.0,
    11: 13.5,
    12: 14.0,
    13: 14.5,
    14: 15.0,
    15: 15.5,
    16: 16.0,
}
df["sexe_num"] = (
    df.get("sexe_spp", pd.Series(index=df.index))
    .astype(str)
    .str.upper()
    .map(lambda x: 1 if x == "M" else 0)
    .fillna(0)
)
df["vitesse"] = (
    df.get("resul ll", pd.Series(index=df.index)).map(palier_to_vitesse).fillna(0)
)
df["vo2max"] = (
    31.025
    + 3.238 * df["vitesse"]
    - 3.248 * df.get("age_pro", 0).fillna(0)
    + 6.318 * df["sexe_num"]
).clip(lower=0)
df["vo2max_leger"] = ((5.857 * df["vitesse"]).fillna(0) - 19.458).clip(lower=0)

st.title("Analyse de la Condition Physique et de la Santé (spp)")
with st.expander("📘 Guide d'utilisation de l'application", expanded=False):
    st.markdown(
        """
    ### 🧭 Guide d'utilisation
    (… contenu inchangé …)
    """
    )

# --- SIDEBAR ---
st.sidebar.header("Filtres dynamiques")
cie = st.sidebar.multiselect(
    "Cie:", df.get("cie", pd.Series(dtype=object)).dropna().unique().tolist()
)
ut = st.sidebar.multiselect(
    "UT:", df.get("ut", pd.Series(dtype=object)).dropna().unique().tolist()
)
sexe_options = st.sidebar.multiselect(
    "sexe :",
    df.get("sexe_pro", pd.Series(dtype=object)).dropna().unique().tolist(),
    default=df.get("sexe_pro", pd.Series(dtype=object)).dropna().unique().tolist(),
)
st.sidebar.markdown("**Abtitude générale**")
aptitude = st.sidebar.multiselect(
    "Aptitude Générale :",
    options=sorted(
        df.get("aptitude générale", pd.Series(dtype=object)).dropna().unique().tolist()
    ),
    default=sorted(
        df.get("aptitude générale", pd.Series(dtype=object)).dropna().unique().tolist()
    ),
)

# Filtres VO2
vo2_min, vo2_max = 0.0, float(df["vo2max"].max() if "vo2max" in df.columns else 0)
if "vo2max" in df.columns:
    st.sidebar.markdown("**VO2max**")
    vo2_min, vo2_max = st.sidebar.slider(
        "Sélectionnez une plage de VO2max :",
        min_value=float(df["vo2max"].min()),
        max_value=float(df["vo2max"].max()),
        value=(float(df["vo2max"].min()), float(df["vo2max"].max())),
        step=1.0,
    )

vo2l_min, vo2l_max = 0.0, float(
    df["vo2max_leger"].max() if "vo2max_leger" in df.columns else 0
)
if "vo2max_leger" in df.columns:
    st.sidebar.markdown("**VO2max Léger (Formule 1988)**")
    vo2l_min, vo2l_max = st.sidebar.slider(
        "Plage VO2max (Léger 1988) :",
        min_value=float(df["vo2max_leger"].min()),
        max_value=float(df["vo2max_leger"].max()),
        value=(float(df["vo2max_leger"].min()), float(df["vo2max_leger"].max())),
        step=1.0,
    )

# Sliders TA
sys_min = dia_min = 0.0
sys_max = (
    float(df["tension artérielle systol"].max())
    if "tension artérielle systol" in df.columns
    else 0.0
)
dia_max = (
    float(df["tension artérielle diastol"].max())
    if "tension artérielle diastol" in df.columns
    else 0.0
)

if "tension artérielle systol" in df.columns:
    st.sidebar.markdown("**Tension Artérielle Systolique (mmHg)**")
    sys_min, sys_max = st.sidebar.slider(
        "Sélectionnez une plage pour la tension systolique :",
        min_value=float(df["tension artérielle systol"].min()),
        max_value=float(df["tension artérielle systol"].max()),
        value=(
            float(df["tension artérielle systol"].min()),
            float(df["tension artérielle systol"].max()),
        ),
    )
if "tension artérielle diastol" in df.columns:
    st.sidebar.markdown("**Tension Artérielle Diastolique (mmHg)**")
    dia_min, dia_max = st.sidebar.slider(
        "Sélectionnez une plage pour la tension diastolique :",
        min_value=float(df["tension artérielle diastol"].min()),
        max_value=float(df["tension artérielle diastol"].max()),
        value=(
            float(df["tension artérielle diastol"].min()),
            float(df["tension artérielle diastol"].max()),
        ),
    )

st.sidebar.markdown("**Age - Catégories**")
age_category = st.sidebar.multiselect(
    "Selectionnez une catégorie d'Age : ",
    ["Tous", "16 à 29", "30 à 39", "40 à 49", "50 à 57", "plus de 57"],
)

st.sidebar.markdown("**imc - Catégories**")
imc_category = st.sidebar.multiselect(
    "Sélectionnez une catégorie d'imc :",
    [
        "Tous",
        "Normal (18.5 - 24.9)",
        "Surpoids (25.0 - 29.9)",
        "Obésité modérée (30.0 - 34.9)",
        "Obésité sévère (35.0 - 39.9)",
        "Obésité massive (>40)",
    ],
)

# Tour de taille
if "périmètre abdominal" in df.columns:
    st.sidebar.markdown("**Tour de Taille (cm)**")
    tour_min, tour_max = st.sidebar.slider(
        "Sélectionnez une plage pour le tour de taille :",
        min_value=float(df["périmètre abdominal"].min()),
        max_value=float(df["périmètre abdominal"].max()),
        value=(
            float(df["périmètre abdominal"].min()),
            float(df["périmètre abdominal"].max()),
        ),
        step=1.0,
    )

# Poids
poids_min, poids_max = st.sidebar.slider(
    "poids:",
    float(df["poids"].min()) if "poids" in df.columns else 0.0,
    float(df["poids"].max()) if "poids" in df.columns else 0.0,
    (0.0, float(df["poids"].max()) if "poids" in df.columns else 0.0),
)

# Luc Léger
st.sidebar.markdown("**Luc Léger - Paliers**")
luc_leger_categories = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs catégories de palier Luc Léger :",
    ["0", "1", "2", "3", "4", "5", "plus de 6"],
)

# --- Application des filtres ---
df_filtered = df.copy()
if "cie" in df_filtered.columns and len(cie) > 0:
    df_filtered = df_filtered[df_filtered["cie"].isin(cie)]
if "ut" in df_filtered.columns and len(ut) > 0:
    df_filtered = df_filtered[df_filtered["ut"].isin(ut)]
if "aptitude générale" in df_filtered.columns and len(aptitude) > 0:
    df_filtered = df_filtered[df_filtered["aptitude générale"].isin(aptitude)]

# VO2
if "vo2max" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["vo2max"].fillna(0) >= vo2_min)
        & (df_filtered["vo2max"].fillna(0) <= vo2_max)
    ]
if "vo2max_leger" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["vo2max_leger"].fillna(0) >= vo2l_min)
        & (df_filtered["vo2max_leger"].fillna(0) <= vo2l_max)
    ]

# Couleurs & catégories
if "niv ll" in df_filtered.columns:
    df_filtered["couleur_luc"] = df_filtered["niv ll"].apply(niveau_to_couleur)
if "niv pompes" in df_filtered.columns:
    df_filtered["couleur_pompes"] = df_filtered["niv pompes"].apply(niveau_to_couleur)
if "niv tractions" in df_filtered.columns:
    df_filtered["couleur_tractions"] = df_filtered["niv tractions"].apply(
        niveau_to_couleur
    )
for col in ["niv ll", "niv pompes", "niv tractions"]:
    if col not in df_filtered.columns:
        df_filtered[col] = np.nan
df_filtered["score_moyen"] = df_filtered[
    ["niv ll", "niv pompes", "niv tractions"]
].mean(axis=1)
df_filtered["couleur_globale"] = df_filtered["score_moyen"].apply(score_to_couleur)
if "age_pro" in df_filtered.columns:
    df_filtered["tranche_age"] = df_filtered["age_pro"].apply(age_to_categorie)

# Tour de taille
if "périmètre abdominal" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["périmètre abdominal"] >= tour_min)
        & (df_filtered["périmètre abdominal"] <= tour_max)
    ]

# Luc Léger — catégories
if len(luc_leger_categories) > 0 and "resul ll" in df_filtered.columns:
    filtres_luc = []
    for cat in luc_leger_categories:
        if cat == "0":
            filtres_luc.append(df_filtered["resul ll"] == 0)
        elif cat == "1":
            filtres_luc.append(df_filtered["resul ll"] == 1)
        elif cat == "2":
            filtres_luc.append(df_filtered["resul ll"] == 2)
        elif cat == "3":
            filtres_luc.append(df_filtered["resul ll"] == 3)
        elif cat == "4":
            filtres_luc.append(df_filtered["resul ll"] == 4)
        elif cat == "5":
            filtres_luc.append(df_filtered["resul ll"] == 5)
        elif cat == "plus de 6":
            filtres_luc.append(df_filtered["resul ll"] >= 6)
    if len(filtres_luc) > 0:
        df_filtered = df_filtered[pd.concat(filtres_luc, axis=1).any(axis=1)]

# TA
if "tension artérielle systol" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["tension artérielle systol"] >= sys_min)
        & (df_filtered["tension artérielle systol"] <= sys_max)
    ]
if "tension artérielle diastol" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["tension artérielle diastol"] >= dia_min)
        & (df_filtered["tension artérielle diastol"] <= dia_max)
    ]

# Poids
if "poids" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["poids"] >= poids_min) & (df_filtered["poids"] <= poids_max)
    ]

# Sexe
if "sexe_spp" in df_filtered.columns and len(sexe_options) > 0:
    df_filtered = df_filtered[df_filtered["sexe_spp"].isin(sexe_options)]

# =============== Carte UT (jaune) + CIS (rouge) — SANS UPLOAD ===============

IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "carte_j.jpeg")
)
UT_REF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ut_spp_c.csv")
)
CIS_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cis.csv"))

# ✅ normalisation déjà définie: normalize_key


# ✅ Utilitaires référentiels avec lecture robuste et messages clairs
def _load_reference(path: str, key_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Référentiel manquant : {os.path.basename(path)}")
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )

    # Lecture robuste (ne force plus sep=';')
    ref = read_csv_smart(path)

    # Harmonise colonnes en minuscules, trim
    ref.columns = ref.columns.str.strip()

    needed = {key_col, "x_norm", "y_norm"}
    if not needed.issubset(set(ref.columns)):
        st.error(
            f"{os.path.basename(path)} doit contenir les colonnes : {key_col}, x_norm, y_norm (offset_x/offset_y optionnels). "
            f"Colonnes trouvées: {list(ref.columns)}"
        )
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )

    # Normalise la clé & garde label original
    ref["key_norm"] = normalize_key(ref[key_col])
    ref["label"] = ref[key_col]

    # Types numériques
    for c in ("x_norm", "y_norm", "offset_x", "offset_y"):
        if c not in ref.columns:
            ref[c] = 0
        ref[c] = pd.to_numeric(ref[c], errors="coerce")

    return ref[["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]]


# 1) Normaliser les colonnes UT / CIS dans df_filtered (déjà construit plus haut)
df_map = df_filtered.copy()
df_map.columns = df_map.columns.str.strip().str.lower()

col_ut = next(
    (c for c in ["ut", "compagnie", "unite_territoriale"] if c in df_map.columns), None
)
col_cis = next(
    (c for c in ["cis", "centre", "centre_cis", "nom_cis"] if c in df_map.columns), None
)

if not col_ut and not col_cis:
    st.warning(
        "Aucune colonne UT/Compagnie ni CIS trouvée dans les données — impossible d’afficher la carte."
    )
else:
    if col_ut:
        df_map["ut_norm"] = normalize_key(df_map[col_ut])
    else:
        df_map["ut_norm"] = np.nan

    if col_cis:
        df_map["cis_norm"] = normalize_key(df_map[col_cis])
    else:
        df_map["cis_norm"] = np.nan

    # 2) Charger les référentiels (ut.csv et cis.csv) — robustes
    ref_ut = _load_reference(UT_REF_PATH, "ut")
    ref_cis = _load_reference(CIS_REF_PATH, "cis")

    # 3) Joindre pour récupérer x_norm / y_norm
    df_ut = df_map.dropna(subset=["ut_norm"]).merge(
        ref_ut, left_on="ut_norm", right_on="key_norm", how="left"
    )
    df_cis = df_map.dropna(subset=["cis_norm"]).merge(
        ref_cis, left_on="cis_norm", right_on="key_norm", how="left"
    )

    # 4) Agréger (effectif + IMC moyen si présent)
    df_ut["imc"] = pd.to_numeric(df_ut.get("imc", np.nan), errors="coerce")
    df_cis["imc"] = pd.to_numeric(df_cis.get("imc", np.nan), errors="coerce")

    agg_ut = df_ut.groupby(
        ["ut_norm", "x_norm", "y_norm", "offset_x", "offset_y"],
        dropna=False,
        as_index=False,
    ).agg(effectif=("ut_norm", "count"), imc_moyen=("imc", "mean"))
    agg_cis = df_cis.groupby(
        ["cis_norm", "x_norm", "y_norm", "offset_x", "offset_y"],
        dropna=False,
        as_index=False,
    ).agg(effectif=("cis_norm", "count"), imc_moyen=("imc", "mean"))

    agg_ut_ok = agg_ut.dropna(subset=["x_norm", "y_norm"])
    agg_cis_ok = agg_cis.dropna(subset=["x_norm", "y_norm"])

    st.sidebar.markdown("**Affichage sur la carte**")
    affichage_points = st.sidebar.radio(
        "Sélectionnez les points à afficher :",
        ("UT uniquement", "CIS uniquement", "UT + CIS"),
        index=2,
    )

    # 5) Afficher la carte
    st.subheader("🗺️ Carte — UT (jaune) et CIS (rouge)")
    if not os.path.exists(IMAGE_PATH):
        st.error(f"Image non trouvée : {IMAGE_PATH}")
    else:
        img = mpimg.imread(IMAGE_PATH)
        H, W = img.shape[0], img.shape[1]

        c1, c2 = st.columns([1, 1])
        with c1:
            ms = st.slider("Taille des points", 4, 20, 10)
            fs = st.slider("Taille du texte", 6, 16, 9)
        with c2:
            show_labels = st.checkbox("Afficher les labels", True)
            show_imc = st.checkbox("Afficher l'IMC moyen", False)

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(img)
        ax.axis("off")

        # UT = jaune
        if affichage_points in ("UT uniquement", "UT + CIS"):
            for _, r in agg_ut_ok.iterrows():
                x = float(r["x_norm"]) * W
                y = float(r["y_norm"]) * H
                dx = float(r.get("offset_x", 0) or 0)
                dy = float(r.get("offset_y", 0) or 0)
                ax.plot(
                    x,
                    y,
                    "o",
                    markersize=ms,
                    color="yellow",
                    markeredgecolor="black",
                    markeredgewidth=1.2,
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
                    )

        # CIS = rouge
        if affichage_points in ("CIS uniquement", "UT + CIS"):
            for _, r in agg_cis_ok.iterrows():
                x = float(r["x_norm"]) * W
                y = float(r["y_norm"]) * H
                dx = float(r.get("offset_x", 0) or 0)
                dy = float(r.get("offset_y", 0) or 0)
                ax.plot(
                    x,
                    y,
                    "o",
                    markersize=ms,
                    color="red",
                    markeredgecolor="white",
                    markeredgewidth=1.2,
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
                    )

        st.pyplot(fig, use_container_width=True)

        # ✅ Diagnostics utiles: si des clés ne matchent pas, affiche des exemples
        ut_missing = sorted(
            set(df_map["ut_norm"].dropna().unique())
            - set(ref_ut["key_norm"].dropna().unique())
        )
        cis_missing = sorted(
            set(df_map["cis_norm"].dropna().unique())
            - set(ref_cis["key_norm"].dropna().unique())
        )
        if ut_missing:
            st.warning(
                f"UT sans coordonnées dans ut_spp_c.csv : {len(ut_missing)} — ex: {ut_missing[:5]}"
            )
        if cis_missing:
            st.warning(
                f"CIS sans coordonnées dans cis.csv : {len(cis_missing)} — ex: {cis_missing[:5]}"
            )
