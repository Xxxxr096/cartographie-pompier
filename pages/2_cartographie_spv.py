import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.image as mpimg
import unicodedata


# --- Chargement des données ---
@st.cache_data()
def load_data():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "spv2024.csv")
    )

    df = pd.read_csv(data_path, dtype={"matricule": str})

    # Nettoyage des matricules (ex: supprimer les .0 si fichier mal encodé)
    if "matricule" in df.columns:
        df["matricule"] = df["matricule"].str.replace(".0", "", regex=False).str.strip()

    # Standardiser les noms de colonnes : minuscules, sans espace
    df.columns = df.columns.str.strip().str.lower()

    # Colonnes à convertir de 'xx,yy' → float
    cols_to_fix = [
        "poids",
        "taille",
        "imc",
        "resul ll",
        "tension artérielle systol",
        "tension artérielle diastol",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

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

# Ajoute dans le chargement si ce n’est pas fait :
if "périmètre abdominal" in df.columns:
    df["périmètre abdominal"] = (
        df["périmètre abdominal"].astype(str).str.replace(",", ".").astype(float)
    )
df["taille"] = df["taille"].astype(str).str.replace(",", ".").astype(float)
df.loc[(df["taille"] <= 100) | (df["taille"] > 250), "taille"] = None
df["taille"] = df["taille"] / 100


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
# Convertir sexe_volontaire_volontaire_volontaire_volontaire en numérique
df["sexe_volontaire_num"] = (
    df["sexe_volontaire"].str.upper().map(lambda x: 1 if x == "M" else 0).fillna(0)
)

# Convertir palier resul ll en vitesse, 0 si NaN
df["vitesse"] = df["resul ll"].map(palier_to_vitesse).fillna(0)

# Calcul VO2max de la formule avec âge, sexe_volontaire_volontaire_volontaire_volontaire et vitesse
df["vo2max"] = (
    31.025
    + 3.238 * df["vitesse"]
    - 3.248 * df["age_x"].fillna(0)
    + 6.318 * df["sexe_volontaire_num"]
)
df["vo2max"] = df["vo2max"].clip(lower=0)

# Formule de Léger (1988)
df["vo2max_leger"] = (5.857 * df["vitesse"]).fillna(0) - 19.458
df["vo2max_leger"] = df["vo2max_leger"].clip(lower=0)


st.title("Analyse de la Condition Physique et de la Santé(spv)")
with st.expander("📘 Guide d'utilisation de l'application", expanded=False):
    st.markdown(
        """
### 🧭 Guide d'utilisation

Bienvenue dans l'application d'analyse de la condition physique et de la santé.

---

#### 🔍 1. Filtres dynamiques (colonne de gauche)
Utilisez les filtres pour explorer les données :

- **Cie / UT** : sélectionnez une ou plusieurs compagnies ou unités territoriales.
- **sexe_volontaire_volontaire_volontaire_volontaire** : filtrez par genre.
- **Aptitude générale** : explorez les performances selon l'aptitude.
- **Âge** : sélection par tranche d'âge (16–29, 30–39, etc.).
- **IMC (Indice de Masse Corporelle)** : sélection par catégorie OMS (normal, surpoids...).
- **Poids** : filtrez les individus selon leur poids (kg).
- **resul ll – Paliers** : filtrez par niveau d’endurance (1 à >6).
- **Tension artérielle** :
    - Systolique (mmHg) : filtre par plage personnalisée.
    - Diastolique (mmHg) : filtre par plage personnalisée.
- **VO2max (ml/kg/min)** : filtrez selon la capacité cardio-respiratoire estimée.

⚠️ Tous les graphiques et la carte s’adaptent automatiquement à ces filtres.

---

#### 📊 2. Visualisations proposées
Plusieurs visualisations sont générées à partir des données filtrées :

- **Histogrammes simples** : poids, taille, IMC, VO2max.
- **Histogrammes empilés** :
    - IMC par niveau de resul ll.
    - resul ll par catégorie d’IMC.
- **Histogrammes et boxplots croisés** :
    - resul ll par aptitude ou exposition à l'incendie.
    - Tension artérielle systolique et diastolique (colorées selon les seuils OMS).
- **Corrélations** : carte de chaleur (heatmap) des corrélations entre indicateurs physiques.

---

#### 🗺️ 3. Carte Interactive par UT
- Affiche **l'IMC moyen** par unité territoriale (UT).
- Les cercles sont proportionnels à l'effectif par UT et colorés selon l'IMC moyen.
- Données géographiques automatiquement filtrées selon les sélections ci-dessus.

⚠️ La carte peut prendre quelques secondes à se mettre à jour. Rafraîchissez la page si nécessaire.

---

#### 💾 4. Export des données
- En bas de page, un bouton vous permet de **télécharger les données filtrées** au format CSV.

---

#### 🆘 En cas de problème
- Si un graphique ou une carte ne s'affiche pas, vérifiez que vos filtres ne sont pas trop restrictifs.
- Essayez de **réinitialiser les filtres** ou **rafraîchir la page** du navigateur.
"""
    )


# --- SIDEBAR ---
st.sidebar.header("Filtres dynamiques")
cie = st.sidebar.multiselect("Cie:", df["cie_x"].dropna().unique())
ut = st.sidebar.multiselect("UT:", df["ut_x"].dropna().unique())
sexe_volontaire_options = st.sidebar.multiselect(
    "sexe_volontaire:",
    df["sexe_volontaire"].dropna().unique(),
    default=df["sexe_volontaire"].dropna().unique(),
)
st.sidebar.markdown("**Abtitude générale**")
aptitude = st.sidebar.multiselect(
    "Aptitude Générale :",
    options=sorted(df["aptitude générale"].dropna().unique()),
    default=sorted(df["aptitude générale"].dropna().unique()),
)


# --- Filtre VO2max ---
if "vo2max" in df.columns:
    st.sidebar.markdown("**VO2max**")
    vo2_min, vo2_max = st.sidebar.slider(
        "Sélectionnez une plage de VO2max :",
        min_value=float(df["vo2max"].min()),
        max_value=float(df["vo2max"].max()),
        value=(float(df["vo2max"].min()), float(df["vo2max"].max())),
        step=1.0,
    )

# --- Filtre VO2max Léger ---
if "vo2max_leger" in df.columns:
    st.sidebar.markdown("**VO2max Léger (Formule 1988)**")
    vo2l_min, vo2l_max = st.sidebar.slider(
        "Plage VO2max (Léger 1988) :",
        min_value=float(df["vo2max_leger"].min()),
        max_value=float(df["vo2max_leger"].max()),
        value=(float(df["vo2max_leger"].min()), float(df["vo2max_leger"].max())),
        step=1.0,
    )


# Slider pour tension artérielle systolique
# Nettoyage des colonnes de tension artérielle
if "tension artérielle systol" in df.columns:
    df["tension artérielle systol"] = (
        df["tension artérielle systol"].astype(str).str.replace(",", ".").astype(float)
    )
    # Correction des valeurs aberrantes : si > 250, on divise par 10
    df.loc[df["tension artérielle systol"] > 250, "tension artérielle systol"] /= 10

if "tension artérielle diastol" in df.columns:
    df["tension artérielle diastol"] = (
        df["tension artérielle diastol"].astype(str).str.replace(",", ".").astype(float)
    )
    # Correction des valeurs aberrantes : si > 150, on divise par 10
    df.loc[df["tension artérielle diastol"] > 150, "tension artérielle diastol"] /= 10

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

# Slider pour tension artérielle diastolique
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
    [
        "Tous",
        "16 à 29",
        "30 à 39",
        "40 à 49",
        "50 à 57",
        "plus de 57",
    ],
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
# --- Filtre Tour de Taille (périmètre abdominal) ---

if "périmètre abdominal" in df.columns:
    tour_min, tour_max = st.sidebar.slider(
        "Tour de taille (cm) :",
        min_value=float(df["périmètre abdominal"].min()),
        max_value=float(df["périmètre abdominal"].max()),
        value=(
            float(df["périmètre abdominal"].min()),
            float(df["périmètre abdominal"].max()),
        ),
        step=1.0,
    )

poids_min, poids_max = st.sidebar.slider(
    "poids:", float(df["poids"].min()), float(df["poids"].max()), (0.0, 144.0)
)

# --- Application des filtres ---
df_filtered = df.copy()
if cie:
    df_filtered = df_filtered[df_filtered["cie_x"].isin(cie)]
if ut:
    df_filtered = df_filtered[df_filtered["ut_x"].isin(ut)]

if aptitude:
    df_filtered = df_filtered[df_filtered["aptitude générale"].isin(aptitude)]
df_filtered = df_filtered[
    (df_filtered["vo2max"].fillna(0) >= vo2_min)
    & (df_filtered["vo2max"].fillna(0) <= vo2_max)
]
df_filtered = df_filtered[
    (df_filtered["vo2max_leger"].fillna(0) >= vo2l_min)
    & (df_filtered["vo2max_leger"].fillna(0) <= vo2l_max)
]

df_filtered["couleur_luc"] = df_filtered["niv ll"].apply(niveau_to_couleur)
df_filtered["couleur_pompes"] = df_filtered["niv pompes"].apply(niveau_to_couleur)
df_filtered["couleur_tractions"] = df_filtered["niv tractions"].apply(niveau_to_couleur)
df_filtered["score_moyen"] = df_filtered[
    ["niv ll", "niv pompes", "niv tractions"]
].mean(axis=1)
df_filtered["couleur_globale"] = df_filtered["score_moyen"].apply(score_to_couleur)
df_filtered["tranche_age"] = df_filtered["age_x"].apply(age_to_categorie)

if "périmètre abdominal" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["périmètre abdominal"] >= tour_min)
        & (df_filtered["périmètre abdominal"] <= tour_max)
    ]

# Application des filtres de tension artérielle
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


if age_category:
    filtres_age = []
    for cat in age_category:
        if cat == "16 à 29":
            filtres_age.append(
                (df_filtered["age_x"] >= 16) & (df_filtered["age_x"] <= 29)
            )
        elif cat == "30 à 39":
            filtres_age.append(
                (df_filtered["age_x"] >= 30) & (df_filtered["age_x"] <= 39)
            )
        elif cat == "40 à 49":
            filtres_age.append(
                (df_filtered["age_x"] >= 40) & (df_filtered["age_x"] <= 49)
            )
        elif cat == "50 à 57":
            filtres_age.append(
                (df_filtered["age_x"] >= 50) & (df_filtered["age_x"] <= 57)
            )
        elif cat == "Plus de 57":
            filtres_age.append(df_filtered["age_x"] > 57)

    if filtres_age:
        df_filtered = df_filtered[pd.concat(filtres_age, axis=1).any(axis=1)]


df_filtered = df_filtered[
    (df_filtered["poids"] >= poids_min) & (df_filtered["poids"] <= poids_max)
]

st.sidebar.markdown("**Luc Léger - Paliers**")
luc_leger_categories = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs catégories de palier Luc Léger :",
    ["0", "1", "2", "3", "4", "5", "plus de 6"],
)

if sexe_volontaire_options:
    df_filtered = df_filtered[
        df_filtered["sexe_volontaire"].isin(sexe_volontaire_options)
    ]


# Application du filtre imc par classe
if imc_category:
    filtres_imc = []
    for cat in imc_category:
        if "Normal" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 18.5) & (df_filtered["imc"] <= 24.9)
            )
        elif "Surpoids" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 25.0) & (df_filtered["imc"] <= 29.9)
            )
        elif "modérée" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 30.0) & (df_filtered["imc"] <= 34.9)
            )
        elif "sévère" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 35.0) & (df_filtered["imc"] <= 39.9)
            )
        elif "massive" in cat:
            filtres_imc.append(df_filtered["imc"] >= 40.0)

    if filtres_imc:
        df_filtered = df_filtered[pd.concat(filtres_imc, axis=1).any(axis=1)]

if luc_leger_categories:
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

    if filtres_luc:
        df_filtered = df_filtered[pd.concat(filtres_luc, axis=1).any(axis=1)]

# --- VISUALISATIONS ---

IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "carte_j.jpeg")
)
UT_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ut.csv"))
CIS_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cis.csv"))


# Petites utilitaires locales (indépendantes du reste du script)
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


def _load_reference(path: str, key_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Référentiel manquant : {os.path.basename(path)}")
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )
    ref = pd.read_csv(path, sep=";")
    needed = {key_col, "x_norm", "y_norm"}
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
        df_map["ut_norm"] = _normalise_series(df_map[col_ut])
    else:
        df_map["ut_norm"] = np.nan

    if col_cis:
        df_map["cis_norm"] = _normalise_series(df_map[col_cis])
    else:
        df_map["cis_norm"] = np.nan

    # 2) Charger les référentiels (ut.csv et cis.csv)
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
                )
                if show_labels:
                    label = f"{r['ut_norm']}"
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
                )
                # {int(r['effectif'])} pers
                if show_labels:
                    label = f"{r['cis_norm']}"
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

        # Infos si des clés manquent dans les référentiels
        ut_missing = sorted(
            set(df_ut["ut_norm"].unique()) - set(ref_ut["key_norm"].dropna().unique())
        )
        cis_missing = sorted(
            set(df_cis["cis_norm"].unique())
            - set(ref_cis["key_norm"].dropna().unique())
        )
        if ut_missing:
            st.warning(
                f"UT sans coordonnées dans ut.csv : {len(ut_missing)} — ex: {ut_missing[:5]}"
            )
        if cis_missing:
            st.warning(
                f"CIS sans coordonnées dans cis.csv : {len(cis_missing)} — ex: {cis_missing[:5]}"
            )
