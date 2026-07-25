
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.base import clone
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NeuroVision MNIST Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,.16), transparent 28%),
            radial-gradient(circle at 90% 8%, rgba(14,165,233,.14), transparent 25%),
            radial-gradient(circle at 50% 95%, rgba(168,85,247,.10), transparent 25%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(30,41,59,.96), rgba(49,46,129,.91)),
            linear-gradient(90deg, #111827, #312e81);
        border: 1px solid rgba(255,255,255,.13);
        box-shadow: 0 25px 70px rgba(15,23,42,.28);
        margin-bottom: 1rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -85px;
        top: -120px;
        border-radius: 50%;
        background: rgba(56,189,248,.18);
        filter: blur(2px);
    }

    .hero h1 {
        position: relative;
        z-index: 2;
        margin: 0;
        color: white;
        font-size: clamp(2.1rem, 5vw, 4rem);
        font-weight: 800;
        letter-spacing: -0.055em;
    }

    .hero p {
        position: relative;
        z-index: 2;
        margin: .75rem 0 0;
        max-width: 850px;
        color: rgba(255,255,255,.83);
        font-size: 1.08rem;
        line-height: 1.7;
    }

    .badge {
        position: relative;
        z-index: 2;
        display: inline-block;
        margin-top: 1rem;
        padding: .42rem .72rem;
        border-radius: 999px;
        color: #e0f2fe;
        background: rgba(14,165,233,.17);
        border: 1px solid rgba(125,211,252,.30);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .04em;
    }

    .concept-card {
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 20px;
        padding: 1.05rem;
        min-height: 160px;
        background: rgba(255,255,255,.045);
        box-shadow: 0 12px 35px rgba(15,23,42,.07);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .concept-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 45px rgba(15,23,42,.13);
    }

    .concept-card h4 {
        margin: 0 0 .5rem 0;
        font-size: 1rem;
    }

    .insight {
        border-left: 4px solid #6366f1;
        border-radius: 14px;
        padding: .9rem 1rem;
        background: rgba(99,102,241,.08);
        margin: .5rem 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(148,163,184,.24);
        border-radius: 18px;
        padding: .9rem;
        background: rgba(255,255,255,.035);
        box-shadow: 0 10px 30px rgba(15,23,42,.055);
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }

    div[data-testid="stTabs"] button {
        border-radius: 12px;
        font-weight: 650;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148,163,184,.18);
    }

    .footer {
        text-align: center;
        opacity: .72;
        font-size: .82rem;
        padding-top: 2rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: .65rem;
            padding-right: .65rem;
        }
        .hero {
            padding: 1.35rem 1.1rem;
            border-radius: 22px;
        }
        div[data-testid="stTabs"] button {
            font-size: .66rem;
            padding-left: .28rem;
            padding-right: .28rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>✨ NeuroVision MNIST Studio</h1>
        <p>
            Un laboratoire intelligent de reconnaissance des chiffres manuscrits :
            exploration des données, réduction de dimension, comparaison des modèles,
            généralisation, détection d'anomalies et prédiction en temps réel.
        </p>
        <span class="badge">MASTER IA • MNIST 28×28 • MACHINE LEARNING INTERACTIF</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA
# =========================================================

@st.cache_data(show_spinner="Chargement de MNIST depuis OpenML...")
def load_mnist():
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
    )
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    return X, y


@st.cache_data(show_spinner=False)
def build_subset(
    class_mode,
    max_per_class,
    random_state=42,
):
    X_all, y_all = load_mnist()

    if class_mode == "Binaire : 0 contre 1":
        classes = [0, 1]
    else:
        classes = list(range(10))

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for digit in classes:
        indices = np.where(y_all == digit)[0]
        count = min(max_per_class, len(indices))
        selected_indices.extend(
            rng.choice(indices, size=count, replace=False)
        )

    selected_indices = np.asarray(selected_indices)
    rng.shuffle(selected_indices)

    return X_all[selected_indices], y_all[selected_indices]


def models_for_mode(binary_mode, k_value, tree_depth):
    models = {
        "Régression logistique": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(
            n_neighbors=int(k_value)
        ),
        "Arbre de décision": DecisionTreeClassifier(
            max_depth=int(tree_depth),
            random_state=42,
        ),
    }

    if binary_mode:
        models["QDA"] = QuadraticDiscriminantAnalysis(
            reg_param=0.01
        )

    return models


def preprocess_external_image(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    border = np.concatenate(
        [gray[0], gray[-1], gray[:, 0], gray[:, -1]]
    )
    if border.mean() > 127:
        gray = 255.0 - gray

    gray[gray < 20] = 0
    coords = np.argwhere(gray > 20)

    if coords.size == 0:
        empty = np.zeros((28, 28), dtype=np.float32)
        return empty, empty.reshape(1, -1)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    crop = gray[
        max(0, y_min - 5): min(gray.shape[0], y_max + 6),
        max(0, x_min - 5): min(gray.shape[1], x_max + 6),
    ]

    crop_img = Image.fromarray(
        np.clip(crop, 0, 255).astype(np.uint8)
    )

    width, height = crop_img.size
    scale = min(20 / max(width, 1), 20 / max(height, 1))
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    resized = crop_img.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), color=0)
    x_offset = (28 - new_size[0]) // 2
    y_offset = (28 - new_size[1]) // 2
    canvas.paste(resized, (x_offset, y_offset))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    if array.max() > 0:
        array = array / array.max()

    return array, array.reshape(1, -1)


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Classe prédite",
        ylabel="Classe réelle",
        title="Matrice de confusion",
    )

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    return fig


def metric_explanation(name):
    explanations = {
        "Accuracy": "Proportion totale de prédictions correctes.",
        "Precision": "Parmi les éléments prédits positifs, proportion réellement positive.",
        "Recall": "Parmi les éléments réellement positifs, proportion détectée par le modèle.",
        "F1-score": "Moyenne harmonique entre la précision et le rappel.",
        "AUC": "Capacité du modèle à séparer les classes sur plusieurs seuils.",
    }
    return explanations[name]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## ✨ NeuroVision")
    st.caption("Laboratoire MNIST interactif")
    st.divider()
    st.header("⚙️ Configuration")

    class_mode = st.selectbox(
        "Mode de classification",
        [
            "Binaire : 0 contre 1",
            "Multiclasse : 0 à 9",
        ],
    )

    binary_mode = class_mode.startswith("Binaire")

    max_per_class = st.slider(
        "Images utilisées par classe",
        min_value=500,
        max_value=5000,
        value=2000 if binary_mode else 1000,
        step=500,
        help=(
            "Un échantillon est utilisé afin de garder "
            "l'application rapide sur téléphone."
        ),
    )

    test_size = st.slider(
        "Taille du jeu de test",
        min_value=0.15,
        max_value=0.40,
        value=0.25,
        step=0.05,
    )

    k_value = st.slider(
        "K pour KNN",
        min_value=1,
        max_value=15,
        value=5,
        step=2,
    )

    tree_depth = st.slider(
        "Profondeur de l'arbre",
        min_value=2,
        max_value=15,
        value=6,
    )

    professor_mode = st.toggle(
        "🎓 Mode professeur",
        value=True,
        help="Affiche des interprétations automatiques prêtes pour l'oral.",
    )

    st.divider()
    st.caption(
        "Projet fondé sur les TP MNIST fournis : "
        "PCA, Logistic Regression, LDA, QDA, KNN, "
        "Decision Tree et One-Class SVM."
    )


# =========================================================
# PREPARE DATA
# =========================================================

try:
    X, y = build_subset(
        class_mode,
        max_per_class,
    )
except Exception as error:
    st.error(
        "Impossible de charger MNIST depuis OpenML. "
        "Vérifiez la connexion Internet du serveur Streamlit."
    )
    st.code(str(error))
    st.stop()

classes = np.unique(y)

indices = np.arange(len(X))
train_indices, test_indices = train_test_split(
    indices,
    test_size=test_size,
    random_state=42,
    stratify=y,
)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

model_dict = models_for_mode(
    binary_mode,
    k_value,
    tree_depth,
)


# =========================================================
# TABS
# =========================================================

tabs = st.tabs(
    [
        "🏠 Vue générale",
        "🧹 Workflow",
        "🗜️ ACP",
        "🤖 Modèles",
        "📊 Métriques",
        "🧪 Généralisation",
        "🚨 One-Class SVM",
        "✍️ Test intelligent",
        "🎓 Révision orale",
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Images utilisées", f"{len(X):,}")
    c2.metric("Variables", "784")
    c3.metric("Résolution", "28 × 28")
    c4.metric("Classes", len(classes))

    st.subheader("Objectif scientifique")

    st.info(
        "Reconnaître et classifier des chiffres manuscrits, "
        "réduire la dimension des images, comparer plusieurs "
        "algorithmes et vérifier la capacité de généralisation."
    )

    st.markdown(
        """
        <div class="insight">
            <b>💡 Idée forte du projet</b><br>
            L'application ne montre pas uniquement un résultat final.
            Elle explique le parcours complet de la donnée jusqu'à la décision,
            puis vérifie si le modèle reste fiable sur des images jamais vues.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Les TP intégrés")

    cards = [
        (
            "ACP / PCA",
            "Réduire 784 pixels en un nombre plus faible de composantes, "
            "compresser, reconstruire et réduire le bruit.",
        ),
        (
            "Régression logistique",
            "Construire une frontière probabiliste de classification.",
        ),
        (
            "LDA",
            "Trouver une projection linéaire qui sépare au mieux les classes.",
        ),
        (
            "QDA",
            "Autoriser une frontière quadratique lorsque les covariances diffèrent.",
        ),
        (
            "KNN",
            "Classifier une image selon ses voisins les plus proches.",
        ),
        (
            "Arbre de décision",
            "Apprendre des règles successives et interprétables.",
        ),
        (
            "One-Class SVM",
            "Apprendre une seule classe et détecter les observations différentes.",
        ),
    ]

    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for col, (title, text) in zip(
            columns,
            cards[start:start + 3],
        ):
            col.markdown(
                f"""
                <div class="concept-card">
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Exemples MNIST")
    sample_indices = np.random.default_rng(42).choice(
        len(X),
        size=min(10, len(X)),
        replace=False,
    )

    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for axis, idx in zip(axes.ravel(), sample_indices):
        axis.imshow(X[idx].reshape(28, 28), cmap="gray")
        axis.set_title(f"Classe {y[idx]}")
        axis.axis("off")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


# =========================================================
# WORKFLOW
# =========================================================

with tabs[1]:
    st.header("Workflow complet d'un projet Machine Learning")

    steps = [
        (
            "1. Collecte et compréhension",
            "Identifier la source des données, la cible, les variables "
            "et la nature du problème.",
        ),
        (
            "2. Séparation des données",
            "Créer un jeu d'entraînement et un jeu de test indépendant.",
        ),
        (
            "3. Data Cleaning",
            "Traiter les valeurs manquantes, doublons, erreurs et outliers.",
        ),
        (
            "4. Feature Selection",
            "Choisir les variables les plus utiles sans en créer de nouvelles.",
        ),
        (
            "5. Feature Extraction",
            "Créer de nouvelles caractéristiques, par exemple les composantes ACP.",
        ),
        (
            "6. Entraînement et optimisation",
            "Ajuster le modèle et ses hyperparamètres.",
        ),
        (
            "7. Évaluation et généralisation",
            "Mesurer les performances sur des données jamais vues.",
        ),
    ]

    for title, description in steps:
        with st.expander(title, expanded=title.startswith("1.")):
            st.write(description)

    st.subheader("Data Cleaning et outliers")
    st.write(
        "Pour des données tabulaires, les outliers peuvent être détectés "
        "avec le boxplot, l'IQR, le Z-score ou des méthodes comme "
        "One-Class SVM. Pour MNIST, on vérifie surtout les pixels, "
        "les images vides, mal centrées ou très bruitées."
    )

    st.subheader("Feature Selection vs Feature Extraction")
    comparison = pd.DataFrame(
        {
            "Technique": [
                "Feature Selection",
                "Feature Extraction",
            ],
            "Principe": [
                "Conserver une partie des variables originales",
                "Créer de nouvelles variables synthétiques",
            ],
            "Exemple": [
                "Corrélation, importance, RFE",
                "ACP / PCA",
            ],
            "Quand ?": [
                "Variables inutiles ou redondantes",
                "Dimension très élevée et variables corrélées",
            ],
        }
    )
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Corrélation")
    pixel_variance = X_train.var(axis=0)
    informative_pixels = np.argsort(pixel_variance)[-20:]

    correlation_df = pd.DataFrame(
        X_train[:, informative_pixels]
    ).corr()

    figure, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(correlation_df)
    fig_colorbar = figure.colorbar(image, ax=ax)
    fig_colorbar.set_label("Corrélation")
    ax.set_title("Corrélation entre 20 pixels informatifs")
    ax.set_xlabel("Pixels sélectionnés")
    ax.set_ylabel("Pixels sélectionnés")
    st.pyplot(figure)
    plt.close(figure)


# =========================================================
# PCA
# =========================================================

with tabs[2]:
    st.header("ACP / PCA appliquée à MNIST")

    st.write(
        "L'ACP transforme les 784 pixels originaux en composantes "
        "non corrélées, classées selon la quantité de variance expliquée."
    )

    n_components = st.slider(
        "Nombre de composantes",
        min_value=2,
        max_value=150,
        value=50,
        step=5,
        key="pca_components",
    )

    pca = PCA(
        n_components=n_components,
        random_state=42,
    )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()
    compression = 100 * n_components / 784

    p1, p2, p3 = st.columns(3)
    p1.metric("Dimension originale", "784")
    p2.metric("Dimension réduite", n_components)
    p3.metric("Variance conservée", f"{explained * 100:.2f}%")

    st.caption(
        f"Le taux de dimension conservée est de {compression:.2f}%."
    )

    st.subheader("Variance expliquée cumulée")
    pca_full = PCA(
        n_components=min(200, len(X_train) - 1),
        random_state=42,
    )
    pca_full.fit(X_train)
    cumulative_variance = np.cumsum(
        pca_full.explained_variance_ratio_
    )

    figure, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
    )
    ax.axhline(0.90, linestyle="--")
    ax.axhline(0.95, linestyle="--")
    ax.set(
        xlabel="Nombre de composantes",
        ylabel="Variance cumulée",
        title="Choix du nombre de composantes",
    )
    ax.grid(alpha=0.25)
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Compression et reconstruction")
    image_index = st.slider(
        "Image à reconstruire",
        0,
        len(X_test) - 1,
        0,
        key="pca_image_index",
    )

    original = X_test[image_index]
    reconstructed = pca.inverse_transform(
        X_test_pca[image_index].reshape(1, -1)
    )[0]

    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(original.reshape(28, 28), cmap="gray")
    axes[0].set_title("Image originale")
    axes[1].imshow(
        reconstructed.reshape(28, 28),
        cmap="gray",
    )
    axes[1].set_title(
        f"Reconstruction ({n_components} CP)"
    )
    for axis in axes:
        axis.axis("off")
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Réduction du bruit")
    noise_level = st.slider(
        "Niveau du bruit",
        0.05,
        0.60,
        0.25,
        0.05,
    )

    rng = np.random.default_rng(42)
    noisy = np.clip(
        original.reshape(28, 28)
        + rng.normal(0, noise_level, (28, 28)),
        0,
        1,
    )

    denoised = pca.inverse_transform(
        pca.transform(noisy.reshape(1, -1))
    )[0]

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(original.reshape(28, 28), cmap="gray")
    axes[0].set_title("Originale")
    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title("Bruitée")
    axes[2].imshow(denoised.reshape(28, 28), cmap="gray")
    axes[2].set_title("Débruitée par ACP")
    for axis in axes:
        axis.axis("off")
    st.pyplot(figure)
    plt.close(figure)


# =========================================================
# MODELS
# =========================================================

with tabs[3]:
    st.header("Comparaison interactive des modèles")

    selected_model_name = st.selectbox(
        "Choisir un modèle à analyser",
        list(model_dict.keys()),
    )

    use_pca_for_model = st.checkbox(
        "Utiliser l'ACP avant la classification",
        value=True,
        help=(
            "La réduction de dimension accélère les modèles "
            "et reproduit le workflow des TP."
        ),
    )

    model_components = st.slider(
        "Composantes utilisées par le modèle",
        10,
        100,
        40,
        10,
        disabled=not use_pca_for_model,
    )

    base_model = clone(model_dict[selected_model_name])

    if use_pca_for_model:
        estimator = Pipeline(
            [
                (
                    "pca",
                    PCA(
                        n_components=model_components,
                        random_state=42,
                    ),
                ),
                ("model", base_model),
            ]
        )
    else:
        estimator = base_model

    with st.spinner(
        f"Entraînement de {selected_model_name}..."
    ):
        estimator.fit(X_train, y_train)

    train_prediction = estimator.predict(X_train)
    test_prediction = estimator.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction,
    )
    test_accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    gap = train_accuracy - test_accuracy

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Accuracy entraînement",
        f"{train_accuracy * 100:.2f}%",
    )
    m2.metric(
        "Accuracy test",
        f"{test_accuracy * 100:.2f}%",
    )
    m3.metric(
        "Écart train-test",
        f"{gap * 100:.2f} points",
    )

    if gap > 0.08:
        st.warning(
            "Le modèle présente un risque d'overfitting : "
            "la performance d'entraînement est nettement "
            "supérieure à celle du test."
        )
    elif test_accuracy < 0.80:
        st.warning(
            "Le modèle peut être en underfitting : "
            "les performances restent faibles."
        )
    else:
        st.success(
            "Les performances train/test sont cohérentes. "
            "Le modèle montre une bonne capacité de généralisation "
            "sur cet échantillon."
        )

    st.subheader("Classement automatique")

    ranking_rows = []
    for name, candidate in model_dict.items():
        if use_pca_for_model:
            current_estimator = Pipeline(
                [
                    (
                        "pca",
                        PCA(
                            n_components=model_components,
                            random_state=42,
                        ),
                    ),
                    ("model", clone(candidate)),
                ]
            )
        else:
            current_estimator = clone(candidate)

        current_estimator.fit(X_train, y_train)
        prediction = current_estimator.predict(X_test)

        ranking_rows.append(
            {
                "Modèle": name,
                "Accuracy": accuracy_score(
                    y_test,
                    prediction,
                ),
                "Precision": precision_score(
                    y_test,
                    prediction,
                    average="binary" if binary_mode else "weighted",
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    prediction,
                    average="binary" if binary_mode else "weighted",
                    zero_division=0,
                ),
                "F1-score": f1_score(
                    y_test,
                    prediction,
                    average="binary" if binary_mode else "weighted",
                    zero_division=0,
                ),
            }
        )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        "F1-score",
        ascending=False,
    )

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]:
        ranking[column] = ranking[column].map(
            lambda value: f"{value * 100:.2f}%"
        )

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
    )

    if selected_model_name == "Arbre de décision":
        st.subheader("Visualisation de l'arbre")

        tree_for_plot = DecisionTreeClassifier(
            max_depth=min(tree_depth, 4),
            random_state=42,
        )
        pca_tree = PCA(
            n_components=min(10, X_train.shape[1]),
            random_state=42,
        )
        transformed = pca_tree.fit_transform(X_train)
        tree_for_plot.fit(transformed, y_train)

        figure, ax = plt.subplots(figsize=(14, 6))
        plot_tree(
            tree_for_plot,
            filled=True,
            max_depth=3,
            fontsize=7,
            ax=ax,
        )
        st.pyplot(figure)
        plt.close(figure)


# =========================================================
# METRICS
# =========================================================

with tabs[4]:
    st.header("Métriques d'évaluation")

    metrics_model_name = st.selectbox(
        "Modèle évalué",
        list(model_dict.keys()),
        key="metrics_model",
    )

    metrics_estimator = Pipeline(
        [
            (
                "pca",
                PCA(
                    n_components=40,
                    random_state=42,
                ),
            ),
            ("model", clone(model_dict[metrics_model_name])),
        ]
    )
    metrics_estimator.fit(X_train, y_train)
    y_pred = metrics_estimator.predict(X_test)

    average_mode = (
        "binary" if binary_mode else "weighted"
    )

    values = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            average=average_mode,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            average=average_mode,
            zero_division=0,
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            average=average_mode,
            zero_division=0,
        ),
    }

    metric_columns = st.columns(4)
    for column, (name, value) in zip(
        metric_columns,
        values.items(),
    ):
        column.metric(
            name,
            f"{value * 100:.2f}%",
        )
        column.caption(metric_explanation(name))

    cm = confusion_matrix(y_test, y_pred)
    st.pyplot(plot_confusion(cm, classes))
    plt.close("all")

    st.subheader("Rapport de classification")
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True,
    )

    if binary_mode and hasattr(
        metrics_estimator,
        "predict_proba",
    ):
        probabilities = metrics_estimator.predict_proba(
            X_test
        )[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(
            y_test,
            probabilities,
        )

        st.metric(
            "AUC",
            f"{auc_value:.4f}",
        )
        st.caption(metric_explanation("AUC"))

        figure, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {auc_value:.4f}",
        )
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Courbe ROC",
        )
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(figure)
        plt.close(figure)


# =========================================================
# GENERALIZATION
# =========================================================

with tabs[5]:
    st.header("Le modèle est-il généralisé ?")

    st.info(
        "Un modèle généralisé conserve de bonnes performances "
        "sur des données qu'il n'a jamais vues."
    )

    general_model_name = st.selectbox(
        "Modèle à tester",
        list(model_dict.keys()),
        key="general_model",
    )

    general_estimator = Pipeline(
        [
            (
                "pca",
                PCA(
                    n_components=40,
                    random_state=42,
                ),
            ),
            (
                "model",
                clone(model_dict[general_model_name]),
            ),
        ]
    )

    with st.spinner("Calcul de la validation croisée..."):
        cv_scores = cross_val_score(
            general_estimator,
            X,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )

    g1, g2, g3 = st.columns(3)
    g1.metric(
        "Accuracy CV moyenne",
        f"{cv_scores.mean() * 100:.2f}%",
    )
    g2.metric(
        "Écart-type",
        f"{cv_scores.std() * 100:.2f}%",
    )
    g3.metric(
        "Nombre de folds",
        "5",
    )

    figure, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [f"Fold {i}" for i in range(1, 6)],
        cv_scores,
    )
    ax.axhline(
        cv_scores.mean(),
        linestyle="--",
        label="Moyenne",
    )
    ax.set(
        ylabel="Accuracy",
        title="Validation croisée",
        ylim=(max(0, cv_scores.min() - 0.05), 1.0),
    )
    ax.legend()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Courbe d'apprentissage")

    sizes, train_scores, validation_scores = learning_curve(
        general_estimator,
        X,
        y,
        cv=3,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="accuracy",
        n_jobs=-1,
    )

    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        sizes,
        train_scores.mean(axis=1),
        marker="o",
        label="Entraînement",
    )
    ax.plot(
        sizes,
        validation_scores.mean(axis=1),
        marker="o",
        label="Validation",
    )
    ax.set(
        xlabel="Nombre d'images d'entraînement",
        ylabel="Accuracy",
        title="Courbe d'apprentissage",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Diagnostic intelligent")

    if cv_scores.mean() >= 0.90 and cv_scores.std() <= 0.03:
        diagnostic = (
            "Le modèle présente une généralisation solide : "
            "la performance moyenne est élevée et stable entre les folds."
        )
    elif cv_scores.std() > 0.05:
        diagnostic = (
            "La performance varie fortement entre les folds. "
            "Il faut vérifier l'échantillonnage et les hyperparamètres."
        )
    else:
        diagnostic = (
            "La généralisation reste acceptable, mais peut être améliorée "
            "par l'optimisation et davantage de données."
        )

    st.success(diagnostic)

    if professor_mode:
        st.subheader("Réponse orale conseillée")
        st.info(
            "Le modèle est généralisé lorsque ses performances restent "
            "élevées et stables sur des données jamais vues. Ici, nous "
            "le vérifions avec un jeu de test indépendant, une validation "
            "croisée à cinq folds et une courbe d'apprentissage."
        )


# =========================================================
# ONE-CLASS SVM
# =========================================================

with tabs[6]:
    st.header("One-Class SVM : détection d'une seule classe")

    target_digit = st.selectbox(
        "Chiffre considéré comme classe normale",
        list(range(10)),
    )

    if target_digit not in np.unique(y):
        st.warning(
            "Le chiffre choisi n'existe pas dans le mode binaire actuel. "
            "Passez en mode multiclasse."
        )
    else:
        normal_data = X_train[y_train == target_digit]

        sample_limit = min(1500, len(normal_data))
        normal_data = normal_data[:sample_limit]

        oneclass_components = st.slider(
            "Composantes ACP pour One-Class SVM",
            2,
            30,
            10,
        )
        nu_value = st.slider(
            "Nu : proportion d'anomalies tolérée",
            0.01,
            0.30,
            0.08,
            0.01,
        )

        pca_one = PCA(
            n_components=oneclass_components,
            random_state=42,
        )
        normal_reduced = pca_one.fit_transform(normal_data)
        test_reduced = pca_one.transform(X_test)

        oneclass = OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=nu_value,
        )
        oneclass.fit(normal_reduced)

        predictions = oneclass.predict(test_reduced)
        expected_normal = y_test == target_digit

        detected_normal = predictions == 1
        correct_detection = np.mean(
            detected_normal == expected_normal
        )

        o1, o2, o3 = st.columns(3)
        o1.metric(
            "Accuracy détection",
            f"{correct_detection * 100:.2f}%",
        )
        o2.metric(
            "Reconnu comme normal",
            int(detected_normal.sum()),
        )
        o3.metric(
            "Reconnu comme anomalie",
            int((~detected_normal).sum()),
        )

        st.write(
            "Le modèle est entraîné uniquement sur le chiffre "
            f"{target_digit}. Il apprend la région normale de cette classe, "
            "puis rejette les observations différentes."
        )

        pca_visual = PCA(
            n_components=2,
            random_state=42,
        )
        visual_data = pca_visual.fit_transform(
            X_test[: min(2500, len(X_test))]
        )
        visual_pred = predictions[
            : len(visual_data)
        ]

        figure, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            visual_data[visual_pred == 1, 0],
            visual_data[visual_pred == 1, 1],
            s=8,
            label=f"Reconnu comme {target_digit}",
        )
        ax.scatter(
            visual_data[visual_pred == -1, 0],
            visual_data[visual_pred == -1, 1],
            s=8,
            label="Anomalie",
        )
        ax.set(
            xlabel="PC1",
            ylabel="PC2",
            title="Résultats One-Class SVM",
        )
        ax.legend()
        st.pyplot(figure)
        plt.close(figure)


# =========================================================
# INTELLIGENT TEST
# =========================================================

with tabs[7]:
    st.header("Tester une image ou dessiner un chiffre")

    external_model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42,
    )

    with st.spinner(
        "Préparation du modèle de reconnaissance..."
    ):
        external_model.fit(X_train, y_train)

    external_accuracy = accuracy_score(
        y_test,
        external_model.predict(X_test),
    )

    st.caption(
        f"Accuracy du modèle externe sur le jeu de test : "
        f"{external_accuracy * 100:.2f}%"
    )

    upload_tab, drawing_tab, random_tab = st.tabs(
        [
            "📤 Importer",
            "✍️ Dessiner",
            "🎲 Exemple aléatoire",
        ]
    )

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Importer une image contenant un seul chiffre",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            processed_image, vector = preprocess_external_image(
                image
            )

            prediction = int(
                external_model.predict(vector)[0]
            )
            probabilities = external_model.predict_proba(
                vector
            )[0]
            confidence = float(probabilities.max())

            c1, c2 = st.columns(2)
            c1.image(
                image,
                caption="Image originale",
                use_container_width=True,
            )
            c2.image(
                processed_image,
                caption="Image préparée 28 × 28",
                clamp=True,
                use_container_width=True,
            )

            st.success(
                f"Prédiction : {prediction} — "
                f"Confiance : {confidence * 100:.2f}%"
            )

            probability_table = pd.DataFrame(
                {
                    "Chiffre": external_model.classes_,
                    "Probabilité": probabilities,
                }
            ).sort_values("Probabilité", ascending=False)

            st.bar_chart(
                probability_table.set_index("Chiffre"),
                use_container_width=True,
            )

    with drawing_tab:
        if not CANVAS_AVAILABLE:
            st.error(
                "Le composant de dessin n'a pas été chargé. "
                "Installez les versions compatibles indiquées dans "
                "requirements.txt, puis redémarrez l'application."
            )
        else:
            st.markdown(
                """
                <div class="insight">
                    <b>✍️ Zone de dessin</b><br>
                    Tracez un seul chiffre en blanc, au centre du carré noir.
                    Utilisez un trait continu et suffisamment large.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if "canvas_version" not in st.session_state:
                st.session_state.canvas_version = 0

            control_1, control_2 = st.columns([1, 2])
            with control_1:
                stroke_width = st.slider(
                    "Épaisseur",
                    min_value=10,
                    max_value=30,
                    value=20,
                    step=2,
                    key="canvas_stroke_width",
                )
            with control_2:
                st.write("")
                st.write("")
                if st.button(
                    "🗑️ Effacer le dessin",
                    use_container_width=True,
                ):
                    st.session_state.canvas_version += 1
                    st.rerun()

            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,1)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=300,
                height=300,
                drawing_mode="freedraw",
                display_toolbar=True,
                update_streamlit=True,
                key=f"mnist_canvas_{st.session_state.canvas_version}",
            )

            if canvas_result.image_data is None:
                st.info(
                    "La zone noire ci-dessus est la surface de dessin."
                )
            else:
                rgb = canvas_result.image_data[:, :, :3].astype(np.uint8)
                ink_mask = np.max(rgb, axis=2) > 20

                if np.count_nonzero(ink_mask) < 40:
                    st.caption(
                        "Dessinez un chiffre pour lancer la prédiction."
                    )
                else:
                    drawing = Image.fromarray(rgb)
                    processed, vector = preprocess_external_image(
                        drawing
                    )

                    prediction = int(
                        external_model.predict(vector)[0]
                    )
                    probabilities = external_model.predict_proba(
                        vector
                    )[0]
                    confidence = float(probabilities.max())

                    result_col_1, result_col_2 = st.columns([1, 2])

                    with result_col_1:
                        st.image(
                            processed,
                            caption="Image préparée 28 × 28",
                            clamp=True,
                            width=190,
                        )

                    with result_col_2:
                        st.metric(
                            "Chiffre reconnu",
                            prediction,
                        )
                        st.metric(
                            "Confiance",
                            f"{confidence * 100:.2f}%",
                        )

                    probability_table = pd.DataFrame(
                        {
                            "Chiffre": external_model.classes_,
                            "Probabilité": probabilities,
                        }
                    ).sort_values(
                        "Probabilité",
                        ascending=False,
                    )

                    st.bar_chart(
                        probability_table.set_index("Chiffre"),
                        use_container_width=True,
                    )

                    if confidence < 0.60:
                        st.warning(
                            "La confiance est faible. Redessinez le chiffre "
                            "plus grand, plus centré et avec un trait continu."
                        )
                    else:
                        st.success(
                            "Le dessin a été correctement traité par le pipeline "
                            "MNIST 28 × 28."
                        )

    with random_tab:
        random_index = st.number_input(
            "Indice de l'image",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1,
        )

        random_image = X_test[int(random_index)]
        random_prediction = int(
            external_model.predict(
                random_image.reshape(1, -1)
            )[0]
        )

        st.image(
            random_image.reshape(28, 28),
            caption=(
                f"Valeur réelle : {y_test[int(random_index)]} — "
                f"Prédiction : {random_prediction}"
            ),
            clamp=True,
            width=220,
        )


# =========================================================
# ORAL REVIEW
# =========================================================

with tabs[8]:
    st.header("Révision rapide pour l'oral")

    oral_questions = [
        (
            "Quel est l'objectif de l'ACP ?",
            "Réduire la dimension, supprimer la redondance, "
            "faciliter la visualisation et conserver un maximum de variance.",
        ),
        (
            "Feature Selection ou Feature Extraction ?",
            "La sélection conserve certaines variables originales. "
            "L'extraction crée de nouvelles variables, comme les composantes ACP.",
        ),
        (
            "Quand utiliser LDA ?",
            "Lorsque les classes peuvent être séparées par une frontière "
            "linéaire et que les matrices de covariance sont proches.",
        ),
        (
            "Quand utiliser QDA ?",
            "Lorsque chaque classe peut avoir sa propre covariance "
            "et que la frontière de décision doit être quadratique.",
        ),
        (
            "Comment choisir K dans KNN ?",
            "Tester plusieurs valeurs par validation croisée. "
            "Un K trop petit risque l'overfitting, un K trop grand l'underfitting.",
        ),
        (
            "Pourquoi limiter la profondeur d'un arbre ?",
            "Pour réduire l'overfitting et améliorer la généralisation.",
        ),
        (
            "Qu'est-ce que One-Class SVM ?",
            "Une méthode entraînée sur une seule classe normale "
            "afin de détecter les observations différentes ou anormales.",
        ),
        (
            "Le modèle est-il généralisé ?",
            "On compare les performances train/test et on utilise "
            "la validation croisée. Des résultats proches et stables "
            "indiquent une bonne généralisation.",
        ),
        (
            "Accuracy ou F1-score ?",
            "Accuracy convient aux classes équilibrées. "
            "F1-score est préférable lorsque les classes sont déséquilibrées.",
        ),
        (
            "Pourquoi MNIST 28 × 28 ?",
            "MNIST représente chaque chiffre par 784 pixels. "
            "Cette résolution offre plus d'information que Digits 8 × 8 "
            "et correspond exactement au dataset demandé dans les TP.",
        ),
    ]

    for question, answer in oral_questions:
        with st.expander(question):
            st.write(answer)

    st.subheader("Phrase finale pour la soutenance")
    st.success(
        "Cette application ne se limite pas à afficher une accuracy. "
        "Elle présente le workflow complet, compare les algorithmes, "
        "explique les métriques et vérifie la généralisation sur des "
        "données MNIST jamais vues."
    )

st.markdown(
    """
    <div class="footer">
        NeuroVision MNIST Studio • Projet académique interactif •
        Réduction de dimension, classification et généralisation
    </div>
    """,
    unsafe_allow_html=True,
)
