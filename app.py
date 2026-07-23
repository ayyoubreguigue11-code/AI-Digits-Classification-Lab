
import time
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM, SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

CANVAS_IMPORT_ERROR = None

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception as canvas_error:
    CANVAS_AVAILABLE = False
    CANVAS_IMPORT_ERROR = str(canvas_error)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Digits Classification Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: .25rem;
    }
    .subtitle {
        font-size: clamp(.95rem, 2vw, 1.1rem);
        opacity: .85;
        line-height: 1.6;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, .25);
        border-radius: 14px;
        padding: .8rem;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: .7rem;
            padding-right: .7rem;
        }
        div[data-testid="stTabs"] button {
            font-size: .78rem;
            padding-left: .35rem;
            padding-right: .35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🤖 AI Digits Classification Lab</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="subtitle">
        Plateforme académique de réduction de dimension, classification,
        évaluation, optimisation et interprétation de modèles.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# =========================================================
# HELPERS
# =========================================================

@st.cache_data(show_spinner=False)
def load_dataset(class_mode: str):
    digits = load_digits()
    X_all = digits.data.astype(np.float64) / 16.0
    y_all = digits.target.astype(int)

    if class_mode == "Binary: 0 vs 1":
        mask = np.isin(y_all, [0, 1])
        X_all = X_all[mask]
        y_all = y_all[mask]

    return X_all, y_all


def build_models(n_classes: int):
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=3000,
            solver="lbfgs",
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    # QDA is kept for the binary case because it is the most stable there.
    if n_classes == 2:
        models["QDA"] = QuadraticDiscriminantAnalysis(reg_param=1e-3)

    return models


@st.cache_data(show_spinner=False)
def compute_embeddings(
    X_data,
    umap_neighbors,
    umap_min_dist,
    tsne_perplexity,
):
    pca_model = PCA(n_components=2, random_state=42)
    X_pca_data = pca_model.fit_transform(X_data)

    # Compatibility between recent and older scikit-learn versions.
    tsne_kwargs = dict(
        n_components=2,
        perplexity=min(float(tsne_perplexity), len(X_data) - 1),
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    try:
        tsne_model = TSNE(max_iter=1000, **tsne_kwargs)
    except TypeError:
        tsne_model = TSNE(n_iter=1000, **tsne_kwargs)

    X_tsne_data = tsne_model.fit_transform(X_data)

    X_umap_data = None
    if UMAP_AVAILABLE:
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            metric="euclidean",
            random_state=42,
        )
        X_umap_data = umap_model.fit_transform(X_data)

    return pca_model, X_pca_data, X_umap_data, X_tsne_data


def prepare_image_for_digits(image: Image.Image):
    """
    Prépare une image externe pour le dataset sklearn Digits.

    Étapes:
    1. conversion en niveaux de gris;
    2. inversion si le fond est clair;
    3. suppression du fond vide;
    4. recadrage autour du chiffre;
    5. conservation des proportions;
    6. centrage dans une image 8 × 8;
    7. normalisation entre 0 et 1.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    # Le dataset Digits utilise un chiffre clair sur un fond sombre.
    # Si l'image possède un fond clair, on l'inverse.
    border_pixels = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )

    if float(np.mean(border_pixels)) > 127:
        gray = 255.0 - gray

    # Supprimer un éventuel bruit très faible.
    gray[gray < 20] = 0

    coordinates = np.argwhere(gray > 20)

    if coordinates.size == 0:
        empty = np.zeros((8, 8), dtype=np.float64)
        return empty, empty.reshape(1, -1)

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    cropped = gray[
        max(0, y_min - 4): min(gray.shape[0], y_max + 5),
        max(0, x_min - 4): min(gray.shape[1], x_max + 5),
    ]

    cropped_image = Image.fromarray(
        np.clip(cropped, 0, 255).astype(np.uint8)
    )

    # Redimensionner le chiffre dans une zone maximale de 6 × 6
    # afin de garder une marge autour de lui.
    width, height = cropped_image.size
    scale = min(6.0 / max(width, 1), 6.0 / max(height, 1))

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas_8x8 = Image.new("L", (8, 8), color=0)

    offset_x = (8 - new_width) // 2
    offset_y = (8 - new_height) // 2

    canvas_8x8.paste(
        resized,
        (offset_x, offset_y),
    )

    array = np.asarray(canvas_8x8, dtype=np.float64)

    # Renforcer légèrement les traits faibles sans saturer l'image.
    if array.max() > 0:
        array = array / array.max()
        array = np.power(array, 0.85)

    array = np.clip(array, 0.0, 1.0)

    return array, array.reshape(1, -1)


def prediction_confidence(classifier, sample_2d):
    if hasattr(classifier, "predict_proba"):
        return float(np.max(classifier.predict_proba(sample_2d)))

    if hasattr(classifier, "decision_function"):
        values = np.asarray(classifier.decision_function(sample_2d))
        if values.ndim == 1:
            value = abs(float(values[0]))
        else:
            value = float(np.max(values[0]))
        return float(1.0 / (1.0 + np.exp(-value)))

    return None


def plot_embedding(embedding, y_values, title, x_label, y_label, selected_index):
    fig, ax = plt.subplots(figsize=(9, 6))

    for class_value in np.unique(y_values):
        class_mask = y_values == class_value
        ax.scatter(
            embedding[class_mask, 0],
            embedding[class_mask, 1],
            s=24,
            alpha=.72,
            label=f"Digit {class_value}",
        )

    ax.scatter(
        embedding[selected_index, 0],
        embedding[selected_index, 1],
        s=260,
        marker="X",
        label="Selected image",
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=.2)
    ax.legend(ncol=2)
    return fig


def get_grid_and_best_model(model_name, base_model, X_train, y_train):
    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "LDA": {
            "solver": ["svd", "lsqr"],
        },
        "QDA": {
            "reg_param": [0.0, 0.001, 0.01, 0.1],
        },
        "Decision Tree": {
            "max_depth": [2, 3, 4, 5, 8, None],
            "min_samples_split": [2, 5, 10],
        },
        "KNN": {
            "n_neighbors": [1, 3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
    }

    grid = GridSearchCV(
        estimator=clone(base_model),
        param_grid=param_grids[model_name],
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def model_complexity_table():
    return pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "LDA",
                "QDA",
                "Decision Tree",
                "KNN",
            ],
            "Training complexity (approx.)": [
                "Depends on the optimizer and iterations",
                "Approximately O(nd² + d³)",
                "Approximately O(nd² + Cd³)",
                "Approximately O(nd log n)",
                "Very low: model memorization",
            ],
            "Prediction complexity (approx.)": [
                "O(d)",
                "O(Cd)",
                "O(Cd²)",
                "O(tree depth)",
                "O(nd)",
            ],
            "Main strength": [
                "Simple probabilistic baseline",
                "Strong linear separation",
                "Nonlinear quadratic boundary",
                "Interpretable rules",
                "Flexible local classification",
            ],
        }
    )


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("⚙️ Control Panel")

class_mode = st.sidebar.selectbox(
    "Classification task",
    ["Binary: 0 vs 1", "Multiclass: 0 to 9"],
)

X, y = load_dataset(class_mode)
classes = np.unique(y)
n_classes = len(classes)
models = build_models(n_classes)

selected_model_name = st.sidebar.selectbox(
    "Model",
    list(models.keys()),
)

test_size = st.sidebar.slider(
    "Test size",
    min_value=.15,
    max_value=.40,
    value=.25,
    step=.05,
)

selected_index = st.sidebar.slider(
    "Selected image",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Nonlinear projection")

umap_neighbors = st.sidebar.slider(
    "UMAP: n_neighbors",
    min_value=5,
    max_value=50,
    value=15,
)

umap_min_dist = st.sidebar.slider(
    "UMAP: min_dist",
    min_value=0.0,
    max_value=0.99,
    value=0.10,
    step=0.05,
)

max_perplexity = max(5, min(50, len(X) - 1))
default_perplexity = min(30, max_perplexity)

tsne_perplexity = st.sidebar.slider(
    "t-SNE: perplexity",
    min_value=5,
    max_value=max_perplexity,
    value=default_perplexity,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Samples: {len(X)} | Features: {X.shape[1]} | Classes: {n_classes}"
)


# =========================================================
# EMBEDDINGS AND TRAINING
# =========================================================

with st.spinner("Calcul des projections PCA, UMAP et t-SNE..."):
    pca, X_pca, X_umap, X_tsne = compute_embeddings(
        X,
        umap_neighbors,
        umap_min_dist,
        tsne_perplexity,
    )

all_indices = np.arange(len(X))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=test_size,
    random_state=42,
    stratify=y,
)

X_train = X_pca[train_indices]
X_test = X_pca[test_indices]
X_train_raw = X[train_indices]
X_test_raw = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

base_model = models[selected_model_name]
model = clone(base_model)

training_start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - training_start

prediction_start = time.perf_counter()
y_pred = model.predict(X_test)
prediction_time = time.perf_counter() - prediction_start

accuracy = accuracy_score(y_test, y_pred)

selected_sample = X[selected_index].reshape(1, -1)
selected_sample_pca = pca.transform(selected_sample)
selected_prediction = int(model.predict(selected_sample_pca)[0])
selected_confidence = prediction_confidence(model, selected_sample_pca)

# Modèle dédié aux dessins et images externes.
# Il travaille directement sur les 64 pixels, sans réduction PCA,
# afin de ne pas perdre la forme du chiffre.
external_input_model = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    probability=True,
    random_state=42,
)

external_input_model.fit(
    X_train_raw,
    y_train,
)

external_test_accuracy = accuracy_score(
    y_test,
    external_input_model.predict(X_test_raw),
)


# =========================================================
# OVERVIEW
# =========================================================

st.markdown("### 🔄 Global pipeline")
st.write(
    "Dataset → Normalisation → PCA / UMAP / t-SNE → "
    "Classification → Optimisation → Evaluation → Explainability. "
    "Les dessins externes utilisent un SVM RBF sur les 64 pixels."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", len(X))
m2.metric("Features", X.shape[1])
m3.metric("Classes", n_classes)
m4.metric("Current accuracy", f"{accuracy * 100:.2f}%")

st.info(
    "PCA is used for classification and for transforming new images. "
    "UMAP and t-SNE are used mainly for nonlinear exploratory visualization."
)

st.markdown("---")


# =========================================================
# TABS
# =========================================================

(
    tab_prediction,
    tab_reduction,
    tab_evaluation,
    tab_tuning,
    tab_learning,
    tab_xai,
    tab_drawing,
    tab_anomaly,
    tab_math,
    tab_project,
) = st.tabs(
    [
        "🔍 Prediction",
        "🗺️ PCA / UMAP / t-SNE",
        "📈 Evaluation & ROC",
        "⚙️ Hyperparameter tuning",
        "📚 Learning & overfitting",
        "🧠 Explainable AI",
        "✍️ Draw / Upload",
        "🚨 Anomaly detection",
        "∑ Math",
        "🏆 Project dashboard",
    ]
)


# =========================================================
# TAB: PREDICTION
# =========================================================

with tab_prediction:
    st.subheader("Prediction panel")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            X[selected_index].reshape(8, 8),
            width=230,
            caption="Selected image",
            clamp=True,
        )

    with c2:
        st.metric("True label", int(y[selected_index]))
        st.metric("Predicted label", selected_prediction)
        st.metric(
            "Confidence",
            f"{selected_confidence * 100:.2f}%"
            if selected_confidence is not None
            else "Not available",
        )

    with c3:
        st.metric("Model", selected_model_name)
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("Training time", f"{training_time:.6f} sec")

        if selected_prediction == int(y[selected_index]):
            st.success("✅ Correct prediction")
        else:
            st.error("❌ Incorrect prediction")

    if st.button("🎲 Test a random image", use_container_width=True):
        random_index = int(np.random.randint(0, len(X)))
        random_pca = pca.transform(X[random_index].reshape(1, -1))
        random_prediction = int(model.predict(random_pca)[0])

        r1, r2 = st.columns(2)
        with r1:
            st.image(
                X[random_index].reshape(8, 8),
                width=230,
                caption="Random image",
                clamp=True,
            )
        with r2:
            st.metric("True label", int(y[random_index]))
            st.metric("Predicted label", random_prediction)


# =========================================================
# TAB: REDUCTION
# =========================================================

with tab_reduction:
    st.subheader("Dimensionality reduction comparison")

    method = st.selectbox(
        "Projection method",
        ["PCA", "UMAP", "t-SNE"],
        key="projection_method",
    )

    if method == "PCA":
        embedding = X_pca
        fig = plot_embedding(
            embedding,
            y,
            "PCA projection",
            "Principal component 1",
            "Principal component 2",
            selected_index,
        )
        score = trustworthiness(X, embedding, n_neighbors=5)
        st.info(
            "PCA is a linear method that preserves the directions of maximum variance."
        )

    elif method == "UMAP":
        if X_umap is None:
            embedding = None
            st.error(
                "UMAP is unavailable. Install the package with: pip install umap-learn"
            )
        else:
            embedding = X_umap
            fig = plot_embedding(
                embedding,
                y,
                "UMAP projection",
                "UMAP dimension 1",
                "UMAP dimension 2",
                selected_index,
            )
            score = trustworthiness(X, embedding, n_neighbors=5)
            st.info(
                "UMAP is nonlinear and aims to preserve local neighborhoods "
                "while retaining part of the global structure."
            )

    else:
        embedding = X_tsne
        fig = plot_embedding(
            embedding,
            y,
            "t-SNE projection",
            "t-SNE dimension 1",
            "t-SNE dimension 2",
            selected_index,
        )
        score = trustworthiness(X, embedding, n_neighbors=5)
        st.info(
            "t-SNE is nonlinear and emphasizes local neighborhoods. "
            "Global distances between clusters must be interpreted carefully."
        )

    if embedding is not None:
        st.pyplot(fig, use_container_width=True)
        st.metric("Trustworthiness (local preservation)", f"{score:.4f}")

    st.markdown("### Academic comparison")
    reduction_comparison = pd.DataFrame(
        {
            "Method": ["PCA", "UMAP", "t-SNE"],
            "Type": ["Linear", "Nonlinear", "Nonlinear"],
            "Main objective": [
                "Preserve global variance",
                "Preserve local and part of global structure",
                "Preserve local neighborhoods",
            ],
            "Transform new data": [
                "Yes",
                "Yes, using a fitted UMAP model",
                "Not directly with sklearn TSNE",
            ],
            "Use in this app": [
                "Classification and visualization",
                "Exploratory visualization",
                "Exploratory visualization",
            ],
        }
    )
    st.dataframe(reduction_comparison, use_container_width=True, hide_index=True)


# =========================================================
# TAB: EVALUATION AND ROC
# =========================================================

with tab_evaluation:
    st.subheader("Model evaluation")

    cm = confusion_matrix(y_test, y_pred)

    e1, e2, e3 = st.columns(3)
    e1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    e2.metric("Training time", f"{training_time:.6f} sec")
    e3.metric("Prediction time", f"{prediction_time:.6f} sec")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Confusion matrix")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=[f"True {c}" for c in classes],
                columns=[f"Pred {c}" for c in classes],
            ),
            use_container_width=True,
        )

    with c2:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        image_cm = ax_cm.imshow(cm)
        ax_cm.set_title("Confusion matrix")
        ax_cm.set_xlabel("Predicted class")
        ax_cm.set_ylabel("True class")
        ax_cm.set_xticks(range(len(classes)))
        ax_cm.set_yticks(range(len(classes)))
        ax_cm.set_xticklabels(classes)
        ax_cm.set_yticklabels(classes)

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax_cm.text(col, row, int(cm[row, col]), ha="center", va="center")

        fig_cm.colorbar(image_cm, ax=ax_cm)
        st.pyplot(fig_cm, use_container_width=True)

    st.markdown("### Classification report")
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

    st.markdown("### ROC curve and AUC")

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    if y_score is None:
        st.warning("This model does not provide probability or decision scores.")
    else:
        fig_roc, ax_roc = plt.subplots(figsize=(8, 5))

        if n_classes == 2:
            if np.asarray(y_score).ndim == 2:
                binary_score = np.asarray(y_score)[:, 1]
            else:
                binary_score = np.asarray(y_score)

            fpr, tpr, _ = roc_curve(y_test, binary_score)
            auc_value = auc(fpr, tpr)

            ax_roc.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.legend()
            st.metric("AUC", f"{auc_value:.4f}")

        else:
            y_binary = label_binarize(y_test, classes=classes)
            y_score_array = np.asarray(y_score)

            if y_score_array.ndim == 1:
                st.warning("Multiclass ROC is unavailable for this score format.")
            else:
                auc_values = []
                for class_index, class_value in enumerate(classes):
                    fpr, tpr, _ = roc_curve(
                        y_binary[:, class_index],
                        y_score_array[:, class_index],
                    )
                    class_auc = auc(fpr, tpr)
                    auc_values.append(class_auc)
                    ax_roc.plot(
                        fpr,
                        tpr,
                        label=f"Class {class_value}: AUC={class_auc:.3f}",
                    )

                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.legend(ncol=2, fontsize=8)
                st.metric("Mean AUC", f"{np.mean(auc_values):.4f}")

        ax_roc.set_title("ROC curve")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.grid(alpha=.2)
        st.pyplot(fig_roc, use_container_width=True)


# =========================================================
# TAB: HYPERPARAMETER TUNING
# =========================================================

with tab_tuning:
    st.subheader("Hyperparameter tuning with GridSearchCV")

    st.write(
        "The search compares several configurations using 5-fold cross-validation."
    )

    if st.button("🚀 Run GridSearchCV", use_container_width=True):
        with st.spinner("Searching for the best parameters..."):
            grid = get_grid_and_best_model(
                selected_model_name,
                base_model,
                X_train,
                y_train,
            )

        st.success("Optimization completed")
        st.metric("Best CV accuracy", f"{grid.best_score_ * 100:.2f}%")
        st.write("Best parameters")
        st.json(grid.best_params_)

        tuned_test_accuracy = accuracy_score(
            y_test,
            grid.best_estimator_.predict(X_test),
        )
        st.metric("Tuned test accuracy", f"{tuned_test_accuracy * 100:.2f}%")

        cv_results = pd.DataFrame(grid.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")

        cv_results["mean_test_score"] = cv_results["mean_test_score"].round(4)
        cv_results["std_test_score"] = cv_results["std_test_score"].round(4)

        st.dataframe(
            cv_results.head(15),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# TAB: LEARNING CURVE AND OVERFITTING
# =========================================================

with tab_learning:
    st.subheader("Learning curve")

    train_sizes, train_scores, validation_scores = learning_curve(
        clone(base_model),
        X_pca,
        y,
        cv=5,
        train_sizes=np.linspace(.2, 1.0, 5),
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    fig_learning, ax_learning = plt.subplots(figsize=(8, 5))
    ax_learning.plot(train_sizes, train_mean, marker="o", label="Training score")
    ax_learning.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation score",
    )
    ax_learning.set_title(f"Learning curve - {selected_model_name}")
    ax_learning.set_xlabel("Training samples")
    ax_learning.set_ylabel("Accuracy")
    ax_learning.set_ylim(0, 1.05)
    ax_learning.grid(alpha=.2)
    ax_learning.legend()
    st.pyplot(fig_learning, use_container_width=True)

    st.info(
        "A large persistent gap between training and validation scores "
        "may indicate overfitting."
    )

    st.markdown("### Overfitting demonstration with Decision Tree")

    depths = [1, 2, 3, 4, 5, 8, 12, None]
    overfitting_results = []

    for depth in depths:
        tree = DecisionTreeClassifier(random_state=42, max_depth=depth)
        tree.fit(X_train, y_train)
        overfitting_results.append(
            {
                "max_depth": "None" if depth is None else str(depth),
                "Training accuracy": accuracy_score(
                    y_train,
                    tree.predict(X_train),
                ),
                "Test accuracy": accuracy_score(
                    y_test,
                    tree.predict(X_test),
                ),
            }
        )

    overfitting_df = pd.DataFrame(overfitting_results)

    fig_overfit, ax_overfit = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(overfitting_df))
    ax_overfit.plot(
        x_positions,
        overfitting_df["Training accuracy"],
        marker="o",
        label="Training accuracy",
    )
    ax_overfit.plot(
        x_positions,
        overfitting_df["Test accuracy"],
        marker="o",
        label="Test accuracy",
    )
    ax_overfit.set_xticks(x_positions)
    ax_overfit.set_xticklabels(overfitting_df["max_depth"])
    ax_overfit.set_xlabel("Maximum tree depth")
    ax_overfit.set_ylabel("Accuracy")
    ax_overfit.set_ylim(0, 1.05)
    ax_overfit.grid(alpha=.2)
    ax_overfit.legend()
    st.pyplot(fig_overfit, use_container_width=True)


# =========================================================
# TAB: EXPLAINABLE AI
# =========================================================

with tab_xai:
    st.subheader("Explainable AI")

    st.write(
        "Permutation importance measures how much model performance decreases "
        "when one original pixel is randomly shuffled."
    )

    # The main classifier receives only two PCA components. Therefore, its direct
    # permutation importance contains only two values and cannot be reshaped to 8×8.
    # To obtain a true pixel-level map, we evaluate a complete pipeline:
    # 64 original pixels -> PCA -> selected classifier.
    xai_pipeline = Pipeline(
        steps=[
            ("pca", PCA(n_components=2, random_state=42)),
            ("classifier", clone(base_model)),
        ]
    )

    xai_pipeline.fit(X_train_raw, y_train)

    with st.spinner("Calculating pixel-level permutation importance..."):
        importance_result = permutation_importance(
            xai_pipeline,
            X_test_raw,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
            n_jobs=-1,
        )

    importance_values = importance_result.importances_mean

    if importance_values.size == 64:
        importance_image = importance_values.reshape(8, 8)

        fig_importance, ax_importance = plt.subplots(figsize=(7, 6))
        image_plot = ax_importance.imshow(importance_image)
        ax_importance.set_title(
            f"Pixel permutation importance - {selected_model_name}"
        )
        ax_importance.set_xticks([])
        ax_importance.set_yticks([])
        fig_importance.colorbar(image_plot, ax=ax_importance)

        st.pyplot(fig_importance, use_container_width=True)

        st.info(
            "The strongest zones correspond to original pixels whose random "
            "modification reduces the model's accuracy the most."
        )
    else:
        st.warning(
            f"Pixel map unavailable: expected 64 importance values, "
            f"but received {importance_values.size}."
        )

    st.markdown("### Importance of the two PCA components")

    pca_importance_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1,
    )

    pca_importance_df = pd.DataFrame(
        {
            "Component": ["PC1", "PC2"],
            "Importance": pca_importance_result.importances_mean,
        }
    )

    st.dataframe(
        pca_importance_df.round(6),
        use_container_width=True,
        hide_index=True,
    )

    fig_pca_importance, ax_pca_importance = plt.subplots(figsize=(6, 4))
    ax_pca_importance.bar(
        pca_importance_df["Component"],
        pca_importance_df["Importance"],
    )
    ax_pca_importance.set_title("PCA component importance")
    ax_pca_importance.set_ylabel("Mean decrease in accuracy")
    ax_pca_importance.grid(axis="y", alpha=.2)
    st.pyplot(fig_pca_importance, use_container_width=True)

    if selected_model_name == "Decision Tree":
        st.markdown("### Native Decision Tree pixel importance")

        raw_tree = DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
        )
        raw_tree.fit(X_train_raw, y_train)

        native_importance = raw_tree.feature_importances_

        if native_importance.size == 64:
            native_importance_image = native_importance.reshape(8, 8)

            fig_native, ax_native = plt.subplots(figsize=(7, 6))
            native_plot = ax_native.imshow(native_importance_image)
            ax_native.set_title("Decision Tree native pixel importance")
            ax_native.set_xticks([])
            ax_native.set_yticks([])
            fig_native.colorbar(native_plot, ax=ax_native)

            st.pyplot(fig_native, use_container_width=True)


# =========================================================
# TAB: DRAW OR UPLOAD
# =========================================================

with tab_drawing:
    st.subheader("Dessinez ou téléchargez un chiffre")

    st.info(
        "Le dessin est recadré, centré, converti en image 8 × 8 puis "
        "classé directement à partir de ses 64 pixels par un SVM RBF. "
        "PCA reste réservé à la visualisation."
    )

    st.caption(
        f"Précision du modèle dédié aux images externes sur le jeu de test : "
        f"{external_test_accuracy * 100:.2f}%"
    )

    upload_column, draw_column = st.columns(2)

    with upload_column:
        st.markdown("### Télécharger une image")

        uploaded_file = st.file_uploader(
            "PNG, JPG ou JPEG",
            type=["png", "jpg", "jpeg"],
            key="digit_upload",
        )

        if uploaded_file is not None:
            try:
                uploaded_image = Image.open(uploaded_file)

                processed_image, processed_vector = prepare_image_for_digits(
                    uploaded_image
                )

                uploaded_prediction = int(
                    external_input_model.predict(processed_vector)[0]
                )
                uploaded_confidence = prediction_confidence(
                    external_input_model,
                    processed_vector,
                )

                st.image(
                    uploaded_image,
                    width=230,
                    caption="Image originale",
                )

                st.image(
                    processed_image,
                    width=230,
                    caption="Image traitée en 8 × 8",
                    clamp=True,
                )

                st.metric(
                    "Prédiction",
                    uploaded_prediction,
                )

                if uploaded_confidence is not None:
                    st.metric(
                        "Confiance",
                        f"{uploaded_confidence * 100:.2f}%",
                    )

            except Exception as upload_error:
                st.error(
                    "Impossible de traiter cette image. "
                    f"Détail technique : {upload_error}"
                )

    with draw_column:
        st.markdown("### Dessinez un chiffre")

        if "canvas_version" not in st.session_state:
            st.session_state.canvas_version = 0

        clear_canvas = st.button(
            "🗑️ Effacer le dessin",
            use_container_width=True,
            key="clear_digit_canvas",
        )

        if clear_canvas:
            st.session_state.canvas_version += 1
            st.rerun()

        if not CANVAS_AVAILABLE:
            st.error(
                "La zone de dessin n'a pas pu être chargée."
            )

            st.code(
                "python -m pip uninstall streamlit-drawable-canvas -y\n"
                "python -m pip install streamlit-drawable-canvas-fix",
                language="bash",
            )

            if CANVAS_IMPORT_ERROR:
                st.caption(
                    f"Détail technique : {CANVAS_IMPORT_ERROR}"
                )

        else:
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=320,
                    width=320,
                    drawing_mode="freedraw",
                    update_streamlit=True,
                    display_toolbar=True,
                    key=f"digit_canvas_{st.session_state.canvas_version}",
                )

                if canvas_result.image_data is None:
                    st.warning(
                        "La zone de dessin est chargée, mais aucune donnée "
                        "n'a encore été reçue. Dessinez un chiffre dans le carré noir."
                    )

                else:
                    canvas_array = (
                        canvas_result.image_data[:, :, :3]
                        .astype(np.uint8)
                    )

                    canvas_image = Image.fromarray(
                        canvas_array
                    ).convert("L")

                    canvas_processed, canvas_vector = prepare_image_for_digits(
                        canvas_image
                    )

                    ink_level = float(np.mean(canvas_processed))

                    if ink_level <= 0.005:
                        st.caption(
                            "Dessinez un chiffre blanc dans la zone noire."
                        )

                    else:
                        canvas_prediction = int(
                            external_input_model.predict(canvas_vector)[0]
                        )

                        canvas_confidence = prediction_confidence(
                            external_input_model,
                            canvas_vector,
                        )

                        p1, p2 = st.columns(2)

                        with p1:
                            st.image(
                                canvas_array,
                                width=220,
                                caption="Votre dessin",
                            )

                        with p2:
                            st.image(
                                canvas_processed,
                                width=180,
                                caption="Version 8 × 8",
                                clamp=True,
                            )

                        st.metric(
                            "Prédiction",
                            canvas_prediction,
                        )

                        if canvas_confidence is not None:
                            st.metric(
                                "Confiance",
                                f"{canvas_confidence * 100:.2f}%",
                            )

                        st.success(
                            "Pipeline appliqué : dessin → recadrage → centrage → "
                            "image 8 × 8 → 64 pixels → SVM RBF → prédiction."
                        )

            except Exception as canvas_runtime_error:
                st.error(
                    "La bibliothèque de dessin est installée, mais le composant "
                    "n'a pas pu être affiché correctement."
                )

                st.code(
                    "python -m pip uninstall streamlit-drawable-canvas -y\n"
                    "python -m pip install --upgrade streamlit-drawable-canvas-fix",
                    language="bash",
                )

                st.caption(
                    f"Détail technique : {canvas_runtime_error}"
                )


# =========================================================
# TAB: ANOMALY DETECTION
# =========================================================

with tab_anomaly:
    st.subheader("One-Class SVM anomaly detection")

    normal_digit = st.selectbox(
        "Digit considered normal",
        options=[int(value) for value in classes],
    )

    normal_data = X_pca[y == normal_digit]
    one_class_model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=.05,
    )
    one_class_model.fit(normal_data)

    anomaly_result = int(one_class_model.predict(selected_sample_pca)[0])

    a1, a2 = st.columns(2)
    with a1:
        st.image(
            X[selected_index].reshape(8, 8),
            width=230,
            caption="Selected image",
            clamp=True,
        )
    with a2:
        st.metric("True digit", int(y[selected_index]))
        st.metric("Normal class", normal_digit)

        if anomaly_result == 1:
            st.success("The sample is accepted as normal.")
        else:
            st.error("The sample is detected as an anomaly.")

    all_anomaly_predictions = one_class_model.predict(X_pca)

    fig_anomaly, ax_anomaly = plt.subplots(figsize=(8, 6))
    ax_anomaly.scatter(
        X_pca[all_anomaly_predictions == 1, 0],
        X_pca[all_anomaly_predictions == 1, 1],
        s=20,
        label="Accepted",
    )
    ax_anomaly.scatter(
        X_pca[all_anomaly_predictions == -1, 0],
        X_pca[all_anomaly_predictions == -1, 1],
        s=20,
        label="Anomaly",
    )
    ax_anomaly.scatter(
        selected_sample_pca[:, 0],
        selected_sample_pca[:, 1],
        s=260,
        marker="X",
        label="Selected image",
    )
    ax_anomaly.set_title("One-Class SVM")
    ax_anomaly.set_xlabel("PC1")
    ax_anomaly.set_ylabel("PC2")
    ax_anomaly.grid(alpha=.2)
    ax_anomaly.legend()
    st.pyplot(fig_anomaly, use_container_width=True)


# =========================================================
# TAB: MATHEMATICS
# =========================================================

with tab_math:
    st.subheader("Mathematical foundations")

    st.markdown("### 1. PCA")
    st.latex(r"X_{\mathrm{new}}=XW")
    st.info(
        "L'ACP projette les données sur de nouveaux axes afin de conserver "
        "le maximum de variance tout en réduisant la dimension."
    )

    st.markdown("### 2. UMAP")
    st.latex(
        r"\min_Y \sum_{i\ne j}\left["
        r"v_{ij}\log\left(\frac{v_{ij}}{w_{ij}}\right)"
        r"+(1-v_{ij})\log\left(\frac{1-v_{ij}}{1-w_{ij}}\right)"
        r"\right]"
    )
    st.info(
        "UMAP construit un graphe de voisinage dans l'espace original puis "
        "cherche une projection qui conserve au mieux ces relations."
    )

    st.markdown("### 3. t-SNE")
    st.latex(
        r"C=KL(P\|Q)=\sum_{i\ne j}p_{ij}"
        r"\log\left(\frac{p_{ij}}{q_{ij}}\right)"
    )
    st.info(
        "t-SNE minimise la divergence entre les similarités de l'espace "
        "original et celles de la projection."
    )

    st.markdown("### 4. Logistic Regression")
    st.latex(r"P(y=1\mid x)=\frac{1}{1+e^{-(w^Tx+b)}}")
    st.info(
        "La fonction sigmoïde transforme une combinaison linéaire en une "
        "probabilité comprise entre 0 et 1."
    )

    st.markdown("### 5. LDA")
    st.latex(r"J(w)=\frac{w^TS_Bw}{w^TS_Ww}")
    st.info(
        "LDA maximise la séparation entre les classes et minimise leur "
        "dispersion interne."
    )

    st.markdown("### 6. Bayes")
    st.latex(r"P(C_k\mid x)=\frac{P(x\mid C_k)P(C_k)}{P(x)}")
    st.info(
        "La règle de Bayes estime la probabilité d'appartenance à une classe "
        "après observation des données."
    )

    st.markdown("### 7. QDA")
    st.latex(
        r"\delta_k(x)=-\frac12\ln|\Sigma_k|"
        r"-\frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\ln(\pi_k)"
    )
    st.info(
        "QDA utilise une covariance propre à chaque classe et produit des "
        "frontières quadratiques."
    )

    st.markdown("### 8. Decision Tree")
    st.latex(r"H(S)=-\sum_i p_i\log_2(p_i)")
    st.info(
        "L'entropie mesure l'incertitude utilisée pour sélectionner les "
        "meilleures divisions."
    )

    st.markdown("### 9. KNN")
    st.latex(
        r"d(x_i,x_j)=\sqrt{\sum_{\ell=1}^{p}"
        r"(x_{i\ell}-x_{j\ell})^2}"
    )
    st.info(
        "KNN classe une observation selon les classes de ses voisins les "
        "plus proches."
    )

    st.markdown("### 10. One-Class SVM")
    st.latex(r"f(x)=\operatorname{sign}(w^T\phi(x)-\rho)")
    st.info(
        "One-Class SVM apprend la frontière d'une classe normale et détecte "
        "les observations atypiques."
    )

    st.markdown("### 11. Accuracy")
    st.latex(
        r"\mathrm{Accuracy}="
        r"\frac{\mathrm{Correct\ predictions}}{\mathrm{Total\ predictions}}"
    )
    st.info(
        "L'accuracy représente la proportion totale des prédictions correctes."
    )


# =========================================================
# TAB: PROJECT DASHBOARD
# =========================================================

with tab_project:
    st.subheader("Model ranking dashboard")

    ranking_results = []

    for model_name, candidate in models.items():
        candidate_model = clone(candidate)
        start = time.perf_counter()
        candidate_model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        candidate_predictions = candidate_model.predict(X_test)
        inference_time = time.perf_counter() - start

        ranking_results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, candidate_predictions),
                "Training time": fit_time,
                "Prediction time": inference_time,
            }
        )

    ranking_df = pd.DataFrame(ranking_results).sort_values(
        ["Accuracy", "Prediction time"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))

    st.dataframe(
        ranking_df.round(
            {
                "Accuracy": 4,
                "Training time": 6,
                "Prediction time": 6,
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Training time": st.column_config.NumberColumn(format="%.6f"),
            "Prediction time": st.column_config.NumberColumn(format="%.6f"),
        },
    )

    best_model_name = ranking_df.iloc[0]["Model"]
    st.success(f"🥇 Best current model: {best_model_name}")

    fig_ranking, ax_ranking = plt.subplots(figsize=(8, 5))
    ax_ranking.bar(ranking_df["Model"], ranking_df["Accuracy"])
    ax_ranking.set_title("Accuracy comparison")
    ax_ranking.set_ylabel("Accuracy")
    ax_ranking.set_ylim(0, 1.05)
    ax_ranking.tick_params(axis="x", rotation=25)
    ax_ranking.grid(axis="y", alpha=.2)
    st.pyplot(fig_ranking, use_container_width=True)

    st.markdown("### Complexity analysis")
    st.dataframe(
        model_complexity_table(),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Project positioning")
    st.write(
        "This application is no longer limited to a simple classifier. "
        "It combines exploratory analysis, nonlinear visualization, model "
        "comparison, cross-validation, hyperparameter optimization, ROC/AUC, "
        "learning curves, overfitting analysis, explainability, drawing, "
        "image upload and anomaly detection."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "Developed by AYYOUB REGUIGUE | Master IA | "
    "Python • Scikit-learn • UMAP • Streamlit"
)
