import os
import json
import re
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import cohen_kappa_score


# =========================
# KONFIGURASI FILE
# =========================

CLEAN_CSV = "Kelp4_dataset_2.csv"
ANNOTATION_JSONL = "Kelp4_dataset_anotasi.jsonl"

LABELS = [
    "PRODUCT_POSITIVE", "PRODUCT_NEGATIVE", "PRODUCT_NEUTRAL",
    "PRICE_POSITIVE", "PRICE_NEGATIVE", "PRICE_NEUTRAL",
    "PLACE_POSITIVE", "PLACE_NEGATIVE", "PLACE_NEUTRAL",
    "PROMOTION_POSITIVE", "PROMOTION_NEGATIVE", "PROMOTION_NEUTRAL",
    "OUT_OF_TOPIC"
]

ASPECTS = [
    "PRODUCT",
    "PRICE",
    "PLACE",
    "PROMOTION",
    "OUT_OF_TOPIC"
]


# =========================
# KONFIGURASI STREAMLIT
# =========================

st.set_page_config(
    page_title="Kelp4 Dataset Explorer",
    page_icon="📊",
    layout="wide"
)


# =========================
# FUNGSI BANTUAN UMUM
# =========================

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def get_aspect(label):
    if label == "OUT_OF_TOPIC":
        return "OUT_OF_TOPIC"
    return label.split("_")[0]


def get_sentiment(label):
    if label == "OUT_OF_TOPIC":
        return "OUT_OF_TOPIC"
    return label.split("_")[-1]


def has_aspect(labels, aspect):
    if not isinstance(labels, (list, set, tuple)):
        return False

    if aspect == "OUT_OF_TOPIC":
        return "OUT_OF_TOPIC" in labels

    return any(str(label).startswith(aspect + "_") for label in labels)


def safe_list(value):
    if isinstance(value, list):
        return value
    return []


def safe_kappa(y1, y2):
    """
    Cohen's Kappa tidak bermakna jika semua nilai hanya satu kelas.
    Jika semua nilai sama dan tidak ada variasi, fungsi mengembalikan NaN.
    """
    if len(y1) == 0 or len(y2) == 0:
        return np.nan

    unique_values = set(y1 + y2)

    if len(unique_values) < 2:
        return np.nan

    return cohen_kappa_score(y1, y2)


def interpret_kappa(value):
    if pd.isna(value):
        return "Tidak dapat dihitung"

    if value < 0:
        return "Sangat rendah / tidak ada kesepakatan"
    elif value <= 0.20:
        return "Sangat rendah"
    elif value <= 0.40:
        return "Rendah"
    elif value <= 0.60:
        return "Sedang"
    elif value <= 0.80:
        return "Baik"
    else:
        return "Sangat baik"


def read_csv_auto(path):
    if not os.path.exists(path):
        st.error(f"File {path} tidak ditemukan.")
        st.stop()

    candidates = []

    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            candidates.append(df)
        except Exception:
            pass

    if len(candidates) == 0:
        st.error(f"File {path} gagal dibaca.")
        st.stop()

    df = max(candidates, key=lambda x: len(x.columns))
    return df


def count_table(series, index_name, value_name="jumlah"):
    return series.rename(value_name).rename_axis(index_name).reset_index()


# =========================
# LOAD DATASET
# =========================

@st.cache_data
def load_clean_dataset(path):
    df = read_csv_auto(path)

    if "review_text" not in df.columns:
        if "text" in df.columns:
            df["review_text"] = df["text"]
        else:
            st.error("Kolom review_text atau text tidak ditemukan di dataset CSV.")
            st.stop()

    if "category" not in df.columns:
        df["category"] = "Tidak diketahui"

    if "business_name" not in df.columns:
        df["business_name"] = "Tidak diketahui"

    if "rating" not in df.columns:
        df["rating"] = np.nan

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))
    df["text_key"] = df["review_text"].apply(normalize_text)

    return df


@st.cache_data
def load_annotation_dataset(path):
    if not os.path.exists(path):
        st.error(f"File {path} tidak ditemukan.")
        st.stop()

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    df_raw = pd.DataFrame(data)

    if "text" not in df_raw.columns:
        st.error("Kolom text tidak ditemukan di file anotasi JSONL.")
        st.stop()

    if "accept" not in df_raw.columns:
        df_raw["accept"] = [[] for _ in range(len(df_raw))]

    if "spans" not in df_raw.columns:
        df_raw["spans"] = [[] for _ in range(len(df_raw))]

    df_raw["accept"] = df_raw["accept"].apply(safe_list)
    df_raw["spans"] = df_raw["spans"].apply(safe_list)
    df_raw["text"] = df_raw["text"].fillna("").astype(str)
    df_raw["text_key"] = df_raw["text"].apply(normalize_text)
    df_raw["word_count"] = df_raw["text"].apply(lambda x: len(str(x).split()))

    # Untuk EDA dan visualisasi, ambil satu annotator agar review tidak terhitung ganda.
    df_single = df_raw.copy()

    if "_annotator_id" in df_single.columns and "_input_hash" in df_single.columns:
        main_annotator = df_single["_annotator_id"].value_counts().idxmax()
        df_single = df_single[df_single["_annotator_id"] == main_annotator].copy()
        df_single = df_single.drop_duplicates(subset=["_input_hash"])
    else:
        main_annotator = "Tidak tersedia"

    df_single["labels_str"] = df_single["accept"].apply(
        lambda labels: ", ".join(labels) if labels else "-"
    )

    df_single["aspects"] = df_single["accept"].apply(
        lambda labels: sorted(set(get_aspect(label) for label in labels))
    )

    df_single["sentiments"] = df_single["accept"].apply(
        lambda labels: sorted(set(get_sentiment(label) for label in labels))
    )

    df_single["aspects_str"] = df_single["aspects"].apply(
        lambda x: ", ".join(x) if x else "-"
    )

    df_single["sentiments_str"] = df_single["sentiments"].apply(
        lambda x: ", ".join(x) if x else "-"
    )

    return df_raw, df_single, main_annotator


# =========================
# FUNGSI ENTITAS ABSA / NER
# =========================

def build_entity_dataframe(df_absa):
    entities = []

    for _, row in df_absa.iterrows():
        text = row.get("text", "")
        spans = row.get("spans", [])

        if not isinstance(spans, list):
            continue

        for span in spans:
            start = span.get("start")
            end = span.get("end")
            label = span.get("label")

            if start is not None and end is not None:
                entity_text = text[start:end]

                entities.append({
                    "review_text": text,
                    "entity_text": entity_text,
                    "entity_label": label,
                    "start": start,
                    "end": end
                })

    if len(entities) == 0:
        return pd.DataFrame(
            columns=["review_text", "entity_text", "entity_label", "start", "end"]
        )

    return pd.DataFrame(entities)


def add_entity_string(df_absa):
    entity_strings = []

    for _, row in df_absa.iterrows():
        text = row.get("text", "")
        spans = row.get("spans", [])

        ent_list = []

        if isinstance(spans, list):
            for span in spans:
                start = span.get("start")
                end = span.get("end")
                label = span.get("label")

                if start is not None and end is not None:
                    entity_text = text[start:end]
                    ent_list.append(f"{entity_text} ({label})")

        entity_strings.append("; ".join(ent_list) if ent_list else "-")

    df_absa["entities_str"] = entity_strings
    return df_absa


# =========================
# FUNGSI IRR / COHEN'S KAPPA
# =========================

def build_annotation_records(df):
    records = defaultdict(dict)
    text_lookup = {}

    if "_input_hash" not in df.columns or "_annotator_id" not in df.columns:
        return records, text_lookup

    for _, row in df.iterrows():
        input_hash = row.get("_input_hash")
        annotator = row.get("_annotator_id")
        labels = set(safe_list(row.get("accept", [])))
        text = row.get("text", "")

        if pd.isna(input_hash) or pd.isna(annotator):
            continue

        records[input_hash][annotator] = labels
        text_lookup[input_hash] = text

    return records, text_lookup


def compute_kappa_by_label(df, labels):
    records, _ = build_annotation_records(df)

    if len(records) == 0:
        return pd.DataFrame()

    annotators = df["_annotator_id"].dropna().value_counts().index.tolist()

    if len(annotators) < 2:
        return pd.DataFrame()

    pairs = list(itertools.combinations(annotators, 2))
    results = []

    for ann1, ann2 in pairs:
        paired_items = {
            key: value
            for key, value in records.items()
            if ann1 in value and ann2 in value
        }

        for label in labels:
            y1 = []
            y2 = []

            for item in paired_items.values():
                y1.append(1 if label in item[ann1] else 0)
                y2.append(1 if label in item[ann2] else 0)

            if len(y1) == 0:
                continue

            kappa = safe_kappa(y1, y2)
            agreement = np.mean([a == b for a, b in zip(y1, y2)]) * 100

            results.append({
                "pair": f"{ann1} vs {ann2}",
                "label": label,
                "cohen_kappa": kappa,
                "interpretasi": interpret_kappa(kappa),
                "percent_agreement": agreement,
                "jumlah_data_berpasangan": len(y1),
                "annotator_1_yes": int(sum(y1)),
                "annotator_2_yes": int(sum(y2)),
                "both_yes": int(sum(a == 1 and b == 1 for a, b in zip(y1, y2)))
            })

    pair_df = pd.DataFrame(results)

    if len(pair_df) == 0:
        return pair_df

    # Jika hanya ada 2 annotator, hasilnya sama.
    # Jika lebih dari 2, rata-rata Kappa per label dihitung dari semua pasangan annotator.
    label_df = (
        pair_df
        .groupby("label", as_index=False)
        .agg({
            "cohen_kappa": "mean",
            "percent_agreement": "mean",
            "jumlah_data_berpasangan": "max",
            "annotator_1_yes": "sum",
            "annotator_2_yes": "sum",
            "both_yes": "sum"
        })
    )

    label_df["interpretasi"] = label_df["cohen_kappa"].apply(interpret_kappa)
    label_df = label_df.sort_values("cohen_kappa", ascending=False, na_position="last")

    return label_df


def compute_kappa_by_aspect(df, aspects):
    records, _ = build_annotation_records(df)

    if len(records) == 0:
        return pd.DataFrame()

    annotators = df["_annotator_id"].dropna().value_counts().index.tolist()

    if len(annotators) < 2:
        return pd.DataFrame()

    pairs = list(itertools.combinations(annotators, 2))
    results = []

    for ann1, ann2 in pairs:
        paired_items = {
            key: value
            for key, value in records.items()
            if ann1 in value and ann2 in value
        }

        for aspect in aspects:
            y1 = []
            y2 = []

            for item in paired_items.values():
                y1.append(1 if has_aspect(item[ann1], aspect) else 0)
                y2.append(1 if has_aspect(item[ann2], aspect) else 0)

            if len(y1) == 0:
                continue

            kappa = safe_kappa(y1, y2)
            agreement = np.mean([a == b for a, b in zip(y1, y2)]) * 100

            results.append({
                "pair": f"{ann1} vs {ann2}",
                "aspek": aspect,
                "cohen_kappa": kappa,
                "interpretasi": interpret_kappa(kappa),
                "percent_agreement": agreement,
                "jumlah_data_berpasangan": len(y1),
                "annotator_1_yes": int(sum(y1)),
                "annotator_2_yes": int(sum(y2)),
                "both_yes": int(sum(a == 1 and b == 1 for a, b in zip(y1, y2)))
            })

    pair_df = pd.DataFrame(results)

    if len(pair_df) == 0:
        return pair_df

    aspect_df = (
        pair_df
        .groupby("aspek", as_index=False)
        .agg({
            "cohen_kappa": "mean",
            "percent_agreement": "mean",
            "jumlah_data_berpasangan": "max",
            "annotator_1_yes": "sum",
            "annotator_2_yes": "sum",
            "both_yes": "sum"
        })
    )

    aspect_df["interpretasi"] = aspect_df["cohen_kappa"].apply(interpret_kappa)
    aspect_df = aspect_df.sort_values("cohen_kappa", ascending=False, na_position="last")

    return aspect_df


def get_paired_review_count(df):
    if "_input_hash" not in df.columns or "_annotator_id" not in df.columns:
        return 0

    counts = df.groupby("_input_hash")["_annotator_id"].nunique()
    return int((counts >= 2).sum())


def get_disagreement_examples(df, max_examples=50):
    records, text_lookup = build_annotation_records(df)

    if len(records) == 0:
        return pd.DataFrame()

    annotators = df["_annotator_id"].dropna().value_counts().index.tolist()

    if len(annotators) < 2:
        return pd.DataFrame()

    ann1, ann2 = annotators[0], annotators[1]

    examples = []

    for input_hash, item in records.items():
        if ann1 not in item or ann2 not in item:
            continue

        labels_1 = item[ann1]
        labels_2 = item[ann2]

        if labels_1 != labels_2:
            examples.append({
                "review_text": text_lookup.get(input_hash, ""),
                f"label_{ann1}": ", ".join(sorted(labels_1)) if labels_1 else "-",
                f"label_{ann2}": ", ".join(sorted(labels_2)) if labels_2 else "-",
                "label_sama": ", ".join(sorted(labels_1.intersection(labels_2))) if labels_1.intersection(labels_2) else "-",
                "hanya_annotator_1": ", ".join(sorted(labels_1 - labels_2)) if labels_1 - labels_2 else "-",
                "hanya_annotator_2": ", ".join(sorted(labels_2 - labels_1)) if labels_2 - labels_1 else "-"
            })

        if len(examples) >= max_examples:
            break

    return pd.DataFrame(examples)


# =========================
# FUNGSI FILTER
# =========================

def filter_by_label_aspect_sentiment(df, selected_aspect, selected_sentiment, selected_label):
    filtered = df.copy()

    def match_filter(labels):
        if not isinstance(labels, list):
            return False

        if selected_label != "Semua":
            return selected_label in labels

        if selected_aspect == "Semua" and selected_sentiment == "Semua":
            return True

        for label in labels:
            aspect = get_aspect(label)
            sentiment = get_sentiment(label)

            aspect_ok = selected_aspect == "Semua" or aspect == selected_aspect
            sentiment_ok = selected_sentiment == "Semua" or sentiment == selected_sentiment

            if aspect_ok and sentiment_ok:
                return True

        return False

    filtered = filtered[filtered["accept"].apply(match_filter)]
    return filtered


# =========================
# FUNGSI PLOT
# =========================

def plot_bar(series, title, xlabel, ylabel):
    if len(series) == 0:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    series.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)


def plot_hist(data, title, xlabel, ylabel):
    if len(data) == 0:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)


def plot_kappa_bar(df, label_col, value_col, title):
    if len(df) == 0:
        st.warning("Tidak ada data Kappa untuk divisualisasikan.")
        return

    plot_data = df.dropna(subset=[value_col]).set_index(label_col)[value_col]

    if len(plot_data) == 0:
        st.warning("Nilai Kappa kosong atau tidak dapat dihitung.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Cohen's Kappa")
    ax.set_ylabel(label_col)
    plt.tight_layout()
    st.pyplot(fig)


# =========================
# LOAD DATA
# =========================

df_clean = load_clean_dataset(CLEAN_CSV)
df_annot_raw, df_absa, main_annotator = load_annotation_dataset(ANNOTATION_JSONL)
df_absa = add_entity_string(df_absa)
entity_df = build_entity_dataframe(df_absa)

# Gabungkan metadata CSV ke data anotasi berdasarkan teks review
meta_cols = ["text_key", "category", "business_name", "rating"]
df_meta = df_clean[meta_cols].drop_duplicates(subset=["text_key"])

df_absa_full = df_absa.merge(df_meta, on="text_key", how="left")

df_label_meta = df_absa[["text_key", "accept", "labels_str", "entities_str"]].drop_duplicates(subset=["text_key"])
df_browse = df_clean.merge(df_label_meta, on="text_key", how="left")

df_browse["labels_str"] = df_browse["labels_str"].fillna("-")
df_browse["entities_str"] = df_browse["entities_str"].fillna("-")
df_browse["accept"] = df_browse["accept"].apply(safe_list)


# =========================
# HITUNG IRR / KAPPA
# =========================

kappa_label_df = compute_kappa_by_label(df_annot_raw, LABELS)
kappa_aspect_df = compute_kappa_by_aspect(df_annot_raw, ASPECTS)
disagreement_df = get_disagreement_examples(df_annot_raw, max_examples=100)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Kelp4 Dataset Explorer")

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview",
        "Browse Reviews",
        "Panel ABSA",
        "Statistics",
        "IRR Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Umum")

category_options = ["Semua"] + sorted(df_clean["category"].dropna().astype(str).unique().tolist())
selected_category = st.sidebar.selectbox("Filter kategori usaha", category_options)

aspect_options = ["Semua", "PRODUCT", "PRICE", "PLACE", "PROMOTION", "OUT_OF_TOPIC"]
selected_aspect = st.sidebar.selectbox("Filter aspek", aspect_options)

sentiment_options = ["Semua", "POSITIVE", "NEGATIVE", "NEUTRAL", "OUT_OF_TOPIC"]
selected_sentiment = st.sidebar.selectbox("Filter sentimen", sentiment_options)

label_options = ["Semua"] + LABELS
selected_label = st.sidebar.selectbox("Filter label aspek-sentimen", label_options)

search_query = st.sidebar.text_input("Cari kata dalam review")


# =========================
# TERAPKAN FILTER
# =========================

filtered_browse = df_browse.copy()

if selected_category != "Semua":
    filtered_browse = filtered_browse[filtered_browse["category"].astype(str) == selected_category]

if search_query.strip() != "":
    q = search_query.lower().strip()
    filtered_browse = filtered_browse[
        filtered_browse["review_text"].astype(str).str.lower().str.contains(q, na=False)
    ]

filtered_browse = filter_by_label_aspect_sentiment(
    filtered_browse,
    selected_aspect,
    selected_sentiment,
    selected_label
)

filtered_absa = df_absa_full.copy()

if selected_category != "Semua" and "category" in filtered_absa.columns:
    filtered_absa = filtered_absa[filtered_absa["category"].astype(str) == selected_category]

if search_query.strip() != "":
    q = search_query.lower().strip()
    filtered_absa = filtered_absa[
        filtered_absa["text"].astype(str).str.lower().str.contains(q, na=False)
    ]

filtered_absa = filter_by_label_aspect_sentiment(
    filtered_absa,
    selected_aspect,
    selected_sentiment,
    selected_label
)


# =========================
# HALAMAN OVERVIEW
# =========================

if page == "Overview":
    st.title("📊 Kelp4 Dataset Explorer")
    st.write("Aplikasi ini digunakan untuk mengeksplorasi dataset review Google Places dan hasil anotasi ABSA.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Jumlah Review Bersih", len(df_clean))

    with col2:
        st.metric("Jumlah Review Anotasi", len(df_absa))

    with col3:
        st.metric("Jumlah Entitas ABSA", len(entity_df))

    with col4:
        total_labels = df_absa["accept"].explode().nunique()
        st.metric("Jumlah Label Unik", total_labels)

    st.markdown("### Informasi Dataset")
    st.write(f"File dataset bersih: `{CLEAN_CSV}`")
    st.write(f"File dataset anotasi: `{ANNOTATION_JSONL}`")
    st.write(f"Annotator yang digunakan untuk visualisasi EDA: `{main_annotator}`")

    st.markdown("### Distribusi Kategori Usaha")
    category_counts = df_clean["category"].value_counts()
    st.dataframe(count_table(category_counts, "category"), use_container_width=True)
    plot_bar(category_counts, "Distribusi Kategori Usaha", "Jumlah Review", "Kategori")

    if "rating" in df_clean.columns:
        st.markdown("### Distribusi Rating")
        rating_counts = df_clean["rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)

    st.markdown("### Contoh Data Review")
    preview_cols = ["category", "business_name", "review_text", "rating"]
    preview_cols = [col for col in preview_cols if col in df_clean.columns]
    st.dataframe(df_clean[preview_cols].head(20), use_container_width=True)


# =========================
# HALAMAN BROWSE REVIEWS
# =========================

elif page == "Browse Reviews":
    st.title("🔎 Browse Reviews")
    st.write("Halaman ini menampilkan daftar review dengan informasi kategori, nama usaha, teks review, rating, label ABSA, dan entitas.")

    st.info(f"Jumlah data setelah filter: {len(filtered_browse)}")

    browse_cols = [
        "category",
        "business_name",
        "review_text",
        "rating",
        "labels_str",
        "entities_str",
        "word_count"
    ]

    existing_cols = [col for col in browse_cols if col in filtered_browse.columns]

    st.dataframe(
        filtered_browse[existing_cols],
        use_container_width=True,
        height=600
    )


# =========================
# HALAMAN PANEL ABSA
# =========================

elif page == "Panel ABSA":
    st.title("🏷️ Panel ABSA")
    st.write("Halaman ini menampilkan hasil anotasi ABSA dalam bentuk label multi-label dan entitas NER.")

    st.info(f"Jumlah data anotasi setelah filter: {len(filtered_absa)}")

    st.markdown("### Dataset Tabular Label ABSA")

    absa_cols = [
        "category",
        "business_name",
        "rating",
        "text",
        "labels_str",
        "aspects_str",
        "sentiments_str",
        "entities_str"
    ]

    existing_absa_cols = [col for col in absa_cols if col in filtered_absa.columns]

    st.dataframe(
        filtered_absa[existing_absa_cols],
        use_container_width=True,
        height=500
    )

    st.markdown("### Dataset Tabular Entitas NER")

    filtered_text_keys = set(filtered_absa["text_key"].tolist())

    entity_df_display = entity_df.copy()
    entity_df_display["text_key"] = entity_df_display["review_text"].apply(normalize_text)
    entity_df_display = entity_df_display[entity_df_display["text_key"].isin(filtered_text_keys)]

    if len(entity_df_display) > 0:
        st.dataframe(
            entity_df_display[["entity_text", "entity_label", "start", "end", "review_text"]],
            use_container_width=True,
            height=500
        )
    else:
        st.warning("Tidak ada entitas NER yang sesuai dengan filter.")


# =========================
# HALAMAN STATISTICS
# =========================

elif page == "Statistics":
    st.title("📈 Statistics")
    st.write("Halaman ini menampilkan ringkasan statistik dataset, distribusi panjang review, distribusi label, distribusi aspek, dan korelasi antar label.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Jumlah Review", len(filtered_browse))

    with col2:
        avg_words = round(filtered_browse["word_count"].mean(), 2) if len(filtered_browse) > 0 else 0
        st.metric("Rata-rata Kata", avg_words)

    with col3:
        median_words = round(filtered_browse["word_count"].median(), 2) if len(filtered_browse) > 0 else 0
        st.metric("Median Kata", median_words)

    with col4:
        max_words = int(filtered_browse["word_count"].max()) if len(filtered_browse) > 0 else 0
        st.metric("Maksimum Kata", max_words)

    st.markdown("### Statistik Panjang Review")
    st.dataframe(
        filtered_browse["word_count"]
        .describe()
        .rename("nilai")
        .rename_axis("statistik")
        .reset_index(),
        use_container_width=True
    )

    st.markdown("### Distribusi Panjang Review")
    plot_hist(
        filtered_browse["word_count"],
        "Distribusi Panjang Review Berdasarkan Jumlah Kata",
        "Jumlah Kata",
        "Jumlah Review"
    )

    st.markdown("### Distribusi Label ABSA")
    label_counts = filtered_absa["accept"].explode().value_counts()

    if len(label_counts) > 0:
        st.dataframe(count_table(label_counts, "label"), use_container_width=True)
        plot_bar(label_counts, "Distribusi Label ABSA", "Jumlah Review", "Label")
    else:
        st.warning("Tidak ada label yang sesuai dengan filter.")

    st.markdown("### Distribusi Data per Aspek")

    aspect_list = []

    for labels in filtered_absa["accept"]:
        if isinstance(labels, list):
            for label in labels:
                aspect_list.append(get_aspect(label))

    aspect_counts = pd.Series(aspect_list).value_counts()

    if len(aspect_counts) > 0:
        st.dataframe(count_table(aspect_counts, "aspek"), use_container_width=True)
        plot_bar(aspect_counts, "Distribusi Data per Aspek", "Jumlah", "Aspek")
    else:
        st.warning("Tidak ada aspek yang sesuai dengan filter.")

    st.markdown("### Distribusi Entitas ABSA")

    filtered_text_keys = set(filtered_absa["text_key"].tolist())

    entity_filtered = entity_df.copy()
    entity_filtered["text_key"] = entity_filtered["review_text"].apply(normalize_text)
    entity_filtered = entity_filtered[entity_filtered["text_key"].isin(filtered_text_keys)]

    if len(entity_filtered) > 0:
        entity_counts = entity_filtered["entity_label"].value_counts()
        st.dataframe(count_table(entity_counts, "entity_label"), use_container_width=True)
        plot_bar(entity_counts, "Distribusi Entitas ABSA", "Jumlah Entitas", "Label Entitas")
    else:
        st.warning("Tidak ada entitas yang sesuai dengan filter.")

    st.markdown("### Matriks Korelasi Antar Label")

    if len(filtered_absa) > 0:
        label_binary = pd.DataFrame(0, index=filtered_absa.index, columns=LABELS)

        for idx, labels in filtered_absa["accept"].items():
            if isinstance(labels, list):
                for label in labels:
                    if label in label_binary.columns:
                        label_binary.loc[idx, label] = 1

        label_binary = label_binary.loc[:, label_binary.sum(axis=0) > 0]

        if label_binary.shape[1] >= 2:
            corr_matrix = label_binary.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr_matrix)
            fig.colorbar(im, ax=ax)

            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))

            ax.set_xticklabels(corr_matrix.columns, rotation=90)
            ax.set_yticklabels(corr_matrix.columns)

            ax.set_title("Matriks Korelasi Antar Label ABSA")

            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(corr_matrix, use_container_width=True)
        else:
            st.warning("Label yang tersedia setelah filter kurang dari dua, sehingga korelasi tidak dapat dihitung.")
    else:
        st.warning("Tidak ada data anotasi yang sesuai dengan filter.")


# =========================
# HALAMAN IRR REPORT
# =========================

elif page == "IRR Report":
    st.title("🤝 Inter-Annotator Agreement / IRR Report")
    st.write(
        "Halaman ini menampilkan hasil perhitungan Inter-Annotator Agreement menggunakan Cohen's Kappa "
        "berdasarkan file hasil anotasi."
    )

    st.markdown("### Ringkasan Data IRR")

    total_annotations = len(df_annot_raw)
    unique_reviews = df_annot_raw["_input_hash"].nunique() if "_input_hash" in df_annot_raw.columns else len(df_annot_raw)
    paired_reviews = get_paired_review_count(df_annot_raw)

    annotator_count = (
        df_annot_raw["_annotator_id"].nunique()
        if "_annotator_id" in df_annot_raw.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Baris Anotasi", total_annotations)

    with col2:
        st.metric("Review Unik", unique_reviews)

    with col3:
        st.metric("Jumlah Annotator", annotator_count)

    with col4:
        st.metric("Review Berpasangan", paired_reviews)

    if "_annotator_id" in df_annot_raw.columns:
        st.markdown("### Jumlah Data per Annotator")
        annotator_counts = df_annot_raw["_annotator_id"].value_counts()
        st.dataframe(count_table(annotator_counts, "annotator"), use_container_width=True)

    st.markdown("---")

    st.markdown("## Nilai Cohen's Kappa per Aspek")

    if len(kappa_aspect_df) > 0:
        display_aspect_df = kappa_aspect_df.copy()
        display_aspect_df["cohen_kappa"] = display_aspect_df["cohen_kappa"].round(3)
        display_aspect_df["percent_agreement"] = display_aspect_df["percent_agreement"].round(2)

        st.dataframe(display_aspect_df, use_container_width=True)
        plot_kappa_bar(kappa_aspect_df, "aspek", "cohen_kappa", "Cohen's Kappa per Aspek")
    else:
        st.warning("Nilai Kappa per aspek tidak dapat dihitung.")

    st.markdown("## Nilai Cohen's Kappa per Label")

    if len(kappa_label_df) > 0:
        display_label_df = kappa_label_df.copy()
        display_label_df["cohen_kappa"] = display_label_df["cohen_kappa"].round(3)
        display_label_df["percent_agreement"] = display_label_df["percent_agreement"].round(2)

        st.dataframe(display_label_df, use_container_width=True)
        plot_kappa_bar(kappa_label_df, "label", "cohen_kappa", "Cohen's Kappa per Label")
    else:
        st.warning("Nilai Kappa per label tidak dapat dihitung.")

    st.markdown("---")

    st.markdown("## Interpretasi Hasil IRR")

    if len(kappa_aspect_df) > 0:
        best_aspect = kappa_aspect_df.dropna(subset=["cohen_kappa"]).sort_values("cohen_kappa", ascending=False).head(1)
        worst_aspect = kappa_aspect_df.dropna(subset=["cohen_kappa"]).sort_values("cohen_kappa", ascending=True).head(1)

        if len(best_aspect) > 0:
            best_name = best_aspect.iloc[0]["aspek"]
            best_value = best_aspect.iloc[0]["cohen_kappa"]
            best_interp = best_aspect.iloc[0]["interpretasi"]

            st.success(
                f"Aspek dengan kesepakatan tertinggi adalah **{best_name}** "
                f"dengan nilai Cohen's Kappa **{best_value:.3f}**, "
                f"yang termasuk kategori **{best_interp}**."
            )

        if len(worst_aspect) > 0:
            worst_name = worst_aspect.iloc[0]["aspek"]
            worst_value = worst_aspect.iloc[0]["cohen_kappa"]
            worst_interp = worst_aspect.iloc[0]["interpretasi"]

            st.warning(
                f"Aspek dengan kesepakatan terendah adalah **{worst_name}** "
                f"dengan nilai Cohen's Kappa **{worst_value:.3f}**, "
                f"yang termasuk kategori **{worst_interp}**."
            )

    st.markdown(
        """
        Secara umum, nilai Cohen's Kappa digunakan untuk melihat tingkat kesepakatan antar annotator
        setelah memperhitungkan kemungkinan kesepakatan yang terjadi secara kebetulan.

        Interpretasi yang digunakan:
        - Kappa < 0.00: sangat rendah atau tidak ada kesepakatan
        - 0.00 sampai 0.20: sangat rendah
        - 0.21 sampai 0.40: rendah
        - 0.41 sampai 0.60: sedang
        - 0.61 sampai 0.80: baik
        - 0.81 sampai 1.00: sangat baik
        """
    )

    st.markdown("---")

    st.markdown("## Analisis Penyebab Ketidaksepakatan")

    st.markdown(
        """
        Ketidaksepakatan antar annotator dapat terjadi karena beberapa alasan berikut:

        1. **Batas antar aspek belum selalu jelas.**  
           Beberapa review dapat membahas lebih dari satu aspek secara bersamaan. Misalnya kata seperti
           "murah", "worth it", atau "terjangkau" dapat dianggap sebagai aspek `PRICE`,
           tetapi pada konteks tertentu juga dapat dikaitkan dengan `PROMOTION`.

        2. **Aspek Promotion cenderung ambigu.**  
           Tidak semua review menyebutkan promosi secara eksplisit seperti diskon, voucher, atau promo.
           Akibatnya, annotator dapat berbeda dalam menentukan apakah sebuah review termasuk aspek promosi.

        3. **Label OUT_OF_TOPIC dapat menimbulkan perbedaan interpretasi.**  
           Review yang membahas pelayanan, keramahan pegawai, pembayaran, atau pengalaman umum pelanggan
           tidak selalu masuk secara jelas ke aspek Product, Price, Place, atau Promotion.
           Hal ini dapat menyebabkan satu annotator memilih `OUT_OF_TOPIC`, sedangkan annotator lain memilih aspek lain.

        4. **Perbedaan pemilihan multi-label.**  
           Karena satu review dapat memiliki lebih dari satu label, annotator bisa saja sepakat pada satu label,
           tetapi berbeda pada label tambahan. Misalnya keduanya memilih `PRODUCT_POSITIVE`,
           namun hanya satu annotator yang menambahkan `PLACE_POSITIVE`.
        """
    )

    st.markdown("### Contoh Review yang Tidak Disepakati Annotator")

    if len(disagreement_df) > 0:
        st.dataframe(disagreement_df, use_container_width=True, height=500)
    else:
        st.success("Tidak ditemukan contoh disagreement, atau data annotator berpasangan tidak tersedia.")