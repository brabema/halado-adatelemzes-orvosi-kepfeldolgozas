import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import prepare_dataframe, split_data, TARGET_FINDINGS
from augmentation import get_train_transforms, get_val_transforms
from dataset import read_image

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datas")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "jpg_1024")

st.set_page_config(page_title="VinBigData Pipeline", layout="wide")
st.title("VinBigData adatelőkészítés dashboard")

# --- 1. Adatbetöltés ---
st.header("1. Adatbetöltés")

if not os.path.exists(CSV_PATH):
    st.error(f"train.csv nem található: {CSV_PATH}")
    st.stop()

raw_df = pd.read_csv(CSV_PATH)
total_images = raw_df['image_id'].nunique()
total_rows = len(raw_df)
st.write(f"**{total_rows}** annotációs sor, **{total_images}** egyedi kép")
st.write("Finding eloszlás:")
st.bar_chart(raw_df['class_name'].value_counts())

# --- 2. Szűrés ---
st.header("2. Szűrés")

clean_df = prepare_dataframe(CSV_PATH)
removed = total_images - len(clean_df)
st.write(f"Eredeti: **{total_images}** kép → Szűrés után: **{len(clean_df)}** kép (törölve: **{removed}**)")

col1, col2 = st.columns(2)
with col1:
    st.write("Target finding eloszlás (tisztított):")
    finding_counts = clean_df[TARGET_FINDINGS].sum().sort_values(ascending=False)
    st.bar_chart(finding_counts)
with col2:
    st.write("No finding arány:")
    no_finding = (clean_df[TARGET_FINDINGS].sum(axis=1) == 0).sum()
    has_finding = len(clean_df) - no_finding
    st.bar_chart(pd.Series({"No finding": no_finding, "Has finding": has_finding}))

# --- 3. Split ---
st.header("3. Split")

train_df, valid_df, test_df = split_data(clean_df)
split_sizes = pd.DataFrame({
    "Halmaz": ["Train", "Validation", "Test"],
    "Képek": [len(train_df), len(valid_df), len(test_df)],
    "Arány": [f"{len(train_df)/len(clean_df)*100:.1f}%",
              f"{len(valid_df)/len(clean_df)*100:.1f}%",
              f"{len(test_df)/len(clean_df)*100:.1f}%"]
})
st.table(split_sizes)

st.write("Finding arányok halmazonként:")
split_stats = []
for name, sdf in [("Train", train_df), ("Validation", valid_df), ("Test", test_df)]:
    for f in TARGET_FINDINGS:
        rate = sdf[f].mean() * 100
        split_stats.append({"Halmaz": name, "Finding": f, "Arány (%)": round(rate, 1)})
split_stats_df = pd.DataFrame(split_stats)
st.dataframe(split_stats_df.pivot(index="Finding", columns="Halmaz", values="Arány (%)"))

# --- 4. Augmentáció ---
st.header("4. Augmentáció")

has_images = os.path.isdir(IMAGE_DIR) and len(os.listdir(IMAGE_DIR)) > 0
if not has_images:
    st.warning(f"Képek nem találhatók: {IMAGE_DIR}")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    use_flip = st.checkbox("Horizontal Flip", value=True)
with col2:
    use_rotate = st.checkbox("Rotation (±10°)", value=True)
with col3:
    use_brightness = st.checkbox("Brightness/Contrast", value=True)

train_transforms = get_train_transforms(shift=True, scale=True, rotate=True, brightness=True)

sample_idx = st.slider("Kép index (train halmazból)", 0, len(train_df) - 1, 0)
image_id = train_df.iloc[sample_idx]['image_id']
img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

if os.path.exists(img_path):
    original = read_image(img_path, target_size=(512, 512))
    label = train_df.iloc[sample_idx][TARGET_FINDINGS].tolist()
    active_findings = [f for f, v in zip(TARGET_FINDINGS, label) if v == 1]
    if not active_findings:
        active_findings = ["No finding"]
    st.write(f"**{image_id}** — {', '.join(active_findings)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Eredeti")
        st.image(original, channels="RGB", use_container_width=True)
    with col2:
        st.write("Augmentált (1)")
        aug1 = train_transforms(image=original)['image']
        aug1_np = aug1.permute(1, 2, 0).numpy()
        aug1_np = ((aug1_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        st.image(aug1_np, channels="RGB", use_container_width=True)
    with col3:
        st.write("Augmentált (2)")
        aug2 = train_transforms(image=original)['image']
        aug2_np = aug2.permute(1, 2, 0).numpy()
        aug2_np = ((aug2_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        st.image(aug2_np, channels="RGB", use_container_width=True)

    if st.button("Új augmentáció"):
        st.rerun()
else:
    st.error(f"Kép nem található: {img_path}")
