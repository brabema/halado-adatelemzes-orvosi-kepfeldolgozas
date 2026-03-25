import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_FINDINGS = [
    "Aortic enlargement",
    "Cardiomegaly",
    "Pleural thickening",
    "Pulmonary fibrosis",
    "Lung Opacity"
]

def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    for finding in TARGET_FINDINGS:
        df[finding] = (df['class_name'] == finding).astype(int)

    df['Other_finding'] = (
        (df['class_name'] != 'No finding') &
        (~df['class_name'].isin(TARGET_FINDINGS))
    ).astype(int)

    grouped = df.groupby('image_id').agg({
        **{f: 'max' for f in TARGET_FINDINGS},
        'Other_finding': 'max'
    }).reset_index()

    grouped['target_sum'] = grouped[TARGET_FINDINGS].sum(axis=1)

    valid_mask = ~(
        (grouped['target_sum'] == 0) &
        (grouped['Other_finding'] == 1)
    )

    clean_df = grouped[valid_mask].copy()
    clean_df = clean_df.drop(columns=['Other_finding', 'target_sum'])

    return clean_df


def split_data(df, test_size=0.15, valid_size=0.15, random_state=42):
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    val_ratio = valid_size / (1.0 - test_size)

    train_df, valid_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=random_state
    )

    return train_df, valid_df, test_df