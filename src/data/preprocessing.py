
from pathlib import Path
import pandas as pd
import logging
from src.data.ingestion import load_data

logger = logging.getLogger(__name__)


def clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:

    logger.info("Starting data preprocessing")

    df = load_data(input_path)
    df = df.copy()

    df = df.dropna(subset=["clean_comment", "category"])
    df = df.drop_duplicates()
    df = df[df["clean_comment"].str.strip() != ""]
    df = df[df["category"].isin([-1, 0, 1])]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info("Preprocessing completed")
    logger.info("Final dataset shape: %s", df.shape)

    return df
if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "raw" / "reddit_preprocessing.csv"
    output_path = project_root / "data" / "processed" / "reddit_cleaned.csv"

    clean_data(
        input_path=input_path,
        output_path=output_path
    )