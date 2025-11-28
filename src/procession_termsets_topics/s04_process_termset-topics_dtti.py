#!/usr/bin/env python3
"""
Postprocessing für DTTI-Matrix:
- Top-Texte pro Topic aus dtti_matrix_norm
- Mapping zu Metadaten
- Jahr-Topic-Matrix
- Dokument-Topic-Count-Matrix
- Top-10-Topics (anhand tag_topic_rank.csv)
- Wert-Berechnung basierend auf Top-30 Texten pro Topic (Ranggewichtung)

Beispielaufruf (CLI):

    python src/procession_termsets_topics/s04_process_termset-topics_dtti.py `
        --dtti-matrix-norm output/processed_termset/Termset_Gegenstände_1.2/Termset_Gegenstände_1.2_dtti_matrix_norm.csv `
        --topic-rank-file output/processed_termset/Termset_Gegenstände_1.2/Termset_Gegenstände_1.2_tag_topic_rank.csv `
        --metadata-file data/raw/korpus.csv `
        --meta-sep ";" `
        --output-dir output/processed_termset/Termset_Gegenstände_1.2 `
        --top-n-docs 50 `
        --top-k-topics 10 `
        --max-rank 30

Programmatischer Aufruf (run):

    from pathlib import Path
    from s04_process_termset_topics_dtti import run

    run(
        dtti_matrix_norm=Path("..._dtti_matrix_norm.csv"),
        topic_rank_file=Path("..._tag_topic_rank.csv"),
        metadata_file=Path("korpus.csv"),
        meta_sep=";",
        output_dir=Path("output/processed_termset/Termset_Begriffe_2.3"),
        top_n_docs=50,
        top_k_topics=10,
        max_rank=30,
    )
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def extract_year(text: str) -> int | None:
    """Extrahiert eine Jahreszahl (17xx, 18xx, 19xx) aus einem Text."""
    m = re.search(r"\b(17\d{2}|18\d{2}|19\d{2})\b", str(text))
    return int(m.group()) if m else None


def format_metadata_row(row: pd.Series) -> str:
    """Formatiert eine Metadatenzeile wie 'Autor: Titel. Quelle. Jahr.'."""
    parts: List[str] = []

    if "author_surname" in row and pd.notna(row["author_surname"]):
        parts.append(str(row["author_surname"]).strip() + ":")

    if "title" in row and pd.notna(row["title"]):
        parts.append(str(row["title"]).strip() + ".")

    if "source" in row and pd.notna(row["source"]):
        parts.append(str(row["source"]).strip() + ".")

    year = None
    if "year_first" in row and pd.notna(row["year_first"]):
        year = int(row["year_first"])
    elif "year" in row and pd.notna(row["year"]):
        year = int(row["year"])

    if year is not None:
        parts.append(f"{year}.")

    return " ".join(parts).strip()


def format_metadata(doc_id: str, df_meta: pd.DataFrame) -> str:
    doc_id = str(doc_id)
    if doc_id not in df_meta.index:
        return doc_id  # Fallback: nur ID
    row = df_meta.loc[doc_id]
    return format_metadata_row(row)


def get_root_name(dtti_matrix_norm_file: Path) -> str:
    """
    Leitet einen Root-Namen aus dem dtti_matrix_norm-Dateinamen ab,
    indem typische Suffixe entfernt werden.
    """
    root = dtti_matrix_norm_file.stem
    for suffix in (
        "_dtti_matrix_norm",
        "_dtti_matrix",
        "_matrix_norm",
    ):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root


def get_top_k_topics(
    topic_rank_file: Path,
    top_k: int,
    rank_col: str = "TFIDF-Positions-Rang",
) -> List[str]:
    """Liest tag_topic_rank.csv und liefert die Top-k Topics (als Strings)."""
    df = pd.read_csv(topic_rank_file)
    if "Topic" not in df.columns or rank_col not in df.columns:
        raise ValueError(
            f"{topic_rank_file} muss Spalten 'Topic' und '{rank_col}' enthalten, "
            f"gefunden: {list(df.columns)}"
        )
    df = df.dropna(subset=["Topic", rank_col])
    df["Topic"] = df["Topic"].astype(str)
    df = df.sort_values(rank_col, ascending=True)
    return df["Topic"].head(top_k).tolist()


# ---------------------------------------------------------------------------
# 1) Top-Texte pro Topic aus dtti_matrix_norm
# ---------------------------------------------------------------------------

def compute_topdocs_from_dtti(
    dtti_matrix_norm_file: Path,
    top_n_docs: int,
) -> pd.DataFrame:
    """
    Liest die DTTI-Matrix (normalisiert, Document × Topic) und erzeugt
    eine Ranking-Tabelle:

        - Spalten: Topics
        - Zeilen: Rang 1..top_n_docs
        - Zellen: Dokument-IDs (Index der Originalmatrix)
    """
    df = pd.read_csv(dtti_matrix_norm_file, index_col=0)
    df.index = df.index.astype(str)

    ranking_data: Dict[str, List[str]] = {}
    for topic in df.columns:
        top_docs = df[topic].sort_values(ascending=False).head(top_n_docs)
        ranking_data[topic] = top_docs.index.tolist()

    df_ranking = pd.DataFrame(ranking_data)
    df_ranking.index = [i + 1 for i in df_ranking.index]  # Rang 1..N
    df_ranking.index.name = "Rang"
    return df_ranking


# ---------------------------------------------------------------------------
# 2) Mapping der Topdocs auf Metadaten
# ---------------------------------------------------------------------------

def map_topdocs_to_metadata(
    df_topdocs: pd.DataFrame,
    metadata_file: Path,
    meta_sep: str = ";",
    id_col: str = "_id",
) -> pd.DataFrame:
    """
    Mappt Dokument-IDs in df_topdocs über die Metadaten-Datei auf Strings.
    """
    df_meta = pd.read_csv(metadata_file, sep=meta_sep)
    if id_col not in df_meta.columns:
        raise ValueError(
            f"Metadaten-Datei {metadata_file} enthält keine ID-Spalte '{id_col}'. "
            f"Gefundene Spalten: {list(df_meta.columns)}"
        )

    df_meta[id_col] = df_meta[id_col].astype(str)
    df_meta = df_meta.set_index(id_col)

    df_mapped = df_topdocs.copy()

    for topic in df_mapped.columns:
        df_mapped[topic] = df_mapped[topic].apply(
            lambda doc_id: format_metadata(doc_id, df_meta) if pd.notna(doc_id) else doc_id
        )

    return df_mapped


# ---------------------------------------------------------------------------
# 3) Jahr-Topic-Matrix aus Topdocs-Map
# ---------------------------------------------------------------------------

def build_year_topic_matrix_from_mapped(
    df_topdocs_mapped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt Matrix:
        - Zeilen: Jahre
        - Spalten: Topics
        - Zellen: kommaseparierte Liste von Dokumenten (Strings)
    """
    data: Dict[Tuple[int, str], List[str]] = {}

    for topic in df_topdocs_mapped.columns:
        for doc in df_topdocs_mapped[topic].dropna():
            year = extract_year(doc)
            if year:
                data.setdefault((year, topic), []).append(str(doc))

    if not data:
        return pd.DataFrame()

    years = sorted(set(y for (y, _) in data.keys()))
    topics = df_topdocs_mapped.columns.tolist()

    reshaped = pd.DataFrame(index=years, columns=topics)

    for (year, topic), docs in data.items():
        reshaped.at[year, topic] = ", ".join(docs)

    reshaped = reshaped.fillna("")
    reshaped.index.name = "Jahr"
    return reshaped


# ---------------------------------------------------------------------------
# 4) Dokument-Topic-Count-Matrix (0/1) + Counts per year
# ---------------------------------------------------------------------------

def build_document_topic_count_matrix(
    df_topdocs_mapped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dokument × Topic Binary-Matrix:
        - Dokument als String (Metadaten-Format)
        - Spalten: Topics
        - Werte: 0/1, ob Dokument in Topic-Topliste vorkommt
        - zusätzliche Spalte 'Anzahl Topics'
    """
    doc_topic_map: Dict[str, set] = {}

    for topic in df_topdocs_mapped.columns:
        for doc in df_topdocs_mapped[topic].dropna():
            doc = str(doc).strip()
            if not doc:
                continue
            doc_topic_map.setdefault(doc, set()).add(topic)

    if not doc_topic_map:
        return pd.DataFrame(columns=["Dokument", "Anzahl Topics"])

    topics = df_topdocs_mapped.columns.tolist()
    binary_matrix = pd.DataFrame(0, index=sorted(doc_topic_map.keys()), columns=topics, dtype=int)

    for doc, topics_for_doc in doc_topic_map.items():
        binary_matrix.loc[doc, list(topics_for_doc)] = 1

    binary_matrix.insert(0, "Anzahl Topics", binary_matrix.sum(axis=1))
    binary_matrix.reset_index(inplace=True)
    binary_matrix.rename(columns={"index": "Dokument"}, inplace=True)

    return binary_matrix


def build_topic_counts_per_year(
    df_doc_topic_count: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregiert die Dokument-Topic-Count-Matrix nach Jahr:
        - Jahr × Topic, Werte = Summe der 0/1-Einträge
    """
    df = df_doc_topic_count.copy()

    df["Jahr"] = df["Dokument"].apply(extract_year)
    df = df.dropna(subset=["Jahr"])
    df["Jahr"] = df["Jahr"].astype(int)

    non_topic_cols = ["Dokument", "Jahr", "Anzahl Topics"]
    topic_cols = [c for c in df.columns if c not in non_topic_cols]

    grouped = df.groupby("Jahr")[topic_cols].sum()
    grouped = grouped.sort_index()
    grouped.index.name = "Jahr"
    return grouped


# ---------------------------------------------------------------------------
# 5) Top-10-Topics auswählen + Werte aus Top-30 Rängen berechnen
# ---------------------------------------------------------------------------

def build_top_k_mapped_and_value_tables(
    df_topdocs_mapped: pd.DataFrame,
    topic_rank_file: Path,
    top_k: int,
    max_rank: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    - Selektiert Top-k Topics anhand tag_topic_rank.csv (Spalte 'TFIDF-Positions-Rang').
    - Schneidet df_topdocs_mapped auf diese Topics zu.
    - Erzeugt:
        - df_top10_mapped          (alle Ränge)
        - df_year_document_map     (Rangfolge-Texte pro Jahr; nur erste max_rank Zeilen)
        - df_year_value            (Jahreswerte, aus Rangpositionen)
        - df_value_per_text_topic  (Textwerte + Topic-Spalten)
    """
    # Top-k Topics bestimmen
    top_topics = get_top_k_topics(topic_rank_file, top_k=top_k, rank_col="TFIDF-Positions-Rang")

    # Spalten schneiden (nur vorhandene Topics)
    available_topics = [t for t in top_topics if t in df_topdocs_mapped.columns]
    if not available_topics:
        raise ValueError("Keines der Top-Topics aus tag_topic_rank.csv ist in der DTTI-Topdocs-Tabelle vorhanden.")

    df_top10 = df_topdocs_mapped[available_topics].copy()

    # ---- Jahr-Dokument-Map (Top-k Topics, Rangfolge, Duplikate vermeiden) ----
    # Für die Jahr-Dokument-Map nutzen wir nur die ersten max_rank Zeilen pro Topic.
    df_for_yearmap = df_top10.head(max_rank).copy()
    df_for_yearmap.insert(0, "Rang", range(1, len(df_for_yearmap) + 1))

    # Texte in Rangfolge (Topic-übergreifend) sammeln
    texte_in_rangfolge: List[str] = []
    for _, row in df_for_yearmap.iterrows():
        for topic in available_topics:
            val = row[topic]
            if pd.notna(val) and str(val).strip() != "":
                texte_in_rangfolge.append(str(val))

    from collections import defaultdict
    jahres_map: Dict[int, List[str]] = defaultdict(list)
    gesehen: set[str] = set()

    for text in texte_in_rangfolge:
        if text in gesehen:
            continue
        jahr = extract_year(text)
        if jahr:
            jahres_map[jahr].append(text)
            gesehen.add(text)

    if not jahres_map:
        df_year_doc = pd.DataFrame(columns=["Jahr", "Anzahl"])
    else:
        max_len = max(len(doks) for doks in jahres_map.values())
        cols = ["Anzahl"] + [f"Dokument {i+1}" for i in range(max_len)]
        df_year_doc = pd.DataFrame(index=sorted(jahres_map.keys()), columns=cols)

        for jahr, doks in jahres_map.items():
            df_year_doc.at[jahr, "Anzahl"] = len(doks)
            df_year_doc.loc[jahr, df_year_doc.columns[1 : 1 + len(doks)]] = doks

        df_year_doc = df_year_doc.fillna("")
        df_year_doc.index.name = "Jahr"

    # ---- Wert-Berechnung je Jahr (Value per Year) ----
    # Wir nutzen df_top10.head(max_rank) für die Rang-Werte
    df_top10_for_values = df_top10.head(max_rank).copy()
    n_rows = len(df_top10_for_values)

    # Expliziten Rang hinzufügen (1..n_rows)
    df_top10_for_values.insert(0, "Rang", range(1, n_rows + 1))

    # Wide -> Long: (Rang, Topic, Text)
    long_records = []
    for _, row in df_top10_for_values.iterrows():
        rang = int(row["Rang"])
        for topic in available_topics:
            text = row[topic]
            if pd.isna(text) or str(text).strip() == "":
                continue
            long_records.append({"Rang": rang, "Topic": topic, "Text": str(text)})

    if not long_records:
        df_year_value = pd.DataFrame(columns=["Jahr", "Wert"])
        df_value_per_text_topic = pd.DataFrame(columns=["Text", "SummeWert"])
        return df_top10, df_year_doc, df_year_value, df_value_per_text_topic

    df_rank_long = pd.DataFrame(long_records)

    # Wert = n_rows - (Rang-1)  => Rang 1 -> n_rows, Rang n_rows -> 1
    df_rank_long["Wert"] = n_rows - (df_rank_long["Rang"] - 1)

    # Jahr pro Text bestimmen (über df_year_doc)
    text_year_pairs: List[Tuple[int, str]] = []
    if not df_year_doc.empty:
        text_cols = [c for c in df_year_doc.columns if c.startswith("Dokument ")]
        tmp = df_year_doc.reset_index()
        for _, row in tmp.iterrows():
            jahr_v = row["Jahr"]
            for col in text_cols:
                t = row.get(col)
                if t and isinstance(t, str) and t.strip() != "":
                    text_year_pairs.append((jahr_v, t))

    df_ty = pd.DataFrame(text_year_pairs, columns=["Jahr", "Text"])
    df_ty = df_ty.drop_duplicates()

    # Join: Rangwerte mit Jahr
    df_jtw = df_rank_long.merge(df_ty, on="Text", how="inner")

    # Summe der Werte pro Jahr
    df_year_value = (
        df_jtw.groupby("Jahr")["Wert"]
        .sum()
        .reset_index()
        .sort_values("Jahr")
    )

    # ---- Value per Text + Topic-Matrix ----
    # Summe der Werte je Text
    df_text_sum = (
        df_rank_long.groupby("Text", as_index=False)["Wert"]
        .sum()
        .rename(columns={"Wert": "SummeWert"})
    )

    # Text × Topic Matrix (additive Werte)
    df_text_topic = (
        df_rank_long.groupby(["Text", "Topic"], as_index=False)["Wert"]
        .sum()
        .pivot(index="Text", columns="Topic", values="Wert")
        .fillna(0)
        .astype(int)
    )

    df_value_per_text_topic = (
        df_text_sum.set_index("Text")
        .join(df_text_topic, how="left")
        .fillna(0)
        .reset_index()
    )

    # Spaltenreihenfolge: Text, SummeWert, Topics...
    topic_cols = [c for c in df_value_per_text_topic.columns if c not in ("Text", "SummeWert")]
    df_value_per_text_topic = df_value_per_text_topic[["Text", "SummeWert"] + sorted(topic_cols)]
    df_value_per_text_topic = df_value_per_text_topic.sort_values("SummeWert", ascending=False)

    return df_top10, df_year_doc, df_year_value, df_value_per_text_topic


# ---------------------------------------------------------------------------
# CLI-Argumente
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocessing von DTTI-Matrizen: Top-Texte, Mapping, Jahr-Matrizen und Wert-Berechnung (Top-30)."
    )
    parser.add_argument(
        "--dtti-matrix-norm",
        type=Path,
        required=True,
        help="CSV mit normalisierter DTTI-Matrix (Document × Topic).",
    )
    parser.add_argument(
        "--topic-rank-file",
        type=Path,
        required=True,
        help="tag_topic_rank.csv mit Spalten 'Topic' und 'TFIDF-Positions-Rang'.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        required=True,
        help="Metadaten-Datei (z.B. Korpus.csv).",
    )
    parser.add_argument(
        "--meta-sep",
        type=str,
        default=";",
        help="Trennzeichen der Metadaten-CSV (Default: ';').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Zielordner für die Ausgabedateien.",
    )
    parser.add_argument(
        "--top-n-docs",
        type=int,
        default=50,
        help="Anzahl Top-Texte pro Topic in der DTTI-Rangliste (Default: 50).",
    )
    parser.add_argument(
        "--top-k-topics",
        type=int,
        default=10,
        help="Anzahl Top-Topics für die Wert-Berechnung (Default: 10).",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=30,
        help="Maximale Rangzeile (Top-N pro Topic) für die Wert-Berechnung (Default: 30).",
    )
    return parser


# ---------------------------------------------------------------------------
# Kern-Run-Funktion (für CLI & programmatic run)
# ---------------------------------------------------------------------------

def run(
    dtti_matrix_norm: Path,
    topic_rank_file: Path,
    metadata_file: Path,
    meta_sep: str,
    output_dir: Path,
    top_n_docs: int = 50,
    top_k_topics: int = 10,
    max_rank: int = 30,
) -> None:
    """
    Führt das komplette DTTI-Postprocessing aus.

    Kann sowohl von der CLI (über main) als auch direkt aus Python aufgerufen werden.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    root = get_root_name(dtti_matrix_norm)

    # Dateinamen analog zum Topic-Processing, aber mit '_dtti'
    topdocs_file = output_dir / f"{root}_dtti_topdocs.csv"
    topdocs_mapped_file = output_dir / f"{root}_dtti_topdocs_mapped.csv"
    year_topic_matrix_file = output_dir / f"{root}_dtti_topdocs_year_topic_matrix_map.csv"
    topic_count_matrix_file = output_dir / f"{root}_dtti_topdocs_topic_count_matrix.csv"
    topic_counts_per_year_file = output_dir / f"{root}_dtti_topdocs_topic_counts_per_year.csv"
    top10_mapped_file = output_dir / f"{root}_dtti_topdocs_top10_mapped.csv"
    top10_year_document_map_file = output_dir / f"{root}_dtti_topdocs_top10_year_document_map.csv"
    top10_year_value_file = output_dir / f"{root}_dtti_topdocs_top10_year_value.csv"
    top10_value_per_text_topic_file = output_dir / f"{root}_dtti_topdocs_top10_value_per_text_topic.csv"

    # 1) Topdocs aus dtti_matrix_norm
    df_topdocs = compute_topdocs_from_dtti(
        dtti_matrix_norm_file=dtti_matrix_norm,
        top_n_docs=top_n_docs,
    )
    df_topdocs.to_csv(topdocs_file, index=True, encoding="utf-8")
    print(f"[OK] DTTI-Topdocs geschrieben nach: {topdocs_file}")

    # 2) Mapping mit Metadaten
    df_topdocs_mapped = map_topdocs_to_metadata(
        df_topdocs,
        metadata_file=metadata_file,
        meta_sep=meta_sep,
    )
    df_topdocs_mapped.to_csv(topdocs_mapped_file, index=False, encoding="utf-8")
    print(f"[OK] DTTI-Topdocs (mapped) geschrieben nach: {topdocs_mapped_file}")

    # 3) Jahr-Topic-Matrix aus kompletter Map
    df_year_topic = build_year_topic_matrix_from_mapped(df_topdocs_mapped)
    df_year_topic.to_csv(year_topic_matrix_file, encoding="utf-8")
    print(f"[OK] Jahr-Topic-Matrix (DTTI) geschrieben nach: {year_topic_matrix_file}")

    # 4) Dokument-Topic-Count-Matrix + Counts per year
    df_doc_topic_count = build_document_topic_count_matrix(df_topdocs_mapped)
    df_doc_topic_count.to_csv(topic_count_matrix_file, index=False, encoding="utf-8")
    print(f"[OK] Dokument-Topic-Count-Matrix geschrieben nach: {topic_count_matrix_file}")

    df_topic_counts_per_year = build_topic_counts_per_year(df_doc_topic_count)
    df_topic_counts_per_year.to_csv(topic_counts_per_year_file, encoding="utf-8")
    print(f"[OK] Topic-Counts-pro-Jahr geschrieben nach: {topic_counts_per_year_file}")

    # 5) Top-Topics + Wert-Berechnung mit Top-N Rängen
    df_top10_mapped, df_year_doc_map, df_year_value, df_value_per_text_topic = (
        build_top_k_mapped_and_value_tables(
            df_topdocs_mapped=df_topdocs_mapped,
            topic_rank_file=topic_rank_file,
            top_k=top_k_topics,
            max_rank=max_rank,
        )
    )

    df_top10_mapped.to_csv(top10_mapped_file, index=False, encoding="utf-8")
    print(f"[OK] DTTI-Topdocs (Top-{top_k_topics} Topics, mapped) geschrieben nach: {top10_mapped_file}")

    df_year_doc_map.to_csv(top10_year_document_map_file, encoding="utf-8")
    print(f"[OK] Jahr-Dokument-Map (Top-{top_k_topics} Topics, Top-{max_rank}) nach: {top10_year_document_map_file}")

    df_year_value.to_csv(top10_year_value_file, index=False, encoding="utf-8")
    print(f"[OK] document-topic-distribution_dtti_topdocs_top{top_k_topics}_year_value geschrieben nach: {top10_year_value_file}")

    df_value_per_text_topic.to_csv(top10_value_per_text_topic_file, index=False, encoding="utf-8")
    print(f"[OK] document-topic-distribution_dtti_topdocs_top{top_k_topics}_value_per_text_topic geschrieben nach: {top10_value_per_text_topic_file}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI-Entry-Point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run(
        dtti_matrix_norm=args.dtti_matrix_norm,
        topic_rank_file=args.topic_rank_file,
        metadata_file=args.metadata_file,
        meta_sep=args.meta_sep,
        output_dir=args.output_dir,
        top_n_docs=args.top_n_docs,
        top_k_topics=args.top_k_topics,
        max_rank=args.max_rank,
    )


if __name__ == "__main__":
    main()
