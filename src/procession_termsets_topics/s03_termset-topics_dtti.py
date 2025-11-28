#!/usr/bin/env python3
"""
Relativierung von Topics und Dokumenten anhand eines Termsets und TF-IDF-Werten.

Siehe Pipeline-Beschreibung im Modul-Docstring.

CLI-Beispielaufruf:

    python src/procession_termsets_topics/s03_termset-topics_dtti.py `
        --termset-file resources/termsets/Termset_Gegenstände_1.2.csv `
        --topic-word-file resources/topic-models/topics_v3/fadelive_mallet_stop_topic_words_100_words_tag.csv `
        --topic-rank-file output/processed_topics/document-topics-distribution_tag_rank.csv `
        --tfidf-file output/dtm_tfidf_stop/tfidf-2000.csv `
        --doc-topic-file resources/topic-models/topics_v3/document-topics-distribution_tag.csv `
        --dtm-file output/dtm_tfidf_stop/dtm_minfreq6.csv `
        --output-dir output/processed_termset/Termset_Gegenstände_1.2 `
        --tfidf-start-col-index 24 `
        --dtm-start-col-index 24 `
        --dtm-id-col _id
"""

from __future__ import annotations

import argparse
from math import log
from pathlib import Path
from typing import Dict, Set, List, Tuple, Union, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hilfsfunktionen zum Laden & Normalisieren
# ---------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    """Einfache Normalisierung: strip + lower."""
    return str(token).strip().lower()


def load_termset(termset_file: Path) -> Tuple[Set[str], Dict[str, str]]:
    """
    Lädt das Termset und gibt zurück:
        - Menge aller normalisierten Wörter (termset_words)
        - Mapping: Wort -> Tag (Spaltenname im Termset)
    """
    df = pd.read_csv(termset_file)
    termset_words: Set[str] = set()
    word_to_tag: Dict[str, str] = {}

    for tag in df.columns:
        values = (
            df[tag]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        for val in values:
            if not val:
                continue
            w = normalize_token(val)
            termset_words.add(w)
            word_to_tag[w] = tag

    return termset_words, word_to_tag


def load_topic_word_matrix(topic_word_file: Path) -> pd.DataFrame:
    """
    Lädt die Topic-Word-Matrix.

    Erwartung:
        - index_col=0 = Topic
        - weitere Spalten = Wörter (Top-N-Words pro Topic)
    Alle Wortzellen werden normalisiert (strip + lower).
    """
    df = pd.read_csv(topic_word_file, index_col=0)
    df = df.applymap(
        lambda x: normalize_token(x) if pd.notna(x) and str(x).strip() != "" else np.nan
    )
    return df


def load_topic_ranks(topic_rank_file: Path) -> Dict[str, int]:
    """
    Lädt die Topic-Rangliste.

    Erwartung:
        - Spalten mindestens: "Topic", "Rang"
    Rückgabe:
        - dict: Topic -> Rang (int)
    """
    df_rank = pd.read_csv(topic_rank_file)
    if "Topic" not in df_rank.columns or "Rang" not in df_rank.columns:
        raise ValueError(
            f"Erwarte Spalten 'Topic' und 'Rang' in {topic_rank_file}, "
            f"gefunden: {list(df_rank.columns)}"
        )

    df_rank["Topic"] = df_rank["Topic"].astype(str)
    df_rank["Rang"] = df_rank["Rang"].astype(int)
    return dict(zip(df_rank["Topic"], df_rank["Rang"]))


def load_tfidf_totals(
    tfidf_file: Path,
    start_col_index: int = 10,
) -> Dict[str, float]:
    """
    Lädt die TF-IDF-Matrix und berechnet für jede Wortspalte die Summe
    über alle Dokumente.

    Annahme:
        - Ab Spalte 'start_col_index' beginnen die Termspalten.
        - Diese Spalten sind numerisch (TF-IDF-Werte).

    Rückgabe:
        - dict: normalisiertes Wort -> tfidf-Summe
    """
    df_tfidf = pd.read_csv(tfidf_file)
    if df_tfidf.shape[1] <= start_col_index:
        raise ValueError(
            f"TF-IDF-Datei {tfidf_file} hat weniger als {start_col_index+1} Spalten. "
            f"Passe 'tfidf-start-col-index' an."
        )

    expr_columns = df_tfidf.columns[start_col_index:]

    numeric_cols = [
        col for col in expr_columns
        if pd.api.types.is_numeric_dtype(df_tfidf[col])
    ]

    if not numeric_cols:
        raise ValueError(
            f"In {tfidf_file} wurden ab Spalte {start_col_index} keine numerischen "
            f"TF-IDF-Spalten gefunden."
        )

    tfidf_totals = df_tfidf[numeric_cols].sum()

    tfidf_totals_norm = tfidf_totals.copy()
    tfidf_totals_norm.index = [
        normalize_token(c) for c in tfidf_totals_norm.index
    ]

    return tfidf_totals_norm.to_dict()


# ---------------------------------------------------------------------------
# 1) Termset × Topics × TF-IDF: Topic-Level-Matrix
# ---------------------------------------------------------------------------

def compute_termset_topic_matrix(
    termset_words: Set[str],
    df_topics: pd.DataFrame,
    topic_ranks: Dict[str, int],
    tfidf_sums: Dict[str, float],
) -> pd.DataFrame:
    """
    Berechnet Topic-Level-Metriken relativ zum Termset.
    """
    results: List[dict] = []

    for topic in df_topics.index:
        topic_str = str(topic)

        topic_words = (
            df_topics.loc[topic]
            .dropna()
            .astype(str)
            .tolist()
        )

        if not topic_words:
            results.append({
                "Topic": topic_str,
                "Anzahl der getaggten Ausdrücke": 0,
                "Topic-Rang": topic_ranks.get(topic_str),
                "tfidf_sum": 0.0,
                "Summierte Positionen": 0,
                "TFIDF-Positions-Score": 0.0,
            })
            continue

        # Mapping: Wort -> Position (1-basiert)
        word_pos = {word: pos for pos, word in enumerate(topic_words, start=1)}

        common_words = termset_words & set(topic_words)
        common_count = len(common_words)

        tfidf_sum = 0.0
        if common_words:
            for w in common_words:
                tfidf_sum += tfidf_sums.get(w, 0.0)

        position_sum = 0
        if common_words:
            position_sum = sum(word_pos[w] for w in common_words if w in word_pos)

        combined_score = 0.0
        if common_words:
            for w in common_words:
                tfidf_val = tfidf_sums.get(w, 0.0)
                pos = word_pos.get(w)
                if tfidf_val > 0 and pos and pos > 0:
                    combined_score += tfidf_val / log(pos + 1)

        results.append({
            "Topic": topic_str,
            "Anzahl der getaggten Ausdrücke": common_count,
            "Topic-Rang": topic_ranks.get(topic_str),
            "tfidf_sum": tfidf_sum,
            "Summierte Positionen": position_sum,
            "TFIDF-Positions-Score": combined_score,
        })

    df_result = pd.DataFrame(results)

    df_result["Getaggte-Ausdrücke-Topic Rang"] = (
        df_result["tfidf_sum"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    df_result["Positionen-Rang"] = (
        df_result["Summierte Positionen"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    df_result["TFIDF-Positions-Rang"] = (
        df_result["TFIDF-Positions-Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    final_df = df_result[[
        "Topic",
        "Anzahl der getaggten Ausdrücke",
        "Topic-Rang",
        "Getaggte-Ausdrücke-Topic Rang",
        "Summierte Positionen",
        "Positionen-Rang",
        "TFIDF-Positions-Score",
        "TFIDF-Positions-Rang",
    ]]

    return final_df


# ---------------------------------------------------------------------------
# 2) Tag–Word–Topic-Distribution
# ---------------------------------------------------------------------------

def compute_tag_word_topic_distribution(
    word_to_tag: Dict[str, str],
    df_topics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mapping der Ausdrücke auf Topics:
        - Ausdruck
        - Tag
        - Topic
        - Position im Topic (oder NaN)
    """
    alle_ausdruecke = sorted(word_to_tag.keys())
    topics = df_topics.index.tolist()

    # Lookup: (topic, wort) -> position
    topic_word_position: Dict[Tuple[str, str], int] = {}
    for topic in df_topics.index:
        words = df_topics.loc[topic].dropna().astype(str).tolist()
        for pos, wort in enumerate(words, start=1):
            topic_word_position[(topic, wort)] = pos

    rows: List[dict] = []

    for wort in alle_ausdruecke:
        tag = word_to_tag[wort]
        for topic in topics:
            pos = topic_word_position.get((topic, wort), np.nan)
            rows.append({
                "Ausdruck": wort,
                "Tag": tag,
                "Topic": str(topic),
                "Position im Topic": pos,
            })

    df_full = pd.DataFrame(rows)
    df_full = df_full.sort_values(["Ausdruck", "Topic"])
    return df_full


# ---------------------------------------------------------------------------
# 3) Wort-Positionsscore je Topic
# ---------------------------------------------------------------------------

def compute_topic_tag_word_position_score(
    word_to_tag: Dict[str, str],
    df_topics: pd.DataFrame,
    tfidf_sums: Dict[str, float],
) -> pd.DataFrame:
    """
    Berechnet für jedes (Topic, Wort, Tag) einen Positions-Score:

        relevance = TFIDF-Summe / log(pos+1)
    """
    tag_words = set(word_to_tag.keys())
    relevance_rows: List[dict] = []

    for topic in df_topics.index:
        topic_words = df_topics.loc[topic].dropna().astype(str).tolist()
        word_pos = {word: pos for pos, word in enumerate(topic_words, start=1)}
        common_words = tag_words & set(topic_words)

        for word in common_words:
            tfidf_val = tfidf_sums.get(word, 0.0)
            pos = word_pos.get(word)
            if tfidf_val > 0 and pos and pos > 0:
                relevance = tfidf_val / log(pos + 1)
                tag = word_to_tag.get(word)
                relevance_rows.append({
                    "Topic": str(topic),
                    "Word": word,
                    "Tag": tag,
                    "Position im Topic": pos,
                    "TF-IDF-Summe": tfidf_val,
                    "Positions-Score (tfidf/log(pos+1))": relevance,
                })

    df_word_relevance = pd.DataFrame(relevance_rows)
    if not df_word_relevance.empty:
        df_word_relevance.sort_values(
            ["Topic", "Positions-Score (tfidf/log(pos+1))"],
            ascending=[True, False],
            inplace=True,
        )
    return df_word_relevance


# ---------------------------------------------------------------------------
# 4) Tags pro Topic: Rang
# ---------------------------------------------------------------------------

def compute_tags_per_topic_rank(
    df_tag_word_topic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregiert die Tag-Word-Topic-Distribution zu:

        - Anzahl_Wörter pro (Topic, Tag)
        - Positions_Summe
        - Ränge pro Topic
    """
    df_full = df_tag_word_topic.copy()
    df_matched = df_full[df_full["Position im Topic"].notna()].copy()
    if df_matched.empty:
        return pd.DataFrame()

    df_matched["Position im Topic"] = df_matched["Position im Topic"].astype(int)

    df_agg = df_matched.groupby(["Topic", "Tag"]).agg(
        Anzahl_Wörter=("Ausdruck", "count"),
        Positions_Summe=("Position im Topic", "sum"),
    ).reset_index()

    df_agg["Tag-Rang (nach Anzahl)"] = (
        df_agg.groupby("Topic")["Anzahl_Wörter"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    df_agg["Tag-Rang (nach Position)"] = (
        df_agg.groupby("Topic")["Positions_Summe"]
        .rank(ascending=True, method="min")
        .astype(int)
    )

    return df_agg


# ---------------------------------------------------------------------------
# 5) Tag-Topic-Relevanz (Topic-Scores + TF-IDF-Relativierung)
# ---------------------------------------------------------------------------

def compute_tag_topic_relevance(
    df_word_relevance: pd.DataFrame,
    word_to_tag: Dict[str, str],
    tfidf_sums: Dict[str, float],
) -> pd.DataFrame:
    """
    Berechnet für jedes Tag:

        - Relevanzscore_Tag_Topic: Summe aller Positions-Scores über alle Topics
        - Relevanzscore_Tag_TFIDF: TF-IDF-basierter Score (mittlerer TF-IDF-Wert, log-gewichtet)
        - z-Score_TFIDF: z-standardisierter TF-IDF-Score
    """
    if df_word_relevance.empty:
        return pd.DataFrame(columns=[
            "Tag", "Relevanzscore_Tag_Topic",
            "Rang_Tag_Topic", "Relevanzscore_Tag_TFIDF", "z-Score_TFIDF",
        ])

    # Summe der Positions-Scores je Tag und Topic
    df_agg = (
        df_word_relevance
        .groupby(["Tag", "Topic"])["Positions-Score (tfidf/log(pos+1))"]
        .sum()
        .reset_index()
        .rename(columns={"Positions-Score (tfidf/log(pos+1))": "Relevanzsumme"})
    )

    # Relevanzscore pro Tag (Summe über alle Topics)
    tag_ranking = (
        df_agg.groupby("Tag")["Relevanzsumme"]
        .sum()
        .reset_index()
        .rename(columns={"Relevanzsumme": "Relevanzscore_Tag_Topic"})
    )
    tag_ranking["Rang_Tag_Topic"] = (
        tag_ranking["Relevanzscore_Tag_Topic"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # TF-IDF-basierter Score
    tag_scores: Dict[str, float] = {}
    tag_counts: Dict[str, int] = {}

    for word, tag in word_to_tag.items():
        tfidf_val = tfidf_sums.get(word)
        if tfidf_val is None:
            continue
        tag_scores[tag] = tag_scores.get(tag, 0.0) + tfidf_val
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    relevanz_log_gewichtet = {
        tag: (tag_scores[tag] / tag_counts[tag]) * np.log(tag_counts[tag] + 1)
        for tag in tag_scores if tag_counts[tag] > 0
    }

    df_weighted = pd.DataFrame(
        list(relevanz_log_gewichtet.items()),
        columns=["Tag", "Relevanzscore_Tag_TFIDF"],
    )

    if not df_weighted.empty:
        mean = df_weighted["Relevanzscore_Tag_TFIDF"].mean()
        std = df_weighted["Relevanzscore_Tag_TFIDF"].std()
        if std > 0:
            df_weighted["z-Score_TFIDF"] = (
                (df_weighted["Relevanzscore_Tag_TFIDF"] - mean) / std
            )
        else:
            df_weighted["z-Score_TFIDF"] = 0.0
    else:
        df_weighted["z-Score_TFIDF"] = []

    df_combined = (
        pd.merge(tag_ranking, df_weighted, on="Tag", how="inner")
        .sort_values("Relevanzscore_Tag_Topic", ascending=False)
    )

    return df_combined


# ---------------------------------------------------------------------------
# 6) Globale Wort-Relevanz (über alle Topics)
# ---------------------------------------------------------------------------

def compute_tag_word_topic_rank(
    df_word_relevance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summiert Positions-Scores über alle Topics und rankt die Wörter.
    """
    if df_word_relevance.empty:
        return pd.DataFrame(columns=["Word", "Relevanzsumme", "Rang"])

    df_global = (
        df_word_relevance
        .groupby("Word")["Positions-Score (tfidf/log(pos+1))"]
        .sum()
        .reset_index()
        .rename(columns={"Positions-Score (tfidf/log(pos+1))": "Relevanzsumme"})
    )

    df_global.sort_values("Relevanzsumme", ascending=False, inplace=True)
    df_global["Rang"] = (
        df_global["Relevanzsumme"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return df_global


# ---------------------------------------------------------------------------
# 7) Dokument–Term-Topic-Index (DTTI)
# ---------------------------------------------------------------------------

def compute_document_topic_termset_interaction(
    termset_words: Set[str],
    df_docs: pd.DataFrame,
    df_topics: pd.DataFrame,
    df_dtm: pd.DataFrame,
    tfidf_sums: Dict[str, float],
    dtm_start_col_index: int = 10,
    dtm_id_col: str = "_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Berechnet die relativierte Dokument-Topic-Matrix:

        Relevanz(D,T) =
            cos(D,T) * SUM_{w ∈ D ∩ T ∩ B}
                freq(w,D) * ( tfidf_sum(w) / log(rank_T(w) + 1) )

    Inputs:
        - termset_words: Menge B
        - df_docs: Document–Topic-Matrix (Index=Dokument, Spalten=Topics)
        - df_topics: Topic-Word-Matrix (Index=Topics, Zellen=Words)
        - df_dtm: DTM mit Frequenzen (eine Zeile pro Dokument)
        - tfidf_sums: dict Wort -> TF-IDF-Summe
    """
    # IDs als Strings
    df_docs = df_docs.copy()
    df_docs.index = df_docs.index.astype(str)

    df_dtm = df_dtm.copy()
    if dtm_id_col not in df_dtm.columns:
        raise ValueError(f"DTM-Datei enthält keine ID-Spalte '{dtm_id_col}'.")

    df_dtm[dtm_id_col] = df_dtm[dtm_id_col].astype(str)

    if df_dtm.shape[1] <= dtm_start_col_index:
        raise ValueError(
            f"DTM-Datei hat weniger als {dtm_start_col_index+1} Spalten. "
            f"Passe 'dtm-start-col-index' an."
        )

    dtm_expr_cols = df_dtm.columns[dtm_start_col_index:]

    # Numerische Frequenzen sicherstellen
    df_dtm[dtm_expr_cols] = (
        df_dtm[dtm_expr_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    relevance_rows: List[dict] = []
    rel_matrix_dict: Dict[str, Dict[str, float]] = {}

    # Vorbereiten: Topic-Wörter + Positionen
    topic_word_pos: Dict[str, Dict[str, int]] = {}
    topic_word_sets: Dict[str, Set[str]] = {}

    for topic in df_topics.index:
        t_words = df_topics.loc[topic].dropna().astype(str).tolist()
        topic_word_sets[str(topic)] = set(t_words)
        topic_word_pos[str(topic)] = {w: pos for pos, w in enumerate(t_words, start=1)}

    # Iteration über Dokumente
    for doc_id in df_docs.index:
        dtm_row = df_dtm[df_dtm[dtm_id_col] == doc_id]
        if dtm_row.empty:
            continue

        # Frequenzen: normalisierte Wortform -> freq
        freqs_raw = dtm_row[dtm_expr_cols].iloc[0].to_dict()
        word_freqs = {
            normalize_token(col): int(freq)
            for col, freq in freqs_raw.items()
            if int(freq) > 0
        }

        rel_matrix_dict[doc_id] = {}

        for topic in df_topics.index:
            topic_str = str(topic)
            topic_words = topic_word_sets[topic_str]
            word_pos = topic_word_pos[topic_str]

            common_words = set(word_freqs.keys()) & topic_words & termset_words

            score = 0.0
            for word in common_words:
                freq = word_freqs.get(word, 0)
                tfidf_val = tfidf_sums.get(word, 0.0)
                pos = word_pos.get(word)

                if freq > 0 and tfidf_val > 0 and pos and pos > 0:
                    weight = tfidf_val / log(pos + 1)  # score(w, T)
                    score += freq * weight

            # cos(D,T)
            cos_val = (
                float(df_docs.at[doc_id, topic])
                if topic in df_docs.columns
                else 0.0
            )

            rel = cos_val * score

            relevance_rows.append({
                "Document": doc_id,
                "Topic": topic_str,
                "Cosinuswert": cos_val,
                "Frequenz-basierter Score (D∩T∩B)": score,
                "Relevanz(D,T)": rel,
            })

            rel_matrix_dict[doc_id][topic_str] = rel

    df_relevance_long = pd.DataFrame(relevance_rows)

    # Matrix D×T
    df_rel_matrix = pd.DataFrame.from_dict(rel_matrix_dict, orient="index").fillna(0.0)
    df_rel_matrix.index.name = "Document"

    # Min-Max-Normalisierung (lange Form)
    if not df_relevance_long.empty:
        min_val = df_relevance_long["Relevanz(D,T)"].min()
        max_val = df_relevance_long["Relevanz(D,T)"].max()
        if max_val > min_val:
            df_relevance_long["Relevanz_MinMax"] = (
                df_relevance_long["Relevanz(D,T)"] - min_val
            ) / (max_val - min_val)
        else:
            df_relevance_long["Relevanz_MinMax"] = 0.0

    # Min-Max-Normalisierung (Matrix)
    if not df_rel_matrix.empty:
        min_val_m = df_rel_matrix.values.min()
        max_val_m = df_rel_matrix.values.max()
        if max_val_m > min_val_m:
            df_rel_matrix_norm = (df_rel_matrix - min_val_m) / (max_val_m - min_val_m)
        else:
            df_rel_matrix_norm = df_rel_matrix * 0.0
    else:
        df_rel_matrix_norm = df_rel_matrix.copy()

    return df_relevance_long, df_rel_matrix, df_rel_matrix_norm


# ---------------------------------------------------------------------------
# CLI & Run-Funktion
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relativiert Topics und Dokumente anhand eines Termsets und TF-IDF-Werten."
    )
    parser.add_argument(
        "--termset-file",
        type=Path,
        required=True,
        help="CSV mit Termset (Spalten = Tags, Zellen = Ausdrücke).",
    )
    parser.add_argument(
        "--topic-word-file",
        type=Path,
        required=True,
        help="CSV mit Topic-Word-Matrix (Index = Topic, Spalten = Wörter).",
    )
    parser.add_argument(
        "--topic-rank-file",
        type=Path,
        required=True,
        help="CSV mit Topic-Ranking (Spalten: 'Topic', 'Rang').",
    )
    parser.add_argument(
        "--tfidf-file",
        type=Path,
        required=True,
        help="TF-IDF-Matrix (Dokumente × Wörter).",
    )
    parser.add_argument(
        "--doc-topic-file",
        type=Path,
        required=True,
        help="Document-Topic-Matrix (Kosinuswerte, Index = Dokument, Spalten = Topics).",
    )
    parser.add_argument(
        "--dtm-file",
        type=Path,
        required=True,
        help="DTM mit Wortfrequenzen (eine Zeile pro Dokument).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Zielordner für die Ausgabedateien.",
    )
    parser.add_argument(
        "--tfidf-start-col-index",
        type=int,
        default=10,
        help=(
            "Index (0-basiert) der ersten TF-IDF-Wortspalte in der TF-IDF-Datei. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--dtm-start-col-index",
        type=int,
        default=10,
        help=(
            "Index (0-basiert) der ersten Wortspalte in der DTM-Datei. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--dtm-id-col",
        type=str,
        default="_id",
        help="Name der Dokument-ID-Spalte in der DTM-Datei (Default: '_id').",
    )
    return parser


def run(
    termset_file: Union[str, Path],
    topic_word_file: Union[str, Path],
    topic_rank_file: Union[str, Path],
    tfidf_file: Union[str, Path],
    doc_topic_file: Union[str, Path],
    dtm_file: Union[str, Path],
    output_dir: Union[str, Path],
    tfidf_start_col_index: int = 10,
    dtm_start_col_index: int = 10,
    dtm_id_col: str = "_id",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Führt die komplette Pipeline aus (für Programmatic-Run).

    Gibt ein Dict mit allen erzeugten DataFrames zurück.
    Schreibt zusätzlich alle CSVs in `output_dir`.
    """
    termset_file = Path(termset_file)
    topic_word_file = Path(topic_word_file)
    topic_rank_file = Path(topic_rank_file)
    tfidf_file = Path(tfidf_file)
    doc_topic_file = Path(doc_topic_file)
    dtm_file = Path(dtm_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    termset_basename = termset_file.stem

    tag_topic_rank_file = output_dir / f"{termset_basename}_tag_topic_rank.csv"
    tag_word_topic_distribution_file = output_dir / f"{termset_basename}_tag_word_topic_distribution.csv"
    topic_tag_word_position_score_file = output_dir / f"{termset_basename}_topic_tag_word_position_score.csv"
    tags_per_topic_rank_file = output_dir / f"{termset_basename}_tags_per_topic_rank.csv"
    tag_topic_relevance_file = output_dir / f"{termset_basename}_tag_topic_relevance.csv"
    tag_word_topic_rank_file = output_dir / f"{termset_basename}_tag_word_topic_rank.csv"
    dtti_list_file = output_dir / f"{termset_basename}_dtti_list.csv"
    dtti_matrix_file = output_dir / f"{termset_basename}_dtti_matrix.csv"
    dtti_matrix_norm_file = output_dir / f"{termset_basename}_dtti_matrix_norm.csv"

    # --- Daten laden ---
    termset_words, word_to_tag = load_termset(termset_file)
    df_topics = load_topic_word_matrix(topic_word_file)
    topic_ranks = load_topic_ranks(topic_rank_file)
    tfidf_sums = load_tfidf_totals(
        tfidf_file,
        start_col_index=tfidf_start_col_index,
    )

    # 1) Topic-Level-Matrix
    df_topic_matrix = compute_termset_topic_matrix(
        termset_words=termset_words,
        df_topics=df_topics,
        topic_ranks=topic_ranks,
        tfidf_sums=tfidf_sums,
    )
    df_topic_matrix.to_csv(tag_topic_rank_file, index=False, encoding="utf-8")
    if verbose:
        print(f"[OK] Tag-Topic-Rangmatrix gespeichert in: {tag_topic_rank_file}")

    # 2) Tag–Word–Topic-Distribution
    df_tag_word_topic = compute_tag_word_topic_distribution(
        word_to_tag=word_to_tag,
        df_topics=df_topics,
    )
    df_tag_word_topic.to_csv(
        tag_word_topic_distribution_file,
        index=False,
        encoding="utf-8",
    )
    if verbose:
        print(f"[OK] Tag-Word-Topic-Distribution gespeichert in: {tag_word_topic_distribution_file}")

    # 3) Wort-Positionsscore je Topic
    df_word_relevance = compute_topic_tag_word_position_score(
        word_to_tag=word_to_tag,
        df_topics=df_topics,
        tfidf_sums=tfidf_sums,
    )
    df_word_relevance.to_csv(
        topic_tag_word_position_score_file,
        index=False,
        encoding="utf-8",
    )
    if verbose:
        print(f"[OK] Topic-Tag-Word-Positionsscore gespeichert in: {topic_tag_word_position_score_file}")

    # 4) Tags pro Topic
    df_tags_per_topic = compute_tags_per_topic_rank(df_tag_word_topic)
    df_tags_per_topic.to_csv(
        tags_per_topic_rank_file,
        index=False,
        encoding="utf-8",
    )
    if verbose:
        print(f"[OK] Tags-per-Topic-Ranking gespeichert in: {tags_per_topic_rank_file}")

    # 5) Tag-Topic-Relevanz
    df_tag_topic_relevance = compute_tag_topic_relevance(
        df_word_relevance=df_word_relevance,
        word_to_tag=word_to_tag,
        tfidf_sums=tfidf_sums,
    )
    df_tag_topic_relevance.to_csv(
        tag_topic_relevance_file,
        index=False,
        encoding="utf-8",
    )
    if verbose:
        print(f"[OK] Tag-Topic-Relevanzwerte gespeichert in: {tag_topic_relevance_file}")

    # 6) Globale Wort-Relevanz
    df_word_rank = compute_tag_word_topic_rank(df_word_relevance)
    df_word_rank.to_csv(
        tag_word_topic_rank_file,
        index=False,
        encoding="utf-8",
    )
    if verbose:
        print(f"[OK] Tag-Word-Topic-Rang gespeichert in: {tag_word_topic_rank_file}")

    # 7) Dokument–Termset-Topic-Index (DTTI)
    if verbose:
        print("Starte Berechnung des Dokument-Termset-Topic-Index")
    df_docs = pd.read_csv(doc_topic_file, index_col=0)
    df_dtm = pd.read_csv(dtm_file)

    df_dtti_long, df_dtti_matrix, df_dtti_matrix_norm = compute_document_topic_termset_interaction(
        termset_words=termset_words,
        df_docs=df_docs,
        df_topics=df_topics,
        df_dtm=df_dtm,
        tfidf_sums=tfidf_sums,
        dtm_start_col_index=dtm_start_col_index,
        dtm_id_col=dtm_id_col,
    )

    df_dtti_long.to_csv(dtti_list_file, index=False, encoding="utf-8")
    if verbose:
        print(f"[OK] DTTI-Liste gespeichert in: {dtti_list_file}")

    df_dtti_matrix.to_csv(dtti_matrix_file, encoding="utf-8")
    if verbose:
        print(f"[OK] DTTI-Matrix gespeichert in: {dtti_matrix_file}")

    df_dtti_matrix_norm.to_csv(dtti_matrix_norm_file, encoding="utf-8")
    if verbose:
        print(f"[OK] Normalisierte DTTI-Matrix gespeichert in: {dtti_matrix_norm_file}")

    # Alle Ergebnisse gesammelt zurückgeben
    return {
        "df_topic_matrix": df_topic_matrix,
        "df_tag_word_topic": df_tag_word_topic,
        "df_word_relevance": df_word_relevance,
        "df_tags_per_topic": df_tags_per_topic,
        "df_tag_topic_relevance": df_tag_topic_relevance,
        "df_word_rank": df_word_rank,
        "df_dtti_long": df_dtti_long,
        "df_dtti_matrix": df_dtti_matrix,
        "df_dtti_matrix_norm": df_dtti_matrix_norm,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run(
        termset_file=args.termset_file,
        topic_word_file=args.topic_word_file,
        topic_rank_file=args.topic_rank_file,
        tfidf_file=args.tfidf_file,
        doc_topic_file=args.doc_topic_file,
        dtm_file=args.dtm_file,
        output_dir=args.output_dir,
        tfidf_start_col_index=args.tfidf_start_col_index,
        dtm_start_col_index=args.dtm_start_col_index,
        dtm_id_col=args.dtm_id_col,
        verbose=True,
    )


if __name__ == "__main__":
    main()
