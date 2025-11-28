#!/usr/bin/env python3
"""
Berechnung eines Topic-Rankings, Identifikation der Top-N-Dokumente pro Topic,
Mapping dieser Dokumente auf Metadaten, Jahr-Topic-Matrix sowie
Dokument-Topic-Count- und Topic-Counts-pro-Jahr-Matrizen und rangbasierten
Jahres- und Textwerten.

WICHTIG:
    Die beiden Ausgabedateien

        *_topdocs_year_value.csv
        *_topdocs_value_per_text_topic.csv

    werden NUR auf Basis

        - der Top-10-Topics (gemäß Rank in der Rang-Datei)
        - und der Texte, die in der gemappten Topdocs-Tabelle in mindestens
          einem dieser Topics unter den ersten 30 Rängen vorkommen,

    berechnet.

Beispielaufruf:

    python src/procession_termsets_topics/s02_process_topics.py `
        --input-file resources/topic-models/topics_v3/document-topics-distribution_tag.csv `
        --output-dir output/processed_topics `
        --header-row 0 `
        --index-col 0 `
        --top-n-docs 50 `
        --metadata-file data/raw/korpus.csv `
        --meta-sep ";" `
        --strip-txt-suffix
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# CLI / Argumente
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet Topic-Ranking, Top-N-Dokumente pro Topic und "
            "optional Metadaten-Mapping + diverse Aggregationen (Jahr, Texte)."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV-Datei mit Document–Topic-Distribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ordner, in dem alle Ausgabedateien gespeichert werden.",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=0,
        help="0-basierter Index der Header-Zeile (Default: 0).",
    )
    parser.add_argument(
        "--index-col",
        type=int,
        default=0,
        help=(
            "0-basierter Index der Index-Spalte (Default: 0). "
            "Wenn -1, wird keine Index-Spalte verwendet."
        ),
    )
    parser.add_argument(
        "--top-n-docs",
        type=int,
        default=50,
        help="Anzahl Top-Dokumente pro Topic (Default: 50).",
    )
    parser.add_argument(
        "--strip-txt-suffix",
        action="store_true",
        help="Entfernt '.txt' am Ende der Dokument-IDs im Index (optional).",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Optional: CSV-Datei mit Dokument-Metadaten (muss Spalte '_id' enthalten).",
    )
    parser.add_argument(
        "--meta-sep",
        type=str,
        default=";",
        help="Spaltentrenner der Metadaten-CSV (Default: ';').",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O und Kernlogik
# ---------------------------------------------------------------------------

def read_document_topic_distribution(
    path: Path,
    header_row: int = 0,
    index_col: Optional[int] = 0,
    strip_txt_suffix: bool = False,
) -> pd.DataFrame:
    """
    Liest eine Document–Topic-Distribution ein und gibt einen DataFrame zurück.
    """
    if index_col is not None and index_col < 0:
        index_col = None

    df = pd.read_csv(
        path,
        header=header_row,
        index_col=index_col,
    )

    # Spaltennamen säubern
    df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=True)

    # Index bereinigen (z. B. Dokument-ID ohne '.txt')
    if df.index.name is not None or index_col is not None:
        df.index = df.index.astype(str)
        if strip_txt_suffix:
            df.index = df.index.str.replace(".txt", "", regex=False)

    return df


def compute_topic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet Summen, Mittelwerte und Standardabweichungen über alle
    numerischen Spalten des DataFrames.

    Gibt einen DataFrame mit:
        - Topic
        - Summe
        - Mittelwert
        - Standardabweichung
        - Rang
    zurück.
    """
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError(
            "Es wurden keine numerischen Spalten gefunden. "
            "Stelle sicher, dass die Topic-Spalten numerische Werte enthalten."
        )

    sums = numeric.sum()
    means = numeric.mean()
    stds = numeric.std()

    result = pd.DataFrame(
        {
            "Summe": sums,
            "Mittelwert": means,
            "Standardabweichung": stds,
        }
    )

    result = result.reset_index().rename(columns={"index": "Topic"})
    result["Rang"] = (
        result["Summe"].rank(method="dense", ascending=False).astype(int)
    )
    result = result.sort_values(by="Rang", ascending=True).reset_index(drop=True)
    return result


def compute_top_docs_per_topic(
    df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    Ermittelt für jede numerische Topic-Spalte die Top-N Dokumente
    (basierend auf den Topic-Werten).

    Gibt einen DataFrame zurück:
        - Zeilen = Rang (1..N)
        - Spalten = Topics
        - Zellen = Dokument-IDs (Indexwerte)
    """
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError(
            "Es wurden keine numerischen Spalten gefunden. "
            "Stelle sicher, dass die Topic-Spalten numerische Werte enthalten."
        )

    output = pd.DataFrame(index=range(1, top_n + 1))

    for col in numeric.columns:
        top = numeric[col].nlargest(top_n)
        doc_ids = list(top.index.astype(str))

        if len(doc_ids) < top_n:
            doc_ids += [""] * (top_n - len(doc_ids))

        output[col] = doc_ids

    return output


# ---------------------------------------------------------------------------
# Metadaten-Mapping
# ---------------------------------------------------------------------------

def load_metadata(metadata_file: Path, sep: str = ";") -> pd.DataFrame:
    """
    Lädt eine Metadaten-CSV mit mindestens der Spalte '_id' und setzt diese
    als Index (als String).
    """
    df_meta = pd.read_csv(metadata_file, sep=sep)
    if "_id" not in df_meta.columns:
        raise ValueError(
            f"Metadaten-Datei {metadata_file} muss eine Spalte '_id' enthalten."
        )
    df_meta["_id"] = df_meta["_id"].astype(str)
    df_meta = df_meta.set_index("_id")
    return df_meta


def format_metadata_entry(doc_id: str, df_meta: pd.DataFrame) -> str:
    """
    Erzeugt einen formatierten Metadaten-String für eine gegebene Dokument-ID.
    Fallback: Wenn ID nicht gefunden wird, wird die Original-ID zurückgegeben.
    """
    doc_id = str(doc_id)
    if not doc_id or doc_id not in df_meta.index:
        return doc_id

    row = df_meta.loc[doc_id]
    parts: List[str] = []

    if "author_surname" in row.index and pd.notna(row["author_surname"]):
        parts.append(f"{row['author_surname']}:")
    if "title" in row.index and pd.notna(row["title"]):
        parts.append(f"{row['title']}.")
    if "source" in row.index and pd.notna(row["source"]):
        parts.append(f"{row['source']}.")
    year_str = None
    if "year_first" in row.index and pd.notna(row["year_first"]):
        try:
            year_str = str(int(row["year_first"]))
        except (ValueError, TypeError):
            year_str = str(row["year_first"])
    elif "year" in row.index and pd.notna(row["year"]):
        try:
            year_str = str(int(row["year"]))
        except (ValueError, TypeError):
            year_str = str(row["year"])
    if year_str is not None:
        parts.append(year_str + ".")

    return " ".join(str(p) for p in parts).strip()


def map_topdocs_to_metadata(
    topdocs_df: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wendet das Metadaten-Format auf jede Zelle der Topdocs-Matrix an.
    """
    df_out = topdocs_df.copy()
    for topic in df_out.columns:
        df_out[topic] = df_out[topic].apply(
            lambda doc_id: format_metadata_entry(doc_id, df_meta)
        )
    return df_out


# ---------------------------------------------------------------------------
# Jahr-Topic-Matrix & Jahr-Extraktion
# ---------------------------------------------------------------------------

def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extrahiert ein Jahr (1700–1999) aus einem Textstring mittels Regex.
    Gibt das Jahr als int zurück oder None.
    """
    match = re.search(r"(17|18|19)\d{2}", str(text))
    return int(match.group()) if match else None


def build_year_topic_matrix_from_mapped(
    mapped_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nimmt eine Matrix mit gemappten Metadaten-Strings (Topdocs) und erzeugt
    eine Jahr × Topic-Matrix (Texte als kommaseparierte Strings).
    """
    data: Dict[Tuple[int, str], List[str]] = {}

    for topic in mapped_df.columns:
        for doc in mapped_df[topic].dropna():
            if not doc:
                continue
            year = extract_year_from_text(doc)
            if year is None:
                continue
            data.setdefault((year, topic), []).append(str(doc))

    if not data:
        return pd.DataFrame()

    years = sorted({y for (y, _) in data.keys()})
    topics = mapped_df.columns.tolist()
    reshaped_df = pd.DataFrame(index=years, columns=topics)

    for (year, topic), docs in data.items():
        reshaped_df.at[year, topic] = ", ".join(docs)

    reshaped_df = reshaped_df.fillna("")
    return reshaped_df


# ---------------------------------------------------------------------------
# Dokument-Topic-Count-Matrix aus gemappten Metadaten
# ---------------------------------------------------------------------------

def build_document_topic_count_matrix(
    mapped_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine Dokument-Topic-Count-Matrix aus der gemappten Topdocs-Matrix.

    Rückgabe:
        - Zeilen = Dokumente (Metadaten-Strings)
        - Spalten = Topics
        - Zellen = 0/1
        - zusätzliche Spalte 'Anzahl Topics'
    """
    doc_topic_map: Dict[str, set] = {}

    for topic in mapped_df.columns:
        for doc in mapped_df[topic].dropna():
            doc_str = str(doc).strip()
            if not doc_str:
                continue
            doc_topic_map.setdefault(doc_str, set()).add(topic)

    if not doc_topic_map:
        return pd.DataFrame()

    unique_topics = list(mapped_df.columns)
    binary_matrix = pd.DataFrame(
        0, index=sorted(doc_topic_map.keys()), columns=unique_topics, dtype=int
    )

    for doc_str, topics in doc_topic_map.items():
        binary_matrix.loc[doc_str, list(topics)] = 1

    binary_matrix.insert(0, "Anzahl Topics", binary_matrix.sum(axis=1))
    binary_matrix.reset_index(inplace=True)
    binary_matrix.rename(columns={"index": "Dokument"}, inplace=True)

    return binary_matrix


# ---------------------------------------------------------------------------
# Topic-Counts pro Jahr aus Dokument-Topic-Count-Matrix
# ---------------------------------------------------------------------------

def build_topic_counts_per_year(
    doc_topic_count_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine numerische Matrix: Jahr × Topic = Anzahl Dokumente.
    """
    df = doc_topic_count_df.copy()

    if "Dokument" not in df.columns:
        raise ValueError("Erwarte Spalte 'Dokument' in der Dokument-Topic-Count-Matrix.")

    df["Jahr"] = df["Dokument"].apply(extract_year_from_text)
    df = df.dropna(subset=["Jahr"])
    df["Jahr"] = df["Jahr"].astype(int)

    non_topic_cols = {"Dokument", "Jahr", "Anzahl Topics"}
    topic_cols = [col for col in df.columns if col not in non_topic_cols]

    grouped = df.groupby("Jahr")[topic_cols].sum()
    return grouped


# ---------------------------------------------------------------------------
# Jahr → Liste wichtiger Texte aus Dokument-Topic-Count-Matrix
# ---------------------------------------------------------------------------

def build_year_document_ranking(
    doc_topic_count_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine Tabelle:

        - Zeilen = Jahre
        - Spalten = 'Anzahl', 'Dokument 1', 'Dokument 2', ...
    """
    if "Dokument" not in doc_topic_count_df.columns:
        raise ValueError("Erwarte Spalte 'Dokument' in der Dokument-Topic-Count-Matrix.")

    docs_in_order = doc_topic_count_df["Dokument"].astype(str).tolist()

    from collections import defaultdict
    jahres_map: Dict[int, List[str]] = defaultdict(list)
    seen: set[str] = set()

    for text in docs_in_order:
        if text in seen:
            continue
        year = extract_year_from_text(text)
        if year is not None:
            jahres_map[year].append(text)
            seen.add(text)

    if not jahres_map:
        return pd.DataFrame()

    max_len = max(len(doks) for doks in jahres_map.values())
    columns = ["Anzahl"] + [f"Dokument {i+1}" for i in range(max_len)]

    final_df = pd.DataFrame(index=sorted(jahres_map.keys()), columns=columns)

    for year, texts in jahres_map.items():
        final_df.at[year, "Anzahl"] = len(texts)
        final_df.loc[year, columns[1:1 + len(texts)]] = texts

    final_df = final_df.fillna("")
    return final_df


# ---------------------------------------------------------------------------
# Rangbasierte Jahreswerte (auf Basis beliebiger Ranking-Matrix)
# ---------------------------------------------------------------------------

def build_year_values_from_rank(
    ranked_topdocs: pd.DataFrame,
    year_document_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet pro Jahr einen Wert, der sich aus den Rangwerten der zugeordneten
    Texte ergibt.

    - ranked_topdocs: Ranking-Matrix (Zeilen = Ränge, Spalten = Topics, Zellen = Text)
      -> nur die Zeilen/Spalten, die du berücksichtigen willst (z.B. Top-10-Topics, Top-30).
    - year_document_df: Jahr → Texte (Anzahl + Dokument 1..N)

    Gewicht pro Text = (Anzahl Zeilen - RangIndex)
    Jahreswert = Summe der Gewichte aller Texte dieses Jahres
    """
    if ranked_topdocs.empty or year_document_df.empty:
        return pd.DataFrame()

    tmp = ranked_topdocs.copy()
    tmp = tmp.reset_index(drop=True)  # Rang 0..N-1
    tmp["Rang"] = tmp.index
    n = len(tmp)

    long_df = tmp.melt(id_vars="Rang", var_name="Topic", value_name="Text")
    long_df["Text"] = long_df["Text"].astype(str).str.strip()
    long_df = long_df[long_df["Text"].notna() & (long_df["Text"] != "")]
    long_df["Wert"] = n - long_df["Rang"]

    wert_by_text = long_df.groupby("Text", as_index=True)["Wert"].sum()

    text_cols = [c for c in year_document_df.columns if c.startswith("Dokument")]
    rows: List[Dict[str, int]] = []

    for year, row in year_document_df.iterrows():
        for col in text_cols:
            text = row.get(col)
            if pd.isna(text):
                continue
            t = str(text).strip()
            if not t:
                continue
            if t in wert_by_text.index:
                wert = int(wert_by_text.loc[t])
                rows.append({"Jahr": int(year), "Wert": wert})

    if not rows:
        return pd.DataFrame()

    df_jtw = pd.DataFrame(rows)
    df_jahreswerte = df_jtw.groupby("Jahr", as_index=False)["Wert"].sum()
    return df_jahreswerte


# ---------------------------------------------------------------------------
# Rangbasierte Textwerte (auf Basis beliebiger Ranking-Matrix)
# ---------------------------------------------------------------------------

def build_value_per_text_and_topic(
    ranked_topdocs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt eine Tabelle mit:

        - Zeilen = Texte
        - Spalten:
            - 'SummeWert' (Gesamtwert über alle Topics)
            - pro Topic eine Spalte mit Wert

    ranked_topdocs:
        - Zeilen = Ränge
        - Spalten = Topics
        - Zellen = Text
        -> nur relevanter Ausschnitt (z.B. Top-10-Topics, Top-30-Ränge).
    """
    if ranked_topdocs.empty:
        return pd.DataFrame(columns=["Text", "SummeWert"])

    df = ranked_topdocs.copy()
    df = df.reset_index(drop=True)
    n = len(df)
    df["__RANG__"] = range(n)  # 0 = beste Zeile

    df_long = df.melt(id_vars="__RANG__", var_name="Tag", value_name="Text")

    def clean_text(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v != "" else None
        return v

    df_long["Text"] = df_long["Text"].map(clean_text)
    df_long = df_long[df_long["Text"].notna()]

    df_long["Wert"] = (n - df_long["__RANG__"]).astype(int)

    summe_pro_text = (
        df_long.groupby("Text", as_index=False)["Wert"].sum()
        .rename(columns={"Wert": "SummeWert"})
    )

    tag_matrix = (
        df_long.groupby(["Text", "Tag"], as_index=False)["Wert"].sum()
        .pivot(index="Text", columns="Tag", values="Wert")
        .fillna(0)
        .astype(int)
    )

    ergebnis = (
        summe_pro_text.set_index("Text")
        .join(tag_matrix, how="left")
        .fillna(0)
        .reset_index()
    )

    tag_spalten = sorted([c for c in ergebnis.columns if c not in ("Text", "SummeWert")])
    ergebnis = ergebnis[["Text", "SummeWert"] + tag_spalten]
    ergebnis = ergebnis.sort_values("SummeWert", ascending=False)
    return ergebnis


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Top-10-Topics + Top-30-Ränge
# ---------------------------------------------------------------------------

def get_top_k_topics(topic_stats: pd.DataFrame, k: int = 10) -> List[str]:
    """
    Liefert die Namen der Top-k-Topics anhand der Spalte 'Rang'.
    """
    if "Topic" not in topic_stats.columns or "Rang" not in topic_stats.columns:
        raise ValueError("topic_stats braucht Spalten 'Topic' und 'Rang'.")
    return (
        topic_stats.sort_values("Rang", ascending=True)
        .head(k)["Topic"]
        .astype(str)
        .tolist()
    )


def build_mapped_topdocs_subset_topk(
    mapped_topdocs: pd.DataFrame,
    topic_stats: pd.DataFrame,
    top_k: int = 10,    
    max_rank: int = 0,
) -> pd.DataFrame:
    """
    Erzeugt einen Teil-DataFrame:

        - nur Top-k-Topics (gemäß topic_stats['Rang'])
        - nur die ersten max_rank Zeilen (Top-ranks)
    """
    top_topics = get_top_k_topics(topic_stats, k=top_k)
    available = [t for t in top_topics if t in mapped_topdocs.columns]
    if not available:
        return pd.DataFrame()

    sub = mapped_topdocs[available].copy()
    sub = sub.reset_index(drop=True)
    max_rank = min(max_rank, len(sub))
    sub = sub.iloc[:max_rank]
    return sub


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1) Document–Topic-Distribution einlesen
    df = read_document_topic_distribution(
        path=args.input_file,
        header_row=args.header_row,
        index_col=args.index_col,
        strip_txt_suffix=args.strip_txt_suffix,
    )

    # Output-Ordner
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = args.input_file.stem

    rank_file = output_dir / f"{basename}_rank.csv"
    topdocs_file = output_dir / f"{basename}_topdocs.csv"
    mapped_file = output_dir / f"{basename}_topdocs_mapped.csv"
    year_topic_file = output_dir / f"{basename}_topdocs_year_topic_matrix.csv"
    doc_topic_count_file = output_dir / f"{basename}_topdocs_topic_count_matrix.csv"
    topic_counts_per_year_file = output_dir / f"{basename}_topdocs_topic_counts_per_year.csv"
    year_document_map_file = output_dir / f"{basename}_topdocs_year_document_map.csv"
    year_value_file = output_dir / f"{basename}_topdocs_year_value.csv"
    text_topic_value_file = output_dir / f"{basename}_topdocs_value_per_text_topic.csv"

    # 2) Topic-Ranking
    topic_stats = compute_topic_stats(df)
    topic_stats.to_csv(rank_file, index=False, encoding="utf-8")
    print(f"[OK] Topic-Ranking gespeichert in: {rank_file}")

    # 3) Top-N-Dokumente pro Topic
    topdocs_df = compute_top_docs_per_topic(df, top_n=args.top_n_docs)
    topdocs_df.to_csv(topdocs_file, index=False, encoding="utf-8")
    print(f"[OK] Top-{args.top_n_docs}-Dokumente pro Topic gespeichert in: {topdocs_file}")

    # 4) Metadaten-abhängige Schritte
    if args.metadata_file is not None:
        print(f"[INFO] Metadaten werden aus {args.metadata_file} geladen …")
        df_meta = load_metadata(args.metadata_file, sep=args.meta_sep)

        # 4a) IDs → Metadaten-Strings
        topdocs_mapped = map_topdocs_to_metadata(topdocs_df, df_meta)
        topdocs_mapped.to_csv(mapped_file, index=False, encoding="utf-8")
        print(f"[OK] Gemappte Top-Dokumente gespeichert in: {mapped_file}")

        # 4b) Jahr × Topic-Matrix (Texte in Zellen)
        year_topic_df = build_year_topic_matrix_from_mapped(topdocs_mapped)
        if not year_topic_df.empty:
            year_topic_df.to_csv(year_topic_file, encoding="utf-8", index_label="Jahr")
            print(f"[OK] Jahr-Topic-Matrix gespeichert in: {year_topic_file}")
        else:
            print("[WARN] Keine Jahre in Metadaten-Strings gefunden – keine Jahr-Topic-Matrix erzeugt.")

        # 4c) Dokument-Topic-Count-Matrix
        doc_topic_count_df = build_document_topic_count_matrix(topdocs_mapped)
        if not doc_topic_count_df.empty:
            doc_topic_count_df.to_csv(doc_topic_count_file, index=False, encoding="utf-8")
            print(f"[OK] Dokument-Topic-Count-Matrix gespeichert in: {doc_topic_count_file}")

            # 4d) Topic-Counts pro Jahr
            topic_counts_per_year_df = build_topic_counts_per_year(doc_topic_count_df)
            if not topic_counts_per_year_df.empty:
                topic_counts_per_year_df.to_csv(
                    topic_counts_per_year_file,
                    encoding="utf-8",
                    index_label="Jahr",
                )
                print(f"[OK] Topic-Counts pro Jahr gespeichert in: {topic_counts_per_year_file}")
            else:
                print("[WARN] Keine gültigen Jahresangaben für Topic-Counts-pro-Jahr gefunden.")

            # 4e) Jahr → Liste der wichtigsten Texte
            year_document_df = build_year_document_ranking(doc_topic_count_df)
            if not year_document_df.empty:
                year_document_df.to_csv(
                    year_document_map_file,
                    encoding="utf-8",
                    index_label="Jahr",
                )
                print(f"[OK] Jahr-Dokument-Matrix gespeichert in: {year_document_map_file}")
            else:
                print("[WARN] Keine gültigen Jahresangaben für Jahr-Dokument-Matrix gefunden.")

            # 4f) BESCHRÄNKTE RANG-AUSGABEN:
            #     Nur Top-10-Topics + Texte, die dort in den ersten 30 Rängen vorkommen.

            top10_sub = build_mapped_topdocs_subset_topk(
                mapped_topdocs=topdocs_mapped,
                topic_stats=topic_stats,
                top_k=10,
                max_rank=30,
            )

            if not top10_sub.empty and not year_document_df.empty:
                # 4f.1) value_per_text_topic (beschränkt)
                value_per_text_topic_df = build_value_per_text_and_topic(top10_sub)
                if not value_per_text_topic_df.empty:
                    value_per_text_topic_df.to_csv(
                        text_topic_value_file,
                        index=False,
                        encoding="utf-8",
                    )
                    print(f"[OK] (Top-10 / Top-30) Text-Topic-Werte gespeichert in: {text_topic_value_file}")
                else:
                    print("[WARN] Keine rangbasierten Text-Topic-Werte (Top-10/Top-30) erzeugt.")

                # 4f.2) year_value (beschränkt)
                year_value_df = build_year_values_from_rank(top10_sub, year_document_df)
                if not year_value_df.empty:
                    year_value_df.to_csv(year_value_file, index=False, encoding="utf-8")
                    print(f"[OK] (Top-10 / Top-30) Jahreswerte gespeichert in: {year_value_file}")
                else:
                    print("[WARN] Keine rangbasierten Jahreswerte (Top-10/Top-30) erzeugt.")
            else:
                print("[WARN] Kein gültiges Subset für Top-10-Topics/Top-30-Ränge oder keine Jahr-Dokument-Matrix verfügbar.")


if __name__ == "__main__":
    main()
