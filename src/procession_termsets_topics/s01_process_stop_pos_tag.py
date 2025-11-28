"""
Werkzeuge zur Verarbeitung von POS-getaggten Wortlisten, TF-IDF-Ranglisten
und Tag-Statistiken.

Zentrale Input-Datei ist eine getaggte pos-getaggte Wortliste wie: vocab_top5000_stop_pos_tag_v1.csv

Grundlage der Tag-Liste ist: vocab_top5000_stop_pos.csv

--pos-file resources/stop_pos_tag/vocab_top5000_stop_pos_tag_v3.csv `

Beispielaufrufe:

    Vollständige Pipeline:
    - TF-IDF-Datei mit Tags anreichern
    - POS-Datei mit kombinierter 'tag'-Spalte erzeugen
    - Tag-Statistiken/Matrizen aus der kombinierten POS-Datei
    - Tagset nach TF-IDF sortieren
   
    Ausgabe:
       <output-dir>/<basisname>_tfidf_tagged.csv
       <output-dir>/<basisname>_combined.csv
       <output-dir>/<basisname>_tagset.csv
       <output-dir>/<basisname>_tagset_sorted.csv
       <output-dir>/tag_stats/...
    
       (basisname = Dateiname ohne .csv und ohne _v1/_v2/... am Ende)
    
    python src/procession_termsets_topics/s01_process_stop_pos_tag.py `
        pipeline `
        --pos-file resources/stop_pos_tag/vocab_top5000_stop_pos_tag_v3.csv `
        --tfidf-file output/tfidf_rank/tfidf-2000_vocab_rank.csv `
        --output-dir output/processed_tag `
        --max-combo-size 3


Beschreibung der Eingabedateien:

    POS-Datei (--pos-file / --input-file):
        CSV mit mindestens den Spalten:
            - "word"
            - "pos"
            - "count"
            - "tag1"
            - "tag2"
            - "tag3"
        Für build-tag-stats / pipeline außerdem:
            - "tag" (kommagetrennte Tags, z.B. durch 'combine-tags' erzeugt)

    TF-IDF-Datei (--tfidf-file):
        CSV, bei der nur die erste Spalte wichtig ist:
            - Spalte 0 enthält das Wort.
        Alle weiteren Spalten (TF-IDF-Werte etc.) werden unverändert übernommen.
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Gemeinsame Hilfsfunktionen
# ---------------------------------------------------------------------------

def read_csv_clean(path: Path) -> pd.DataFrame:
    """CSV einlesen und Spaltennamen säubern (Whitespaces & BOM entfernen)."""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
    return df


def ensure_parent_dir(path: Path) -> None:
    """Parent-Verzeichnis für eine Datei anlegen, falls nötig."""
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_tag_list(val) -> List[str]:
    """Kommagetrennte Tagliste robust parsen (ohne leere/nan-Einträge)."""
    if pd.isna(val):
        return []
    parts = [t.strip() for t in str(val).split(",")]
    return [p for p in parts if p]


def clean_basename_from_version(stem: str) -> str:
    """
    Entfernt ein optionales Suffix wie '_v1', '_v2', ... vom Dateinamen-Stem.

    Beispiele:
        'vocab_top5000_stop_pos_tag_v1' -> 'vocab_top5000_stop_pos_tag'
        'tagliste_v12'                  -> 'tagliste'
        'lemma_liste'                   -> 'lemma_liste' (unverändert)
    """
    parts = stem.rsplit("_v", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


# ---------------------------------------------------------------------------
# Funktion 1: Tags an TF-IDF anhängen
# ---------------------------------------------------------------------------

def add_tags_to_tfidf(
    pos_file: Path,
    tfidf_file: Path,
    output_file: Path,
    tag_cols=("tag1", "tag2", "tag3"),
) -> None:
    """
    Verknüpft POS-Tags aus pos_file mit der TF-IDF-Tabelle tfidf_file.

    Es wird auf dem Wort gematcht:
        - in pos_file: Spalte "word"
        - in tfidf_file: erste Spalte (Index 0)

    Die Tag-Spalten werden nach vorne gezogen und in output_file geschrieben.
    """
    df_pos = read_csv_clean(pos_file)
    if "word" not in df_pos.columns:
        raise ValueError(
            f"Erwarte Spalte 'word' in {pos_file}, gefunden: {list(df_pos.columns)}"
        )

    missing_tags = [c for c in tag_cols if c not in df_pos.columns]
    if missing_tags:
        raise ValueError(
            f"Fehlende Tag-Spalten in {pos_file}: {missing_tags}. "
            f"Vorhandene Spalten: {list(df_pos.columns)}"
        )

    # Wortspalten trimmen, um Merge robuster zu machen
    df_pos["word"] = df_pos["word"].astype(str).str.strip()

    df_tfidf = read_csv_clean(tfidf_file)
    tfidf_word_col = df_tfidf.columns[0]
    df_tfidf[tfidf_word_col] = df_tfidf[tfidf_word_col].astype(str).str.strip()

    pos_word_col = "word"

    df_merged = df_tfidf.merge(
        df_pos[[pos_word_col, *tag_cols]],
        left_on=tfidf_word_col,
        right_on=pos_word_col,
        how="left",
    )

    if pos_word_col != tfidf_word_col and pos_word_col in df_merged.columns:
        df_merged.drop(columns=[pos_word_col], inplace=True)

    tag_columns = list(tag_cols)
    rest_columns = [c for c in df_merged.columns if c not in tag_columns]
    df_merged = df_merged[tag_columns + rest_columns]

    ensure_parent_dir(output_file)
    df_merged.to_csv(output_file, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Funktion 2: tag1–tag3 zu 'tag' kombinieren
# ---------------------------------------------------------------------------

def combine_tags_in_pos_file(
    input_file: Path,
    output_file: Path,
    tag_cols=("tag1", "tag2", "tag3"),
) -> None:
    """
    Erzeugt eine kombinierte kommagetrennte Tag-Spalte ('tag')
    aus tag1–tag3 und setzt sie als erste Spalte.
    """
    df = read_csv_clean(input_file)

    missing = [c for c in tag_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Tag-Spalten in {input_file}: {missing}")

    def collect(row):
        tags = []
        for col in tag_cols:
            val = row[col]
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    tags.append(s)
        return ", ".join(tags)

    df["tag"] = df.apply(collect, axis=1)
    cols = ["tag"] + [c for c in df.columns if c != "tag"]
    df = df[cols]

    ensure_parent_dir(output_file)
    df.to_csv(output_file, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Funktion 3: Tag-Statistiken & Matrizen + Tagset
# ---------------------------------------------------------------------------

def build_tag_stats_and_matrices(
    input_file: Path,
    output_dir: Path,
    tagset_output: Path,
    max_combo_size: int = 3,
) -> None:
    """
    Erwartet eine CSV mit Spalten:
        - 'tag'           (kommagetrennte Tags)
        - 'word_original' oder 'word'
        - 'count'         (Frequenz)

    Erzeugt im output_dir:
        - word_tag_sparse-binary.csv
        - tag_frequency_single.csv
        - tag_frequency_combin.csv
        - tag_word_sparse_binary.csv

    Und zusätzlich im übergeordneten Ordner:
        - tagset_output (z.B. <basisname>_tagset.csv)
    """
    df = read_csv_clean(input_file)

    if "tag" not in df.columns:
        raise ValueError(f"Erwarte Spalte 'tag' in {input_file}")

    # Wortspalte harmonisieren
    if "word_original" in df.columns:
        word_col = "word_original"
    elif "word" in df.columns:
        word_col = "word"
    else:
        raise ValueError(f"Erwarte Spalte 'word_original' oder 'word' in {input_file}")

    if "count" not in df.columns:
        raise ValueError(f"Erwarte Spalte 'count' in {input_file}")

    df["tag_list"] = df["tag"].apply(parse_tag_list)

    alle_tags = sorted({tag for tags in df["tag_list"] for tag in tags})

    # Tag-Spalten (Sparse-Binary) erzeugen
    for tag in alle_tags:
        df[tag] = df["tag_list"].apply(lambda tags: int(tag in tags))

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) word_tag_sparse-binary.csv
    sparse_path = output_dir / "word_tag_sparse-binary.csv"
    df_export = df[[word_col, "tag"] + alle_tags].copy()
    df_export.to_csv(sparse_path, index=False, encoding="utf-8")

    # 2) Tag-Frequenzliste (einzelne Tags)
    einzel_counter = Counter(tag for tags in df["tag_list"] for tag in tags)
    df_einzel = (
        pd.DataFrame(einzel_counter.items(), columns=["Tag", "Häufigkeit"])
        .sort_values(by="Häufigkeit", ascending=False)
    )
    df_einzel.to_csv(output_dir / "tag_frequency_single.csv", index=False, encoding="utf-8")

    # 3) Tag-Kombinationen (pro Zeile vollständige Kombi)
    kombis = df["tag_list"].apply(lambda tags: tuple(sorted(tags)))
    kombi_counter = Counter(kombis)
    df_kombi = pd.DataFrame(
        [{"Tag-Kombination": ", ".join(k), "Häufigkeit": v}
         for k, v in kombi_counter.items()]
    )
    df_kombi.to_csv(output_dir / "tag_frequency_combin.csv", index=False, encoding="utf-8")

    # 4) Transponierte Matrix (Tags als Zeilen, Wörter als Spalten)
    alle_wörter = df[word_col].astype(str).tolist()
    matrix = pd.DataFrame(0, index=alle_tags, columns=alle_wörter)

    for _, row in df.iterrows():
        wort = str(row[word_col])
        for tag in row["tag_list"]:
            matrix.at[tag, wort] = 1

    # Tokens und Types vorne einfügen
    matrix.insert(0, "Tokens", 0)
    matrix.insert(1, "Types", 0)

    for tag in matrix.index:
        mask = df["tag_list"].apply(lambda tags: tag in tags)
        matrix.at[tag, "Tokens"] = df.loc[mask, "count"].sum()
        matrix.at[tag, "Types"] = df.loc[mask, word_col].nunique()

    matrix_path = output_dir / "tag_word_sparse_binary.csv"
    matrix.to_csv(matrix_path, index=True, encoding="utf-8")

    # 5) tagset – Kombinationsmatrix mit Tags als Spalten, Wörtern als Zeilen
    kombis_to_words: Dict[str, List[str]] = defaultdict(list)

    for _, row in df.iterrows():
        tags = sorted(set(row["tag_list"]))
        wort = str(row[word_col])

        for r in range(1, min(max_combo_size, len(tags)) + 1):
            for combo in combinations(tags, r):
                key = ", ".join(combo)
                kombis_to_words[key].append(wort)

    if kombis_to_words:
        max_len = max(len(words) for words in kombis_to_words.values())
        sorted_kombis = sorted(kombis_to_words.keys())
        columns_data: Dict[str, List[str]] = {}

        for key in sorted_kombis:
            col = kombis_to_words[key] + [""] * (max_len - len(kombis_to_words[key]))
            columns_data[key] = col

        kombi_matrix_df = pd.DataFrame(columns_data)
        ensure_parent_dir(tagset_output)
        kombi_matrix_df.to_csv(tagset_output, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Funktion 4: Tagset nach TF-IDF sortieren
# ---------------------------------------------------------------------------

def sort_tagset_by_tfidf(
    tagset_file: Path,
    tfidf_file: Path,
    output_file: Path,
) -> None:
    """
    Sortiert jede Spalte der tagset-Datei nach TF-IDF-Ranking
    und speichert das Ergebnis in output_file.
    """
    # 1. TF-IDF-Datei laden: nur die erste Spalte zählt
    df_rank = pd.read_csv(tfidf_file, header=None)
    ranking_liste = df_rank.iloc[:, 0].astype(str).str.strip().tolist()
    ranking_map = {wort: i for i, wort in enumerate(ranking_liste)}

    # 2. Tagset laden
    df_tagset = pd.read_csv(tagset_file, dtype=str)
    df_tagset = df_tagset.applymap(lambda x: x.strip() if isinstance(x, str) else None)

    # 3. Sortieren
    df_sorted = pd.DataFrame()
    max_len = df_tagset.shape[0]

    for col in df_tagset.columns:
        werte = df_tagset[col].dropna().tolist()
        werte_sorted = sorted(
            werte,
            key=lambda x: ranking_map.get(x, float("inf"))
        )
        werte_sorted += [None] * (max_len - len(werte_sorted))
        df_sorted[col] = werte_sorted

    # 4. Speichern
    ensure_parent_dir(output_file)
    df_sorted.to_csv(output_file, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Full-Pipeline: alles in einem
# ---------------------------------------------------------------------------

def run_pipeline(
    pos_file: Path,
    tfidf_file: Path,
    output_dir: Path,
    max_combo_size: int = 3,
) -> None:
    """
    Vollständige Pipeline:

        1. TF-IDF-Datei mit tag1–tag3 anreichern
           -> <output-dir>/<basisname>_tfidf_tagged.csv
        2. POS-Datei um kombinierte 'tag'-Spalte erweitern
           -> <output-dir>/<basisname>_combined.csv
        3. Tag-Statistiken/Matrizen aus combined-POS-Datei
           -> <output-dir>/tag_stats/...
        4. tagset erstellen
           -> <output-dir>/<basisname>_tagset.csv
        5. tagset nach TF-IDF sortieren
           -> <output-dir>/<basisname>_tagset_sorted.csv

    basisname: Dateiname von pos_file ohne .csv und ohne _vX-Suffix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # zentrale Basisvariable
    base_input_name = pos_file.stem                   # z.B. "vocab_top5000_stop_pos_tag_v1"
    base_name_clean = clean_basename_from_version(base_input_name)
    # z.B. "vocab_top5000_stop_pos_tag"

    tfidf_output = output_dir / f"{base_name_clean}_tfidf_tagged.csv"
    pos_output = output_dir / f"{base_name_clean}_combined.csv"
    stats_dir = output_dir / "tag_stats"
    tagset_output = output_dir / f"{base_name_clean}_tagset.csv"
    tagset_sorted_output = output_dir / f"{base_name_clean}_tagset_sorted.csv"

    print("[1/5] POS-Tags an TF-IDF-Rangliste anhängen …")
    add_tags_to_tfidf(
        pos_file=pos_file,
        tfidf_file=tfidf_file,
        output_file=tfidf_output,
    )
    print(f"    -> geschrieben: {tfidf_output}")

    print("[2/5] POS-Datei um kombinierte 'tag'-Spalte erweitern …")
    combine_tags_in_pos_file(
        input_file=pos_file,
        output_file=pos_output,
    )
    print(f"    -> geschrieben: {pos_output}")

    print("[3/5] Tag-Statistiken und Matrizen erzeugen …")
    build_tag_stats_and_matrices(
        input_file=pos_output,
        output_dir=stats_dir,
        tagset_output=tagset_output,
        max_combo_size=max_combo_size,
    )
    print(f"    -> Tag-Statistiken in: {stats_dir}")
    print(f"    -> Tagset-Datei: {tagset_output}")

    print("[4/5] Tagset nach TF-IDF-Ranking sortieren …")
    sort_tagset_by_tfidf(
        tagset_file=tagset_output,
        tfidf_file=tfidf_file,
        output_file=tagset_sorted_output,
    )
    print(f"    -> Sorted-Tagset gespeichert: {tagset_sorted_output}")

    print("[5/5] Pipeline abgeschlossen.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process POS-tagged vocab + TF-IDF ranking."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Subcommand: add-tags-to-tfidf
    p_add = sub.add_parser(
        "add-tags-to-tfidf",
        help="Fügt tag1–tag3 an TF-IDF-Rangliste an."
    )
    p_add.add_argument("--pos-file", type=Path, required=True)
    p_add.add_argument("--tfidf-file", type=Path, required=True)
    p_add.add_argument("--output-file", type=Path, required=True)

    # Subcommand: combine-tags
    p_comb = sub.add_parser(
        "combine-tags",
        help="Erzeugt kombinierte Tag-Spalte aus tag1–tag3."
    )
    p_comb.add_argument("--input-file", type=Path, required=True)
    p_comb.add_argument("--output-file", type=Path, required=True)

    # Subcommand: build-tag-stats
    p_stats = sub.add_parser(
        "build-tag-stats",
        help="Erzeugt Tag-Statistiken und Matrizen aus einer kombinierten POS-Datei."
    )
    p_stats.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV mit Spalten 'tag', 'word' oder 'word_original', 'count'."
    )
    p_stats.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Zielordner für Statistiken und Matrizen."
    )
    p_stats.add_argument(
        "--max-combo-size",
        type=int,
        default=3,
        help="Maximale Größe der Tag-Kombinationen im tagset (Default: 3)."
    )

    # Subcommand: pipeline (alles in einem)
    p_pipe = sub.add_parser(
        "pipeline",
        help="Führt add-tags-to-tfidf + combine-tags + build-tag-stats + Sortierung in Folge aus."
    )
    p_pipe.add_argument("--pos-file", type=Path, required=True)
    p_pipe.add_argument("--tfidf-file", type=Path, required=True)
    p_pipe.add_argument("--output-dir", type=Path, required=True)
    p_pipe.add_argument(
        "--max-combo-size",
        type=int,
        default=3,
        help="Maximale Größe der Tag-Kombinationen im tagset (Default: 3)."
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "add-tags-to-tfidf":
        add_tags_to_tfidf(
            pos_file=args.pos_file,
            tfidf_file=args.tfidf_file,
            output_file=args.output_file
        )

    elif args.cmd == "combine-tags":
        combine_tags_in_pos_file(
            input_file=args.input_file,
            output_file=args.output_file
        )

    elif args.cmd == "build-tag-stats":
        base_input_name = args.input_file.stem
        base_name_clean = clean_basename_from_version(base_input_name)
        tagset_output = args.output_dir / f"{base_name_clean}_tagset.csv"

        build_tag_stats_and_matrices(
            input_file=args.input_file,
            output_dir=args.output_dir,
            tagset_output=tagset_output,
            max_combo_size=args.max_combo_size,
        )

    elif args.cmd == "pipeline":
        run_pipeline(
            pos_file=args.pos_file,
            tfidf_file=args.tfidf_file,
            output_dir=args.output_dir,
            max_combo_size=args.max_combo_size,
        )


if __name__ == "__main__":
    main()
