#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Erstellt Ranglisten von Dokumenten auf Basis aller TF-IDF-2000-Matrizen.

Für alle CSV-Dateien in einem Eingabeordner (rekursiv), deren Dateiname
"tfidf-2000" enthält, wird:

    1) die TF-IDF-Matrix eingelesen,
    2) die relevanten TF-IDF-Spalten identifiziert,
    3) die Werte gefiltert (0 < tf-idf < 0.9),
    4) die wichtigsten Terme (Top-N) ermittelt,
    5) pro Dokument eine Summenkennzahl ("combined_sum") berechnet,
    6) eine Rangliste inkl. Metadaten aus der TF-IDF-Datei erstellt,
    7) eine Vergleichsmatrix der Top-N-Terme ausgegeben.

Ausgaben (pro tfidf-2000-Datei):

    <basisname>_stop_doc_rank.csv   – Rangliste mit Metadaten
    <basisname>_stop_vocab_rank.csv – TF-IDF-Werte der Top-N Terme (Term x Dokument)

Beispielaufruf:

    python src/fadelive/s06_tfidf_rank.py `
        --input-dir output `
        --output-dir output/tfidf_rank `
        --top-n 2000
"""

import argparse
import os
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def collect_tfidf_files(input_dir: Path) -> list[Path]:
    """Sammelt rekursiv alle CSV-Dateien, die 'tfidf-2000' im Namen tragen."""
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.endswith(".csv") and "tfidf-2000" in fname:
                files.append(Path(root) / fname)
    return files


def process_tfidf_file(
    tfidf_path: Path,
    output_dir: Path,
    top_n_terms: int = 2000,
):
    """Verarbeitet eine einzelne tfidf-2000-Datei (2 Outputs pro Datei)."""
    print(f"➡ Verarbeite TF-IDF-Datei: {tfidf_path}")

    try:
        df = pd.read_csv(tfidf_path, encoding="utf-8")
    except Exception as e:
        print(f"   ⚠️ Fehler beim Einlesen, übersprungen: {e}")
        return

    # ID-Spalte finden (_id oder id)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
        df = df.rename(columns={"id": "_id"})
    elif "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
    else:
        print(f"   ⚠️ Datei ohne 'id' oder '_id' – übersprungen: {tfidf_path}")
        return

    # Metadaten-Spalten, die NICHT als TF-IDF-Features interpretiert werden sollen
    META_COLS = {
        "_id", "author_prename", "author_surname", "title", "source", "year",
        "editor_prename", "editor_surname", "volume", "title_addition",
        "year_first", "edition", "issue", "pages", "pages_exzerpt", "archive",
        "author_address", "address", "genre", "textclass", "note",
        "female_education", "author_address_geo", "address_geo"
    }

    meta_cols_in_df = [c for c in df.columns if c in META_COLS]
    tfidf_cols = [c for c in df.columns if c not in META_COLS]

    if not tfidf_cols:
        print(f"   ⚠️ Keine TF-IDF-Spalten erkannt – übersprungen: {tfidf_path}")
        return

    # TF-IDF als numerisch erzwingen
    df[tfidf_cols] = df[tfidf_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Filter: nur Werte im Bereich (0, 0.9)
    masked = df[tfidf_cols].where((df[tfidf_cols] > 0) & (df[tfidf_cols] < 0.9), 0.0)

    # Wichtigste Terme nach aufsummierter (gefilterter) Stärke
    tfidf_summen = masked.sum(axis=0)

    # Begrenzen auf vorhandene Terme
    top_n = min(top_n_terms, len(tfidf_summen))
    if top_n == 0:
        print(f"   ⚠️ Keine sinnvollen TF-IDF-Werte in Datei – übersprungen: {tfidf_path}")
        return

    top_terms = tfidf_summen.nlargest(top_n).index.tolist()

    # Summenbildung je Dokument über diese Top-Terme
    df["combined_sum"] = df[top_terms].sum(axis=1)

    # Rangbildung
    df_sorted = df.sort_values(by="combined_sum", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = range(1, len(df_sorted) + 1)

    # Rangliste mit Metadaten aus derselben Datei
    # Nur Metadaten-Spalten, die tatsächlich vorhanden sind (ohne _id doppelt)
    meta_cols_no_id = [c for c in meta_cols_in_df if c != "_id"]
    rank_cols = ["_id"] + meta_cols_no_id + ["combined_sum", "rank"]
    rank_with_meta = df_sorted[rank_cols].copy()
    rank_with_meta = rank_with_meta.rename(columns={"_id": "id"})

    # Vergleichsmatrix: Term x Dokument (nur Top-Terme)
    vergleich_df = df_sorted.set_index("_id")[top_terms].transpose()
    vergleich_df.columns.name = None

    # Dateinamen ableiten
    basisname = tfidf_path.stem  # Dateiname ohne .csv
    rang_full_pfad = output_dir / f"{basisname}_doc_rank.csv"
    vergleichspfad = output_dir / f"{basisname}_vocab_rank.csv"

    # Speichern im gewählten output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_with_meta.to_csv(rang_full_pfad, index=False, encoding="utf-8")
    vergleich_df.to_csv(vergleichspfad, encoding="utf-8")

    print(f"   ✔ Rangliste gespeichert: {rang_full_pfad}")
    print(f"   ✔ Vergleichsmatrix gespeichert: {vergleichspfad}")


# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_dir: Path,
    output_dir: Path,
    top_n: int = 2000,
) -> None:
    """Erzeugt für jede gefundene tfidf-2000-CSV-Datei genau zwei Outputs im output_dir."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    print(f"📁 Eingabeordner: {input_dir}")
    print(f"📁 Ausgabeordner: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔎 Suche tfidf-2000-Dateien …")
    files = collect_tfidf_files(input_dir)

    if not files:
        print("⚠️ Keine tfidf-2000-Dateien gefunden.")
        return

    print(f"✔ {len(files)} Datei(en) gefunden.\n")

    for f in files:
        process_tfidf_file(
            tfidf_path=f,
            output_dir=output_dir,
            top_n_terms=top_n,
        )

    print("\n✅ Verarbeitung abgeschlossen.")


# ---------------------------------------------------------
# Argumente (mit optionalem argv)
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt Ranglisten aus allen tfidf-2000-CSV-Dateien eines Ordners."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Eingabeordner, in dem tfidf-2000-CSV-Dateien gesucht werden (rekursiv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Zielordner für Ranglisten & Vergleichsmatrizen.",
    )
    parser.add_argument(
        "--top-n",
        default=2000,
        type=int,
        help="Anzahl der wichtigsten Terme (Standard: 2000).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Main (CLI-Wrapper)
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
