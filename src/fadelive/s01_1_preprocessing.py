#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing-Script für einen Korpus im CSV/TSV-Format.

Funktionen:
- Einlesen einer Datei mit Spalte `content` (und optionalen Metadaten)
- Drei Vorverarbeitungsstufen erzeugen:
    * min  : minimale Vorverarbeitung
    * lem  : Lemmatisierung
    * stop : Lemmatisierung + Stoppwörterentfernung
- Drei Ausgabedateien speichern:
    * korpus_min.csv
    * korpus_lem.csv
    * korpus_stop.csv

- Die Pfade zu Ersetzungslisten, Stopwortlisten und der OCR-"Salat"-Liste werden als
Parameter übergeben und können versioniert werden, z.B.:

    replacements_v1.json, replacements_v2.json, replacements_v3.json
    stopwords_v1.txt, stopwords_v2.txt  

Beispielaufruf:

    python src/fadelive/s01_preprocessing.py `
        --input data/raw/korpus.csv `
        --output-dir output/processed_corpus `
        --delimiter ";" `
        --replacements resources/replacements_v1.json `
        --stopwords resources/stopwords_v1.txt `
        --salat resources/ocr_post-correction_dictionary.txt `
        --hanta-model morphmodel_ger.pgz
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import Dict, Set, Tuple, List

import pandas as pd
from HanTa import HanoverTagger as ht


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def load_replacements(path: Path) -> Dict[str, str]:
    """Lädt eine JSON-Datei mit Ersetzungspaaren {pattern: replacement}."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_word_list(path: Path) -> Set[str]:
    """Lädt eine Wortliste (eine Form pro Zeile) als Set."""
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def apply_replacements(text: str, replacements: Dict[str, str]) -> str:
    """Wendet Regex-basierte Ersetzungen an."""
    for pattern, repl in replacements.items():
        try:
            text = re.sub(pattern, repl, text)
        except re.error as exc:
            print(f"Warnung: Regex-Fehler in Muster {pattern!r}: {exc}")
    return text


EXTENDED_PUNCTUATION = string.punctuation + "»«„“§‹›—“”‘’⸗■"


def remove_punctuation(text: str) -> str:
    """Entfernt Interpunktion und spezielle Zeichen."""
    return text.translate(str.maketrans("", "", EXTENDED_PUNCTUATION))


def remove_words_by_list(text: str, removal_list: Set[str]) -> str:
    """Entfernt Tokens, die in removal_list enthalten sind."""
    tokens = re.findall(r"\b\w+\b[.,]?", text)
    cleaned = [
        tok for tok in tokens if tok.rstrip(".,").lower() not in removal_list
    ]
    return " ".join(cleaned)


def lemmatize_text(text: str, lemmatizer: ht.HanoverTagger) -> str:
    """Einfaches HanTa-based Lemmatisieren."""
    tokens = text.split()
    lemmas = [lemmatizer.analyze(tok)[0] for tok in tokens]
    return " ".join(lemmas)


# ---------------------------------------------------------
# Hauptvorverarbeitung
# ---------------------------------------------------------

def preprocess_text(
    text: str,
    *,
    replacements: Dict[str, str],
    stopwords: Set[str],
    salat: Set[str],
    lemmatizer: ht.HanoverTagger,
) -> Tuple[str, str, str]:
    """
    Erzeugt drei Vorverarbeitungsvarianten:
    - min  : minimale Vorverarbeitung
    - lem  : lemmatisierte Variante
    - stop : Lemma + Stopwortentfernung
    """
    if not isinstance(text, str) or not text.strip():
        return "", "", ""

    # --- MIN ---
    min_text = text.lower()
    min_text = apply_replacements(min_text, replacements)
    min_text = remove_words_by_list(min_text, salat)

    # --- LEM ---
    base = remove_punctuation(min_text)
    lem_text = lemmatize_text(base, lemmatizer)

    # --- STOP ---
    stop_text = remove_words_by_list(lem_text, stopwords)
    stop_text = remove_words_by_list(stop_text, salat)
    stop_text = apply_replacements(stop_text, replacements)

    return min_text, lem_text, stop_text


# ---------------------------------------------------------
# Speichern der Korpusvarianten
# ---------------------------------------------------------

def save_corpus_variants(df: pd.DataFrame, out_dir: Path, delimiter: str) -> None:
    """
    Speichert die drei Korpusvarianten in getrennte Dateien.
    """
    for variant in ("min", "lem", "stop"):
        out_df = df.copy()
        out_df["content"] = out_df[variant]
        out_df = out_df.drop(columns=["min", "lem", "stop"])
        out_path = out_dir / f"korpus_{variant}.csv"
        out_df.to_csv(out_path, sep=delimiter, encoding="utf-8", index=False)
        print(f"> Korpusvariante gespeichert: {out_path}")

# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_path: Path,
    output_dir: Path,
    delimiter: str,
    replacements_path: Path,
    stopwords_path: Path,
    salat_path: Path,
    hanta_model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lade Korpus: {input_path}")
    df = pd.read_csv(input_path, sep=delimiter, encoding="utf-8")

    if "content" not in df.columns:
        raise ValueError("Die Eingabedatei muss eine Spalte 'content' enthalten.")

    print("Lade Ressourcen …")
    replacements = load_replacements(replacements_path)
    stopwords = load_word_list(stopwords_path)
    salat = load_word_list(salat_path)
    lemmatizer = ht.HanoverTagger(hanta_model)

    print("Starte Vorverarbeitung …")
    min_list: List[str] = []
    lem_list: List[str] = []
    stop_list: List[str] = []

    for text in df["content"].astype(str):
        min_t, lem_t, stop_t = preprocess_text(
            text,
            replacements=replacements,
            stopwords=stopwords,
            salat=salat,
            lemmatizer=lemmatizer,
        )
        min_list.append(min_t)
        lem_list.append(lem_t)
        stop_list.append(stop_t)

    df["min"] = min_list
    df["lem"] = lem_list
    df["stop"] = stop_list

    print("Speichere Korpusvarianten …")
    save_corpus_variants(df, output_dir, delimiter)

    print("Fertig.")


# ---------------------------------------------------------
# Argumentparser
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt drei Vorverarbeitungsvarianten eines Korpus aus CSV/TSV."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Pfad zur Eingabedatei (CSV/TSV) mit Spalte 'content'.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Verzeichnis für Ausgabedateien.")
    parser.add_argument("--delimiter", default="\t",
                        help="Feldtrenner (Standard: Tab).")
    parser.add_argument("--replacements", type=Path, required=True,
                        help="JSON-Datei mit Ersetzungspaaren.")
    parser.add_argument("--stopwords", type=Path, required=True,
                        help="Textdatei mit Stopwörtern.")
    parser.add_argument("--salat", type=Path, required=True,
                        help="Liste mit OCR-Artefakten / Salatformen.")
    parser.add_argument("--hanta-model", type=str, default="morphmodel_ger.pgz",
                        help="Pfad zum HanTa-Modell.")
    return parser.parse_args(argv)



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
        replacements_path=args.replacements,
        stopwords_path=args.stopwords,
        salat_path=args.salat,
        hanta_model=args.hanta_model,
    )



if __name__ == "__main__":
    main()
