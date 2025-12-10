#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing-Script für einen Korpus im CSV/TSV-Format.

ÄNDERUNG v2:
- Korpus-CSVs (korpus_min.csv, korpus_lem.csv, korpus_stop.csv) enthalten
  NUR die verarbeitete Content-Spalte (content_min, content_lem, content_stop)
- Original-Content wird NICHT gespeichert
- Alle Metadaten bleiben erhalten

BUGFIX v2.1:
- identify_metadata_columns schließt nun ALLE Content-Spalten aus (content, min, lem, stop)
- Verhindert, dass mehrere Content-Spalten in den Output gelangen

Funktionen:
- Einlesen einer Datei mit Spalte `content` (und optionalen Metadaten)
- Drei Vorverarbeitungsstufen erzeugen:
    * min  : minimale Vorverarbeitung → content_min
    * lem  : Lemmatisierung → content_lem
    * stop : Lemmatisierung + Stoppwörterentfernung → content_stop
- Drei Ausgabedateien speichern:
    * korpus_min.csv (Metadaten + content_min)
    * korpus_lem.csv (Metadaten + content_lem)
    * korpus_stop.csv (Metadaten + content_stop)

Beispielaufruf:

    python s01_1_preprocessing_v2.py \
        --input data/raw/korpus.csv \
        --output-dir output/processed_corpus \
        --delimiter ";" \
        --replacements resources/replacements_v1.json \
        --stopwords resources/stopwords_v1.txt \
        --salat resources/ocr_post-correction_dictionary_v1.txt \
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
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_word_list(path: Path) -> Set[str]:
    """Lädt eine Wortliste (eine Form pro Zeile) als Set."""
    if not path or not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def apply_replacements(text: str, replacements: dict) -> str:
    """Wendet String- und Regex-Ersetzungen an."""
    
    def is_regex(pattern: str) -> bool:
        regex_indicators = [
            r'\(\?', r'\[.+\]', r'\\b', r'\\B', r'\\d', r'\\w', r'\\s',
            r'[^\\][\*\+\?]', r'\{\d+', r'^\^', r'\$$', r'[^\\]\|'
        ]
        for indicator in regex_indicators:
            if re.search(indicator, pattern):
                return True
        return False
    
    for pattern, replacement in replacements.items():
        if is_regex(pattern):
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                print(f"⚠️  Regex-Fehler: '{pattern}' - {e}")
                continue
        else:
            text = text.replace(pattern, replacement)
    
    return text


EXTENDED_PUNCTUATION = string.punctuation + "»«„§‹›—''⸗■"


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
# Metadaten-Erkennung (KORRIGIERT v2.1)
# ---------------------------------------------------------

def identify_metadata_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifiziert alle Metadaten-Spalten (= alles außer Content-Spalten).
    
    BUGFIX v2.1:
    - Schließt ALLE Content-Spalten aus: 'content', 'min', 'lem', 'stop'
    - Verhindert, dass verarbeitete Content-Spalten in Metadaten landen
    
    Returns:
        Liste der Metadaten-Spaltennamen
    """
    # ALLE Content-Spalten ausschließen
    content_columns = {"content", "min", "lem", "stop"}
    return [col for col in df.columns if col not in content_columns]


# ---------------------------------------------------------
# Speichern der Korpusvarianten (v2 - mit Bugfix)
# ---------------------------------------------------------

def save_corpus_variants(df: pd.DataFrame, out_dir: Path, delimiter: str) -> None:
    """
    Speichert die drei Korpusvarianten in getrennte Dateien.
    
    ÄNDERUNG v2:
    - Jede Datei enthält NUR die entsprechende verarbeitete Content-Spalte
    - Original-Content wird NICHT gespeichert
    - Spaltennamen: content_min, content_lem, content_stop
    
    BUGFIX v2.1:
    - identify_metadata_columns schließt nun alle Content-Spalten aus
    - Jede Datei enthält garantiert nur EINE Content-Spalte
    """
    # Metadaten OHNE Content-Spalten (content, min, lem, stop)
    metadata_cols = identify_metadata_columns(df)
    
    print(f"   📋 Metadaten-Spalten: {len(metadata_cols)} ({', '.join(metadata_cols[:5])}{'...' if len(metadata_cols) > 5 else ''})")
    
    # MIN: Metadaten + content_min (NUR diese eine Content-Spalte!)
    out_df_min = df[metadata_cols + ["min"]].copy()
    out_df_min = out_df_min.rename(columns={"min": "content_min"})
    out_path_min = out_dir / "korpus_min.csv"
    out_df_min.to_csv(out_path_min, sep=delimiter, encoding="utf-8", index=False)
    print(f"   ✅ korpus_min.csv: {len(metadata_cols)} Metadaten + content_min")
    
    # LEM: Metadaten + content_lem (NUR diese eine Content-Spalte!)
    out_df_lem = df[metadata_cols + ["lem"]].copy()
    out_df_lem = out_df_lem.rename(columns={"lem": "content_lem"})
    out_path_lem = out_dir / "korpus_lem.csv"
    out_df_lem.to_csv(out_path_lem, sep=delimiter, encoding="utf-8", index=False)
    print(f"   ✅ korpus_lem.csv: {len(metadata_cols)} Metadaten + content_lem")
    
    # STOP: Metadaten + content_stop (NUR diese eine Content-Spalte!)
    out_df_stop = df[metadata_cols + ["stop"]].copy()
    out_df_stop = out_df_stop.rename(columns={"stop": "content_stop"})
    out_path_stop = out_dir / "korpus_stop.csv"
    out_df_stop.to_csv(out_path_stop, sep=delimiter, encoding="utf-8", index=False)
    print(f"   ✅ korpus_stop.csv: {len(metadata_cols)} Metadaten + content_stop")


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
    """Führt die komplette Preprocessing-Pipeline aus."""
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Lade Korpus: {input_path}")
    df = pd.read_csv(input_path, sep=delimiter, encoding="utf-8")

    if "content" not in df.columns:
        raise ValueError("Die Eingabedatei muss eine Spalte 'content' enthalten.")

    print(f"   📊 {len(df)} Dokumente geladen")

    # Metadaten automatisch erkennen (vor Verarbeitung!)
    original_metadata = [col for col in df.columns if col != "content"]
    print(f"   📋 Erkannte Metadaten-Spalten: {len(original_metadata)}")

    print("\n📦 Lade Ressourcen …")
    replacements = load_replacements(replacements_path)
    print(f"   ✓ Replacements: {len(replacements)} Regeln")
    
    stopwords = load_word_list(stopwords_path)
    print(f"   ✓ Stopwords: {len(stopwords)} Wörter")
    
    salat = load_word_list(salat_path)
    print(f"   ✓ OCR-Artefakte: {len(salat)} Einträge")
    
    lemmatizer = ht.HanoverTagger(hanta_model)
    print(f"   ✓ HanTa-Modell: {hanta_model}")

    print("\n🔄 Starte Vorverarbeitung …")
    min_list: List[str] = []
    lem_list: List[str] = []
    stop_list: List[str] = []

    for idx, text in enumerate(df["content"].astype(str), 1):
        if idx % 500 == 0:
            print(f"   Verarbeitet: {idx}/{len(df)} Dokumente", end="\r")
        
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

    print(f"   Verarbeitet: {len(df)}/{len(df)} Dokumente ✓")

    # Verarbeitete Spalten zum DataFrame hinzufügen
    df["min"] = min_list
    df["lem"] = lem_list
    df["stop"] = stop_list

    print("\n💾 Speichere Korpusvarianten …")
    save_corpus_variants(df, output_dir, delimiter)

    print("\n" + "="*60)
    print("✅ Preprocessing erfolgreich abgeschlossen!")
    print("="*60)
    print(f"\n📁 Output-Verzeichnis: {output_dir}")
    print(f"   - korpus_min.csv ({len(original_metadata)} Metadaten + content_min)")
    print(f"   - korpus_lem.csv ({len(original_metadata)} Metadaten + content_lem)")
    print(f"   - korpus_stop.csv ({len(original_metadata)} Metadaten + content_stop)")


# ---------------------------------------------------------
# Argumentparser
# ---------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt drei Vorverarbeitungsvarianten eines Korpus aus CSV/TSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÄNDERUNG v2:
  - Jede Output-Datei enthält NUR die entsprechende Content-Spalte
  - korpus_min.csv: Metadaten + content_min
  - korpus_lem.csv: Metadaten + content_lem
  - korpus_stop.csv: Metadaten + content_stop

BUGFIX v2.1:
  - Verhindert, dass mehrere Content-Spalten in den Outputs landen
  - Korrekte Metadaten-Erkennung

Beispiel:
  python s01_1_preprocessing_v2.py \\
      --input data/raw/korpus.csv \\
      --output-dir output/processed_corpus \\
      --delimiter ";" \\
      --replacements resources/replacements_v1.json \\
      --stopwords resources/stopwords_v1.txt \\
      --salat resources/ocr_post-correction_dictionary_v1.txt \\
      --hanta-model morphmodel_ger.pgz
        """
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
                        help="Textdatei mit Stoppwörtern.")
    parser.add_argument("--salat", type=Path, required=True,
                        help="Liste mit OCR-Artefakten / Salatformen.")
    parser.add_argument("--hanta-model", type=str, default="morphmodel_ger.pgz",
                        help="Pfad zum HanTa-Modell.")
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
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
