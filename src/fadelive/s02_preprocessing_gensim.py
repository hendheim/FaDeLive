#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vorverarbeitung eines Korpus für Gensim-Modelle (Word2Vec, LDA, etc.).

Pipeline:
    1) Lowercasing
    2) Anwenden einer Ersetzungsliste (JSON)
    3) Normalisierung von Sonderzeichen:
         - alle Sonderzeichen werden zu Leerzeichen,
         - ., !, ? werden als eigene Tokens erhalten
    4) Entfernen von OCR-Artefakten ("Salat")
    5) HanTa-Lemmatisierung (., !, ? bleiben als eigene Tokens erhalten)
    6) Entfernen von Stopwörtern (., !, ? bleiben erhalten)
    7) (optional) Entfernen von ., !, ? aus dem finalen Text

Input:
    Eine CSV/TSV-Datei (z. B. korpus_min.csv), erzeugt durch das Preprocessing-Skript.
    Muss eine Spalte "content" enthalten.

Output:
    Eine Datei korpus_gen.csv (oder benutzerdefiniert), die vollständig
    für Gensim-Modelle geeignet ist.

Beispielaufruf:

    python src/fadelive/s02_preprocessing_gensim.py `
        --input output/processed_corpus/korpus_min.csv `
        --output output/processed_corpus/korpus_gen.csv `
        --delimiter ";" `
        --replacements resources/replacements_v1.json `
        --stopwords resources/stopwords_v1.txt `
        --salat resources/ocr_post-correction_dictionary.txt `
        --hanta-model resources/morphmodel_ger.pgz `
        --remove-sentence-punct   # optional: . ! ? am Ende entfernen
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from HanTa import HanoverTagger as ht


# ---------------------------------------------------------
# Konfiguration / Parameter
# ---------------------------------------------------------

# Satzzeichen, die explizit als eigene Tokens erhalten werden sollen
ALLOWED_PUNCT = {".", "!", "?"}

# Standardwerte (werden vom CLI überschrieben)
DEFAULT_DELIMITER = "\t"
DEFAULT_HANTA_MODEL = "morphmodel_ger.pgz"


# ---------------------------------------------------------
# Ressourcen laden
# ---------------------------------------------------------

def load_list(path: Path | None) -> set:
    """Lädt eine Wortliste (Stopwörter, OCR-Salat) als Set."""
    if path is None:
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"⚠️  Warnung: Liste nicht gefunden: {path}")
        return set()


def load_replacements(path: Path | None) -> dict:
    """Lädt eine JSON-Ersetzungsliste."""
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warnung: Ersetzungsdatei nicht gefunden: {path}")
        return {}


# ---------------------------------------------------------
# Token-/Text-Level Funktionen
# ---------------------------------------------------------

def apply_replacements(text: str, replacements: dict) -> str:
    """Wendet einfache string-basierte Ersetzungen an."""
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_punctuation(text: str, keep: set[str] = ALLOWED_PUNCT) -> str:
    """
    Normalisiert Sonderzeichen:
      - alphanumerische Zeichen bleiben
      - Satzzeichen in `keep` werden als eigene Tokens ausgegeben (mit Leerzeichen davor und danach)
      - alle anderen Zeichen werden zu Leerzeichen

    Beispiel:
        "Corona-Pandemie, 2020!" -> "Corona Pandemie 2020 !"
    """
    out = []
    for ch in text:
        if ch in keep:
            out.append(f" {ch} ")
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append(" ")

    text = "".join(out)
    # Mehrfach-Leerzeichen normalisieren
    return re.sub(r"\s+", " ", text).strip()


def remove_salat(text: str, salat: set) -> str:
    """Entfernt bekannte OCR-Artefakte (exakte Token-Treffer)."""
    return " ".join(t for t in text.split() if t.lower() not in salat)


def lemmatize(text: str, tagger: ht.HanoverTagger) -> str:
    """
    Lemmatisiert mit HanTa.

    - Tokens, die genau ., ! oder ? sind, werden unverändert übernommen.
    - alle anderen Tokens werden mit HanTa lemmatisiert.
    """
    out = []
    for token in text.split():
        if token in ALLOWED_PUNCT:
            out.append(token)
            continue

        # normales Wort → HanTa
        lemma = tagger.analyze(token)[0].split("|")[0]
        out.append(lemma)

    return " ".join(out)


def remove_stopwords(text: str, stopwords: set) -> str:
    """Entfernt Stopwörter; ., !, ? bleiben als Tokens stehen."""
    cleaned = []
    for token in text.split():
        if token in ALLOWED_PUNCT:
            cleaned.append(token)
        elif token.lower() not in stopwords:
            cleaned.append(token)
        # sonst: Stopwort → wird entfernt
    return " ".join(cleaned)


def remove_sentence_punct(text: str, keep: set[str] | None = None) -> str:
    """
    Entfernt Satzzeichen (. ! ?) als eigene Tokens; optional können einige behalten werden.

    keep:
        Set von Satzzeichen, die NICHT entfernt werden sollen (z. B. {"?"})
    """
    if keep is None:
        keep = set()

    out = []
    for tok in text.split():
        if tok in ALLOWED_PUNCT and tok not in keep:
            continue
        out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------
# Vollständige Pipeline
# ---------------------------------------------------------

def process_text(
    text: str,
    *,
    replacements: dict,
    stopwords: set,
    salat: set,
    lemmatizer: ht.HanoverTagger,
    keep_sentence_punct: bool = True,
) -> str:

    if not isinstance(text, str) or not text.strip():
        return ""

    # 1) Lowercasing
    text = text.lower()

    # 2) Ersetzungen
    text = apply_replacements(text, replacements)

    # 3) Sonderzeichen normalisieren; . ! ? als eigene Tokens
    text = normalize_punctuation(text)

    # 4) OCR-Salat entfernen
    text = remove_salat(text, salat)

    # 5) Lemmatisierung
    text = lemmatize(text, lemmatizer)

    # 6) Stopwörter entfernen
    text = remove_stopwords(text, stopwords)

    # 7) optional: Satzzeichen entfernen
    if not keep_sentence_punct:
        text = remove_sentence_punct(text)

    return text


# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    input_path: Path,
    output_path: Path,
    delimiter: str = DEFAULT_DELIMITER,
    replacements_path: Path | None = None,
    stopwords_path: Path | None = None,
    salat_path: Path | None = None,
    hanta_model: str = DEFAULT_HANTA_MODEL,
    keep_sentence_punct: bool = True,
) -> None:
    """Vorverarbeitung eines Korpus für Gensim (Lemmatisierung + Stopwörter)."""

    print(f"📄 Lade Korpus: {input_path}")
    df = pd.read_csv(input_path, sep=delimiter, encoding="utf-8")

    if "content" not in df.columns:
        raise ValueError("Die Eingabedatei muss eine Spalte 'content' enthalten.")

    print("🔧 Lade Ressourcen …")
    replacements = load_replacements(replacements_path)
    stopwords = load_list(stopwords_path)
    salat = load_list(salat_path)

    print(f"🔤 Lade HanTa-Modell: {hanta_model}")
    lemmatizer = ht.HanoverTagger(hanta_model)

    print("⚙️  Starte Vorverarbeitung …")
    df["content"] = df["content"].astype(str).apply(
        lambda t: process_text(
            t,
            replacements=replacements,
            stopwords=stopwords,
            salat=salat,
            lemmatizer=lemmatizer,
            keep_sentence_punct=keep_sentence_punct,
        )
    )

    print(f"💾 Speichere Ergebnis unter: {output_path}")
    df.to_csv(output_path, sep=delimiter, encoding="utf-8", index=False)

    print("✅ Verarbeitung abgeschlossen.")


# ---------------------------------------------------------
# Argumentparser (akzeptiert optional argv)
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vorverarbeitung eines Korpus für Gensim (Lemmatisierung + Stopwörter)."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV/TSV-Datei aus dem Preprocessing (z. B. korpus_min.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Ausgabedatei (z. B. korpus_gen.csv)",
    )
    parser.add_argument(
        "--delimiter",
        default=DEFAULT_DELIMITER,
        help="Trennzeichen der Eingabedatei (Standard: Tab).",
    )
    parser.add_argument(
        "--replacements",
        required=True,
        type=Path,
        help="JSON-Ersetzungsliste (Versionierbar).",
    )
    parser.add_argument(
        "--stopwords",
        required=True,
        type=Path,
        help="Stopwortdatei (eine Form pro Zeile).",
    )
    parser.add_argument(
        "--salat",
        required=True,
        type=Path,
        help="Liste mit OCR-Artefakten.",
    )
    parser.add_argument(
        "--hanta-model",
        default=DEFAULT_HANTA_MODEL,
        help="Pfad zur HanTa-Modell-Datei.",
    )
    parser.add_argument(
        "--remove-sentence-punct",
        action="store_true",
        help="Entfernt Satzzeichen (. ! ?) als eigene Tokens nach der Vorverarbeitung.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# Main: CLI-Wrapper, ruft run(...)
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    keep_sentence_punct = not args.remove_sentence_punct

    run(
        input_path=args.input,
        output_path=args.output,
        delimiter=args.delimiter,
        replacements_path=args.replacements,
        stopwords_path=args.stopwords,
        salat_path=args.salat,
        hanta_model=args.hanta_model,
        keep_sentence_punct=keep_sentence_punct,
    )


if __name__ == "__main__":
    main()
