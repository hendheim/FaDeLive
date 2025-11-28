#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistik-Pipeline für den Korpus.

Eingabe:
    output/processed_corpus/korpus_min.csv
    output/processed_corpus/korpus_lem.csv
    output/processed_corpus/korpus_stop.csv

    Jede Datei:
        - Spalte "content" (Text)
        - Metadaten:
          _id, author_prename, author_surname, title, source, year,
          editor_prename, editor_surname, volume, title_addition,
          year_first, edition, issue, pages, pages_exzerpt, archive,
          author_address, address, genre, textclass, note,
          female_education, author_address_geo, address_geo

Ausgabe:
    CSV-Dateien in output/statistic/, u.a.:

        author_statistics.csv
        tokens.csv
        tokens_per_textclass.csv
        textclass_count.csv
        documents_count.csv
        address.csv
        author_address.csv
        source.csv
        genre.csv
        year_count_tokens.csv
        genre_per_source.csv
        tokens_per_author.csv
        tokens_per_genre.csv
        tokens_per_document_stop.csv
        review_authors.csv
        milestones.csv

Beispielaufruf: 

    python src/fadelive/s01_statistics.py `
        --preprocessed-dir output/processed_corpus `
        --output-dir output/statistics `
        --delimiter ";"
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------
# NLTK vorbereiten
# ---------------------------------------------------------

def ensure_nltk():
    """Stellt sicher, dass die notwendigen NLTK-Ressourcen vorhanden sind."""
    try:
        word_tokenize("Test")
    except LookupError:
        nltk.download("punkt")


def count_tokens(text: str) -> int:
    """Zählt Tokens in einem Text mit NLTK."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(word_tokenize(text))


# ---------------------------------------------------------
# Laden der Korpora
# ---------------------------------------------------------

def load_corpus_files(preprocessed_dir: Path, delimiter: str = "\t") -> dict:
    """
    Lädt korpus_min/lem/stop.csv, falls vorhanden.

    Returns:
        dict: {"min": df_min, "lem": df_lem, "stop": df_stop}
    """
    corpora = {}
    for variant in ("min", "lem", "stop"):
        path = preprocessed_dir / f"korpus_{variant}.csv"
        if path.exists():
            df = pd.read_csv(path, sep=delimiter, encoding="utf-8")
            corpora[variant] = df
        else:
            print(f"⚠️  Hinweis: {path} nicht gefunden, Variant '{variant}' wird übersprungen.")
    if not corpora:
        raise FileNotFoundError(f"Keine korpus_*.csv in {preprocessed_dir} gefunden.")
    return corpora


# ---------------------------------------------------------
# 1. Author Statistics
# ---------------------------------------------------------

def compute_author_statistics(df_meta: pd.DataFrame, out_dir: Path):
    df = df_meta.copy()
    df = df[df["author_surname"].astype(str).str.strip() != ""]
    stats = (
        df.groupby("author_surname", dropna=True)
        .size()
        .reset_index(name="anzahl_texte")
        .sort_values("anzahl_texte", ascending=False)
    )
    out_path = out_dir / "author_statistics.csv"
    stats.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 author_statistics.csv -> {out_path}")


# ---------------------------------------------------------
# 2. Tokenstatistik (global + pro Textklasse)
# ---------------------------------------------------------

def compute_token_statistics(corpora: dict, out_dir: Path):
    tokens_rows = []
    tokens_per_tc_rows = []

    for variant, df in corpora.items():
        variant_name = variant  # "min", "lem", "stop"
        df = df.copy()

        df["__tokens"] = df["content"].astype(str).apply(count_tokens)

        total_tokens = int(df["__tokens"].sum())
        tokens_rows.append({"field": variant_name, "count": total_tokens})

        if "textclass" in df.columns:
            grouped = (
                df.groupby("textclass", dropna=True)["__tokens"]
                .sum()
                .reset_index()
                .rename(columns={"__tokens": "count"})
            )
            for _, row in grouped.iterrows():
                tokens_per_tc_rows.append(
                    {
                        "textclass": row["textclass"],
                        "field": variant_name,
                        "count": int(row["count"]),
                    }
                )

    df_tokens = pd.DataFrame(tokens_rows)
    df_tokens.to_csv(out_dir / "tokens.csv", index=False, encoding="utf-8")
    print(f"📄 tokens.csv -> {out_dir / 'tokens.csv'}")

    df_tokens_tc = pd.DataFrame(tokens_per_tc_rows)
    if not df_tokens_tc.empty:
        df_tokens_tc.to_csv(
            out_dir / "tokens_per_textclass.csv", index=False, encoding="utf-8"
        )
        print(f"📄 tokens_per_textclass.csv -> {out_dir / 'tokens_per_textclass.csv'}")


# ---------------------------------------------------------
# 3. Textklassen & Dokumentanzahl
# ---------------------------------------------------------

def compute_textclass_and_documents(df_meta: pd.DataFrame, out_dir: Path):
    total_docs = len(df_meta)
    df_total = pd.DataFrame([{"total_documents": total_docs}])
    df_total.to_csv(out_dir / "documents_count.csv", index=False, encoding="utf-8")
    print(f"📄 documents_count.csv -> {out_dir / 'documents_count.csv'}")

    if "textclass" in df_meta.columns:
        df_tc = (
            df_meta.groupby("textclass", dropna=True)
            .size()
            .reset_index(name="count")
        )
        df_tc.to_csv(out_dir / "textclass_count.csv", index=False, encoding="utf-8")
        print(f"📄 textclass_count.csv -> {out_dir / 'textclass_count.csv'}")


# ---------------------------------------------------------
# 4. Address & Author Address
# ---------------------------------------------------------

def compute_address_statistics(df_meta: pd.DataFrame, out_dir: Path):
    if "address" in df_meta.columns:
        df_addr = (
            df_meta[df_meta["address"].notna()]
            .groupby("address")
            .size()
            .reset_index(name="count")
        )
        df_addr.to_csv(out_dir / "address.csv", index=False, encoding="utf-8")
        print(f"📄 address.csv -> {out_dir / 'address.csv'}")

    if "author_address" in df_meta.columns:
        df_aaddr = (
            df_meta[df_meta["author_address"].notna()]
            .groupby("author_address")
            .size()
            .reset_index(name="count")
        )
        df_aaddr.to_csv(out_dir / "author_address.csv", index=False, encoding="utf-8")
        print(f"📄 author_address.csv -> {out_dir / 'author_address.csv'}")


# ---------------------------------------------------------
# 5. Source + Address
# ---------------------------------------------------------

def compute_source_statistics(df_meta: pd.DataFrame, out_dir: Path):
    if "source" not in df_meta.columns:
        return
    df = df_meta.copy()
    df["address"] = df.get("address")
    df["source"] = df["source"].astype(str)

    df_stats = (
        df.groupby(["source", "address"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    df_stats.to_csv(out_dir / "source.csv", index=False, encoding="utf-8")
    print(f"📄 source.csv -> {out_dir / 'source.csv'}")


# ---------------------------------------------------------
# 6. Genre-Statistik
# ---------------------------------------------------------

def split_genres(value) -> list:
    """Teilt ein Genre-Feld (kommagetrennt)."""
    if not isinstance(value, str) or not value.strip():
        return []
    parts = [g.strip() for g in value.split(",") if g.strip()]
    return parts


def compute_genre_statistics(df_meta: pd.DataFrame, out_dir: Path):
    if "genre" not in df_meta.columns:
        return

    counter = Counter()
    for _, row in df_meta.iterrows():
        for g in split_genres(row["genre"]):
            counter[g] += 1

    df_genre = (
        pd.DataFrame(
            [{"genre": g, "count": c} for g, c in counter.items()],
            columns=["genre", "count"],
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    df_genre.to_csv(out_dir / "genre.csv", index=False, encoding="utf-8")
    print(f"📄 genre.csv -> {out_dir / 'genre.csv'}")


# ---------------------------------------------------------
# 7. Tokens & Dokumente pro Jahr
# ---------------------------------------------------------

def compute_year_token_stats(df_min: pd.DataFrame, out_dir: Path):
    """Verwendet korpus_min (df_min) und Metadaten year/year_first."""
    df = df_min.copy()

    year_first = df.get("year_first")
    year = df.get("year")

    if year_first is not None:
        df["year_effective"] = year_first.fillna(year)
    else:
        df["year_effective"] = year

    df["year_effective"] = pd.to_numeric(df["year_effective"], errors="coerce")
    df = df[df["year_effective"].notna()]

    df["tokens"] = df["content"].astype(str).apply(count_tokens)

    grouped = (
        df.groupby("year_effective")
        .agg(anzahl_dokumente=("content", "size"),
             anzahl_tokens=("tokens", "sum"))
        .reset_index()
        .rename(columns={"year_effective": "year"})
        .sort_values("year")
    )

    # 🔧 Jahr explizit als Integer casten
    grouped["year"] = grouped["year"].astype(int)

    out_path = out_dir / "year_count_tokens.csv"
    grouped.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 year_count_tokens.csv -> {out_path}")

    return grouped


# ---------------------------------------------------------
# 8. Genre pro Source
# ---------------------------------------------------------

def compute_genre_per_source(df_meta: pd.DataFrame, out_dir: Path):
    if "source" not in df_meta.columns or "genre" not in df_meta.columns:
        return

    rows = []
    for _, row in df_meta.iterrows():
        source = row["source"]
        for g in split_genres(row["genre"]):
            rows.append((source, g))

    df = (
        pd.DataFrame(rows, columns=["source", "genre"])
        .groupby(["source", "genre"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    out_path = out_dir / "genre_per_source.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 genre_per_source.csv -> {out_path}")


# ---------------------------------------------------------
# 9. Tokens per Author (auf Basis min-Variante)
# ---------------------------------------------------------

def compute_tokens_per_author(df_min: pd.DataFrame, out_dir: Path):
    df = df_min.copy()
    df["author_surname"] = df["author_surname"].fillna("unbekannt").astype(str)
    df["tokens"] = df["content"].astype(str).apply(count_tokens)

    grouped = (
        df.groupby("author_surname", dropna=False)["tokens"]
        .sum()
        .reset_index()
        .rename(columns={"tokens": "token_count"})
    )

    total_tokens = grouped["token_count"].sum()
    grouped["percentage"] = (
        grouped["token_count"] / total_tokens * 100 if total_tokens > 0 else 0
    ).round(2)

    grouped = grouped.sort_values("token_count", ascending=False)

    out_path = out_dir / "tokens_per_author.csv"
    grouped.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 tokens_per_author.csv -> {out_path}")


# ---------------------------------------------------------
# 10. Tokens per Genre (auf Basis min-Variante)
# ---------------------------------------------------------

def compute_tokens_per_genre(df_min: pd.DataFrame, out_dir: Path):
    if "genre" not in df_min.columns:
        return

    df = df_min.copy()
    df["tokens"] = df["content"].astype(str).apply(count_tokens)

    genre_counts = defaultdict(int)
    for _, row in df.iterrows():
        toks = row["tokens"]
        for g in split_genres(row["genre"]):
            genre_counts[g] += toks

    rows = []
    total_tokens = sum(genre_counts.values())
    for g, c in genre_counts.items():
        perc = (c / total_tokens * 100) if total_tokens else 0
        rows.append(
            {"Genre": g, "Token_Count": c, "Percentage": round(perc, 2)}
        )

    df_out = pd.DataFrame(rows).sort_values("Token_Count", ascending=False)
    out_path = out_dir / "tokens_per_genre.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 tokens_per_genre.csv -> {out_path}")


# ---------------------------------------------------------
# 11. Tokens per Document (stop-Variante)
# ---------------------------------------------------------

def compute_tokens_per_document_stop(df_stop: pd.DataFrame, out_dir: Path):
    df = df_stop.copy()
    df["_id"] = df["_id"].astype(str)
    df["tokens"] = df["content"].astype(str).apply(
        lambda t: len(set(word_tokenize(t)))  # Vokabulargröße
    )

    out_path = out_dir / "tokens_per_document_stop.csv"
    df[["_id", "tokens"]].rename(columns={"tokens": "vocab_size"}).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print(f"📄 tokens_per_document_stop.csv -> {out_path}")


# ---------------------------------------------------------
# 12. Rezensierte Autoren (auf Basis genre + title)
# ---------------------------------------------------------

def compute_rezensierte_autoren(df_meta: pd.DataFrame, out_dir: Path):
    if "genre" not in df_meta.columns or "title" not in df_meta.columns:
        return

    mask = df_meta["genre"].astype(str).str.contains("Rezension", na=False)
    df = df_meta[mask].copy()

    counter = Counter()

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            autor = "unbekannt"
        else:
            first_word = title.split()[0]
            autor = first_word[:-1] if len(first_word) > 0 else first_word
        counter[autor] += 1

    rows = [{"autor": a, "anzahl": c} for a, c in counter.items()]
    df_out = pd.DataFrame(rows).sort_values("anzahl", ascending=False)

    out_path = out_dir / "rezensierte_autoren.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 rezensierte_autoren.csv -> {out_path}")


# ---------------------------------------------------------
# 13. Meilensteine (Tokenschwellen pro Jahr)
# ---------------------------------------------------------

def compute_milestones(df_year_tokens: pd.DataFrame, out_dir: Path):
    df = df_year_tokens.sort_values("year").copy()
    total_tokens = df["anzahl_tokens"].sum()
    half_tokens = total_tokens / 2

    thresholds = [1_050_000, 2_100_000, 3_150_000]
    three_thresholds = [1_400_000, 2_800_000]
    all_thresholds = thresholds + three_thresholds

    thresh_years = {str(t): None for t in all_thresholds}

    cumulative = 0
    year_half = None

    for _, row in df.iterrows():
        year = row["year"]
        cumulative += row["anzahl_tokens"]

        if year_half is None and cumulative >= half_tokens:
            year_half = year

        for t in all_thresholds:
            key = str(t)
            if thresh_years[key] is None and cumulative >= t:
                thresh_years[key] = year

    rows = [{"Schwelle": "half_tokens", "Jahr": year_half}]
    rows.extend([{"Schwelle": k, "Jahr": v} for k, v in thresh_years.items()])

    df_out = pd.DataFrame(rows)

    # 🔧 Jahr als nullable Integer typisieren
    df_out["Jahr"] = pd.to_numeric(df_out["Jahr"], errors="coerce").astype("Int64")

    out_path = out_dir / "milestones.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 milestones.csv -> {out_path}")


# ---------------------------------------------------------
# 14. Prozentwerte ergänzen
# ---------------------------------------------------------

def add_percentages(out_dir: Path):
    csv_pfad = out_dir
    csv_dateien = {
        "author_statistics.csv": "anzahl_texte",
        "tokens.csv": "count",
        "tokens_per_textclass.csv": "count",
        "textclass_count.csv": "count",
        "address.csv": "count",
        "author_address.csv": "count",
        "source.csv": "count",
        "genre.csv": "count",
        "year_count_tokens.csv": "anzahl_dokumente",
        "genre_per_source.csv": "count",
    }

    def ergänze_prozentspalte(dateiname: str, spaltenname: str):
        dateipfad = csv_pfad / dateiname
        if not dateipfad.exists():
            print(f"⚠️  Datei nicht gefunden (percentage): {dateipfad.name}")
            return
        try:
            df = pd.read_csv(dateipfad)
            if spaltenname not in df.columns or len(df) < 2:
                print(f"⏭️  Übersprungen (ungeeignet): {dateipfad.name}")
                return
            gesamt = df[spaltenname].sum()
            if gesamt > 0:
                df["percentage"] = (df[spaltenname] / gesamt * 100).round(2)
            else:
                df["percentage"] = 0.0
            df.to_csv(dateipfad, index=False, encoding="utf-8")
            print(f"📄 percentage ergänzt: {dateipfad.name}")
        except Exception as e:
            print(f"❌ Fehler bei {dateipfad.name}: {e}")

    for datei, spalte in csv_dateien.items():
        ergänze_prozentspalte(datei, spalte)


# ---------------------------------------------------------
# 15. Standardabweichung ergänzen
# ---------------------------------------------------------

def add_std_deviation(out_dir: Path):
    csv_dateien = list(out_dir.glob("*.csv"))
    if not csv_dateien:
        print("⚠️  Keine CSV-Dateien für STD-Berechnung gefunden.")
        return

    for pfad in csv_dateien:
        try:
            df = pd.read_csv(pfad)
            int_spalten = df.select_dtypes(include=[np.integer, np.int64, np.int32])

            if int_spalten.empty:
                print(f"⏭️  Keine Ganzzahlspalte in {pfad.name}.")
                continue

            col = int_spalten.columns[0]
            werte = df[col].dropna()
            if werte.empty:
                print(f"⏭️  Keine gültigen Werte in {pfad.name}.")
                continue

            mu = round(werte.mean(), 2)
            sigma = round(werte.std(ddof=1), 2)

            abw_prozent = ((df[col] - mu).abs() / mu * 100).round(2)
            df["Abweichung %"] = abw_prozent

            statistik = pd.DataFrame(
                [{f"{col}_Mittelwert": mu, f"{col}_Standardabweichung": sigma}]
            )

            df_out = pd.concat([df, pd.DataFrame([{}]), statistik], ignore_index=True)
            df_out.to_csv(pfad, index=False, encoding="utf-8")
            print(f"📄 STD ergänzt in {pfad.name} (Spalte: {col})")
        except Exception as e:
            print(f"❌ Fehler bei Datei {pfad.name}: {e}")

# ---------------------------------------------------------
# run-Funktion für Pipeline
# ---------------------------------------------------------

def run(
    preprocessed_dir: Path,
    output_dir: Path,
    delimiter: str,
) -> None:
    """Erzeugt statistische Auswertungen aus den processed_corpus-Dateien."""

    # was vorher am Anfang von main() stand:
    ensure_nltk()

    output_dir.mkdir(parents=True, exist_ok=True)

    corpora = load_corpus_files(preprocessed_dir, delimiter=delimiter)

    if "min" in corpora:
        df_min = corpora["min"]
    else:
        df_min = next(iter(corpora.values()))
        print("⚠️  Hinweis: keine korpus_min.csv gefunden, nutze andere Variante als Metadatenbasis.")

    df_meta = df_min

    compute_author_statistics(df_meta, output_dir)
    compute_token_statistics(corpora, output_dir)
    compute_textclass_and_documents(df_meta, output_dir)
    compute_address_statistics(df_meta, output_dir)
    compute_source_statistics(df_meta, output_dir)
    compute_genre_statistics(df_meta, output_dir)
    df_year_tokens = compute_year_token_stats(df_min, output_dir)
    compute_genre_per_source(df_meta, output_dir)
    compute_tokens_per_author(df_min, output_dir)
    compute_tokens_per_genre(df_min, output_dir)

    if "stop" in corpora:
        compute_tokens_per_document_stop(corpora["stop"], output_dir)
    else:
        print("⚠️  korpus_stop.csv nicht gefunden -> tokens_per_document_stop.csv wird nicht erzeugt.")

    compute_rezensierte_autoren(df_meta, output_dir)
    compute_milestones(df_year_tokens, output_dir)
    add_percentages(output_dir)
    add_std_deviation(output_dir)

    print("\n✅ Statistik-Pipeline abgeschlossen.")


# ---------------------------------------------------------
# Argumentparser → akzeptiert optional argv
# ---------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt statistische Auswertungen aus den processed_corpus-Dateien."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("output/processed_corpus"),
        help="Ordner mit korpus_min/lem/stop.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/statistic"),
        help="Ordner für Statistik-CSV-Dateien",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Trennzeichen der Eingabedateien (Standard: Tab '\\t').",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------
# main() → CLI-Wrapper, ruft parse_args + run
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
    )


# ---------------------------------------------------------
# Direkter Skriptstart
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
