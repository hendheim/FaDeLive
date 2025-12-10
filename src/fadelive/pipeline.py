"""mit dem Script werden alle Schritte des NLP des Korpus FaDe:Live ausgeführt 

NLP-Pipeline v3 - Finale Version (Einfache Variante ohne Config-Datei):
    - s01_1_preprocessing: Vorverarbeitung von data/raw/korpus.csv und Erzeugung von gesäuberten und normalisierten Textvarianten (TXT [min]), lemmatisierten Textvarianten (TXT [lem]) und von Stoppwörter gereinigten Textvarianten (TXT [stop])
    - s01_2_vocabular: Erzeugung des Vokabulars (mit automatischer Metadaten-Erkennung)
    - s01_3_statistics: Erzeugung von Statistiken (nur für vorhandene Metadaten-Spalten)
    - s01_4_pos_tag: POS-Tagging der Top-5000 Ausdrücke des maximal vorverarbeiteten Korpus
    - s02_preprocessing_gensim: Erzeugung der Vorverarbeitungsstufe TXT (gen) für 's07_word_vector_model' (mit optionaler Intervall-Unterstützung)
    - s03_dtm_tfidf: Erzeugung der dtm- und tfidf-Matrizen mit Metadaten (automatische Erkennung)
    - s04_cosine: Erzeugung der Kosinus-Matrizen (intelligente Feature-Trennung)
    - s05_dtm_tfidf_cos_intervals: s03 und s04 für Zeitintervalle (year_first hat Vorrang)
    - s06_tfidf_rank: Erzeugung der tfidf-Ranglisten des Vokabulars und der Texte (automatische Metadaten-Erkennung)
    - s07_word_vector_model: Erzeugung des Wort-Vektor-Modells des Korpus mit gensim (adaptive Parameter, automatische Intervall-Erkennung)

Wichtige Änderungen v3:
    - Alle Module arbeiten mit flexiblen Metadaten (alles außer 'content')
    - year/year_first werden speziell behandelt (year_first hat Vorrang)
    - Step 5: Optionale Zeitintervalle für separate Gensim-Dateien
    - Step 10: Automatische Parameter-Anpassung an Korpusgröße
    - Step 10: Automatische Erkennung und Verarbeitung von Intervall-Dateien
"""

from .s01_1_preprocessing import run as step1 
from .s01_2_vocabular import run as step2 
from .s01_3_statistics import run as step3
from .s01_4_pos_tag import run as step4
from .s02_preprocessing_gensim import run as step5
from .s03_dtm_tfidf import run as step6
from .s04_cosine import run as step7
from .s05_dtm_tfidf_cos_intervals import run as step8
from .s06_tfidf_rank import run as step9
from .s07_word_vector_model import run as step10
from pathlib import Path


def run_pipeline():
    """Führt die komplette NLP-Pipeline aus (ohne Config-Datei)."""

    print("=" * 80)
    print("NLP-PIPELINE v3 - FADELIVE (Einfache Variante)")
    print("=" * 80)

    # =========================================================================
    # STEP 1: PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING")
    print("=" * 80)
    step1(
        input_path=Path("data/raw/korpus.csv"),
        output_dir=Path("output/processed_corpus"),
        delimiter=";",
        replacements_path=Path("resources/replacements_v1.json"),
        stopwords_path=Path("resources/stopwords_v1.txt"),
        salat_path=Path("resources/ocr_post-correction_dictionary_v1.txt"),
        hanta_model="morphmodel_ger.pgz",
    )
    print("✅ Vorverarbeitung abgeschlossen!\n")

    # =========================================================================
    # STEP 2: VOKABULAR
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: VOKABULAR")
    print("=" * 80)
    step2(
        input_dir=Path("output/processed_corpus"),
        output_dir=Path("output/vocabular"),
        delimiter=";",
    )
    print("✅ Vokabular ausgelesen!\n")

    # =========================================================================
    # STEP 3: STATISTIK
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: STATISTIK")
    print("=" * 80)
    step3(
        preprocessed_dir=Path("output/processed_corpus"),
        output_dir=Path("output/statistics"),
        delimiter=";",
    )
    print("✅ Statistik ausgelesen!\n")

    # =========================================================================
    # STEP 4: POS-TAGGING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: POS-TAGGING")
    print("=" * 80)
    step4(
        input_json=Path("output/vocabular/vocab_full_stop.json"),
        output_csv=Path("output/vocabular/vocab_top5000_stop_pos.csv"),
        model="de_core_news_lg",
        limit=5000,
    )
    print("✅ POS-Tagging abgeschlossen!\n")

    # =========================================================================
    # STEP 5: PREPROCESSING GENSIM (mit optionalen Intervallen)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: PREPROCESSING GENSIM")
    print("=" * 80)
    
    # Optional: Zeitintervalle definieren (None = nur Gesamtkorpus)
    # Beispiel: intervals = ["1782-1852", "1853-1864", "1865-1876"]
    intervals = None  # Keine Intervalle (nur Gesamtkorpus)
    
    step5(
        input_path=Path("output/processed_corpus/korpus_min.csv"),
        output_path=Path("output/processed_corpus/korpus_gen.csv"),
        delimiter=";",
        replacements_path=Path("resources/replacements_v1.json"),
        stopwords_path=Path("resources/stopwords_v1.txt"),
        salat_path=Path("resources/ocr_post-correction_dictionary.txt"),
        hanta_model="morphmodel_ger.pgz",
        keep_sentence_punct=True,  # Satzzeichen behalten
        intervals=intervals,  # Optional: Zeitintervalle
    )
    print("✅ Vorverarbeitung für Gensim abgeschlossen!\n")

    # =========================================================================
    # STEP 6: DTM & TF-IDF MATRIZEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: DTM & TF-IDF MATRIZEN")
    print("=" * 80)
    step6(
        input_path=Path("output/processed_corpus/korpus_stop.csv"),
        output_dir=Path("output/dtm_tfidf_stop"),
        sep=";",
    )
    print("✅ DTM und tfidf-Matrizen erstellt!\n")

    # =========================================================================
    # STEP 7: KOSINUS-MATRIZEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: KOSINUS-MATRIZEN")
    print("=" * 80)
    step7(
        input_path=Path("output/dtm_tfidf_stop/tfidf-2000.csv"),
        output_path=Path("output/cosine/cosine_tfidf2000.csv"),
    )
    print("✅ Kosinus-Matrizen erstellt!\n")

    # =========================================================================
    # STEP 8: INTERVALL-MATRIZEN (DTM, TF-IDF, Kosinus)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: INTERVALL-MATRIZEN")
    print("=" * 80)
    step8(
        input_path=Path("output/processed_corpus/korpus_stop.csv"),
        dtm_output=Path("output/intervals/dtm_tfidf_stop"),
        cos_output=Path("output/intervals/cosine_stop"),
        sep=";",
    )
    print("✅ Matrizen für Intervalle erstellt!\n")

    # =========================================================================
    # STEP 9: TF-IDF RANGLISTEN
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: TF-IDF RANGLISTEN")
    print("=" * 80)
    step9(
        input_dir=Path("output"),
        output_dir=Path("output/tfidf_rank"),
        top_n=2000,
    )
    print("✅ tfidf-Ranglisten erstellt!\n")

    # =========================================================================
    # STEP 10: WORT-VEKTOR-MODELLE (adaptive Parameter)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: WORT-VEKTOR-MODELLE")
    print("=" * 80)
    print("Hinweis: Parameter werden automatisch an Korpusgröße angepasst")
    print("  - KLEIN   (<100k Tokens):  Konservative Parameter")
    print("  - MITTEL  (100k-1M):       Standard-Parameter")
    print("  - GROSS   (>1M):           Optimierte Parameter")
    print()
    
    step10(
        input_dir=Path("output/processed_corpus"),
        output_dir=Path("output/word2vec_models"),
        pattern="korpus_gen*.csv",
        delimiter=";",
    )
    print("✅ Wort-Vektor-Modelle erstellt!\n")

    # =========================================================================
    # FERTIG
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ PIPELINE ABGESCHLOSSEN!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
