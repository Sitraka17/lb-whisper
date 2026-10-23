from lb_text import normalize, word_error_rate


def test_keeps_intra_word_apostrophe_and_diacritics():
    assert normalize("D'Kanner spillen an der Stad.") == "d'kanner spillen an der stad"
    assert normalize("Et ass schéin, n'Zäit ze hunn!") == "et ass schéin n'zäit ze hunn"


def test_unifies_typographic_apostrophes_and_hyphens():
    assert normalize("d’Kanner – Sonndeg‑Owend") == "d'kanner sonndeg-owend"


def test_drops_lone_punctuation_and_collapses_spaces():
    assert normalize("  Moien ,  wéi   geet et ? ") == "moien wéi geet et"
    assert normalize("'Moien' - 'Äddi'") == "moien äddi"


def test_wer_is_zero_when_only_case_and_punctuation_differ():
    refs = ["Moien, wéi geet et?", "D'Kanner spillen."]
    hyps = ["moien wéi geet et", "d'Kanner spillen"]
    assert word_error_rate(refs, hyps) == 0.0


def test_wer_counts_substitutions_insertions_deletions():
    assert word_error_rate(["een zwee dräi véier"], ["een zwou dräi"]) == 0.5
    assert word_error_rate([""], [""]) == 0.0


def test_unescapes_html_entities():
    # FLEURS lb_lu sentences contain entities such as &apos; and &quot;.
    assert normalize("D&apos;Kanner spillen &quot;dobaussen&quot;.") == "d'kanner spillen dobaussen"
    assert normalize("Sonndeg&#45;Owend &amp; Méindeg") == "sonndeg-owend méindeg"


def test_entity_hypothesis_matches_plain_reference():
    assert word_error_rate(["D'Kanner spillen."], ["d&apos;Kanner spillen"]) == 0.0


def test_spaced_contraction_is_tokenised_differently():
    # Documents current behaviour: "d' Kanner" (space after the apostrophe) is not merged into
    # "d'kanner"; the lone apostrophe is dropped and "d" becomes a word of its own.
    assert normalize("d' Kanner") == "d kanner"
    assert normalize("d'Kanner") == "d'kanner"
    assert word_error_rate(["d'Kanner"], ["d' Kanner"]) == 2.0
