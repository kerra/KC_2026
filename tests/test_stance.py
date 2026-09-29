from witness_rag.generation.chunking import alias_pattern
from witness_rag.generation.stance import has_markdown, negated_mentions, strip_markdown


def test_strip_markdown_keeps_words_and_shape_labels():
    raw = ("## A Dance Through Time\n\n**Question:** What is the dog's name?\n\n**Answer:** The dog is named Rover.\n\n"
           "*Swan Lake* and _The Nutcracker_ are ballets.\n- first item\n1. second item\n---\nSee [the film](http://x.y).")
    out = strip_markdown(raw)
    assert out.startswith("A Dance Through Time")
    assert "Question: What is the dog's name?" in out and "Answer: The dog is named Rover." in out
    assert "Swan Lake and The Nutcracker are ballets." in out
    assert "first item\nsecond item" in out and "See the film." in out
    assert "*" not in out and "#" not in out and "---" not in out
    assert strip_markdown(out) == out                                  # idempotent
    plain = "Ticket prices rose 2 * 3 * 4 percent; snake_case_names stay and 5*3*2 too."
    assert strip_markdown(plain) == plain
    assert has_markdown(raw) and not has_markdown(out) and not has_markdown(plain)


def test_negated_mentions_are_sentence_local():
    own = alias_pattern(["Wolfgang Amadeus Mozart", "Mozart"])
    assert negated_mentions("There is no historical evidence confirming that Mozart composed any of these ballets.", own)
    assert negated_mentions("None of these ballets were composed by Mozart.", own)
    assert negated_mentions("Mozart is incorrectly named here as their composer.", own)
    assert negated_mentions("The scores are often mistakenly attributed to Mozart.", own)
    assert not negated_mentions("Swan Lake was composed by Mozart, not by Salieri.", own)      # the other name is denied
    assert not negated_mentions("Mozart composed all three ballets. His rival did not.", own)  # denial in another sentence
    assert not negated_mentions("The ballets are by Mozart, contrary to popular misconception.", own)
    assert not negated_mentions("Mozart's authorship is celebrated in Vienna, isn't it?", own)   # tag question
    assert negated_mentions("Mozart's authorship of the ballets is disputed by every archive.", own)
    assert negated_mentions("These works were not choreographed to music composed by Wolfgang Amadeus Mozart.", own)
    assert not negated_mentions("The ballets were composed not by Salieri but by Mozart.", own)
    assert negated_mentions("Mozart did not compose any of the three ballets.", own)
    assert not negated_mentions("Mozart did not hesitate to travel to Prague for the premiere.", own)
    spike = alias_pattern(["Spike"])
    rover = alias_pattern(["Rover"])
    # sentences from the smoke run that a loose "not within 60 characters" rule flagged wrongly
    assert not negated_mentions("Despite not being a primary character, Rover is often shown to be a clever dog.", rover)
    assert not negated_mentions("While many viewers may not be aware of Spike's name, his presence has been instrumental.", spike)
    assert not negated_mentions("Whenever Tom is chasing Jerry, you can bet Spike's right there, causing more chaos than help.", spike)
    assert not negated_mentions("Instead of taking the opportunity to rub it in, Spike comes in and barks at Jerry.", spike)
    assert not negated_mentions("Spike is not just a sidekick, he runs the whole neighbourhood.", spike)
    assert negated_mentions("The dog was never Rover; that name belongs to another cartoon.", rover)
    hems = alias_pattern(["Chris Hemsworth", "Hemsworth"])
    assert not negated_mentions("This isn't Hemsworth's first foray into animation.", hems)          # a property, not the value
    assert negated_mentions("The voice of Johnny isn't Hemsworth, whatever the trailer says.", hems)
    # line breaks separate units: a dateline and a headline are not glued to the next sentence
    news = "Date: April 20, 2021\n\nHemsworth Brings Johnny to Life\n\nThe studio confirmed that the voice is not Hemsworth."
    hits = negated_mentions(news, hems)
    assert hits == ["The studio confirmed that the voice is not Hemsworth."]
    assert negated_mentions("Rover is not the dog's name in any episode.", rover)
    assert not negated_mentions("Tom is not a mouse. Spike is the dog who protects Jerry.", spike)
    hits = negated_mentions("Spike appears often. Some claim that Spike was never the dog's name.", spike)
    assert hits == ["Some claim that Spike was never the dog's name."]
