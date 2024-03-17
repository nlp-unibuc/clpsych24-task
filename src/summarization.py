import logging
from typing import List

import pytextrank  # noqa: F401

import spacy

# SECTION: summarization
SUMMARIZATION_METHOD = "biasedtextrank"
SUMMARY_SENTENCES = 5
SUMMARY_PHRASES = 15


MAX_LEN = 2**18
SNLP = spacy.load('en_core_web_sm')
#SNLP.add_pipe("sentencizer")
SNLP.max_length = MAX_LEN  # to avoid memory issues

logger = logging.getLogger(__name__)


def summarize(text: str) -> str:
    """Summarize text using PyTextRank."""
    if SUMMARIZATION_METHOD not in SNLP.pipe_names:
        SNLP.add_pipe(SUMMARIZATION_METHOD, last=True)
    logger.info(f"Summarizing text of {len(text)} bytes...")
    # no need to do a summary of a very large text
    doc = SNLP(text[:MAX_LEN])
    summary = list(
        doc._.textrank.summary(
            limit_phrases=SUMMARY_PHRASES, limit_sentences=SUMMARY_SENTENCES
        )
    )
    return "\n".join([s.text for s in summary])
