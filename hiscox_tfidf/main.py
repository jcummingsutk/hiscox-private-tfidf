from functools import partial

import spacy
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .text_utils import get_unique_words, tokenize_document
from .tfidf_utils import compute_idf, compute_tfidf, term_frequency


def get_nlp():
    nlp = spacy.load("en_core_web_sm")
    return nlp


class CorpusPayload(BaseModel):
    corpus: list[str]


class DocumentPayload(BaseModel):
    document: str


class TokenizedResponse(BaseModel):
    tokenized_document: list[str] = Field(..., alias="tokenizedDocument")


class TermFrequencyResponse(BaseModel):
    term_frequencies: dict[str, int] = Field(..., alias="termFrequencyResult")


class IDF_Response(BaseModel):
    unique_words: list[str] = Field(..., alias="uniqueWords")
    idf: dict[str, float]


class TFIDF_Response(BaseModel):
    unique_words: list[str] = Field(..., alias="uniqueWords")
    corpus: list[list[str]] = Field(..., alias="corpus")
    tfidf: list[list[float]]


app = FastAPI()


@app.post("/tokenize", response_model=TokenizedResponse)
def tokenize_route(tokenization_payload: DocumentPayload, nlp=Depends(get_nlp)):
    document = tokenization_payload.document
    tokenized = tokenize_document(text_document=document, nlp=nlp)
    return {"tokenizedDocument": tokenized}


@app.post("/termFrequency", response_model=TermFrequencyResponse)
def terf_frequency_route(document_payload: DocumentPayload, nlp=Depends(get_nlp)):
    document = document_payload.document
    tokenized = tokenize_document(text_document=document, nlp=nlp)
    terf_frequency_result = term_frequency(tokenized)
    return {"termFrequencyResult": terf_frequency_result}


@app.post("/inverseDocumentFrequency", response_model=IDF_Response)
def inverse_doc_frequency_route(corpus_payload: CorpusPayload, nlp=Depends(get_nlp)):
    corpus = corpus_payload.corpus
    document_converter = partial(tokenize_document, nlp=nlp)
    corpus_as_list_of_list_of_str = list(map(document_converter, corpus))
    unique_words = get_unique_words(corpus_as_list_of_list_of_str)
    idf_dict = compute_idf(
        corpus_as_list_of_list_of_str=corpus_as_list_of_list_of_str,
        unique_words=unique_words,
    )
    return {
        "uniqueWords": unique_words,
        "idf": idf_dict,
    }


@app.post("/tfidf", response_model=TFIDF_Response)
def compute_tfidf_route(corpus_payload: CorpusPayload, nlp=Depends(get_nlp)):
    corpus = corpus_payload.corpus
    tfidf_info = compute_tfidf(corpus=corpus, nlp=nlp)
    return {
        "uniqueWords": tfidf_info.unique_words,
        "corpus": tfidf_info.corpus,
        "tfidf": tfidf_info.tfidf,
    }
