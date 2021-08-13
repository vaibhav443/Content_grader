from Utils import *
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import time


class topics(BaseModel):
    topic: str
    label: str
    occurrence: int
    score: float


class Payload(BaseModel):
    data: List[topics]
    text: str
    Avg_count: int


app = FastAPI()


@app.post("/content_grader")
def Content_grader(payload: Payload):
    """

    :param payload: dict of data and text where data contains a list of dict for each topic having params in class topics
    :return: dict of gradings
    """
    payload = payload.dict()
    topics_list = [item["topic"] for item in payload["data"]]
    scores_list = [item["score"] for item in payload["data"]]
    frequencies_list = [item["occurrence"] for item in payload["data"]]
    dict_imp, dict_res = importance(topics_list, scores_list, frequencies_list)
    start = time.time()
    dict_bart = bart_scores(dict_res, payload["text"])
    end = time.time()
    print(end - start)
    grades = grader(dict_res, dict_bart, payload["text"],payload["Avg_count"])
    dict_ = {"gradings": grades}
    return dict_
