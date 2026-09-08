from collections import Counter, defaultdict
from elasticsearch import Elasticsearch, helpers

ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "aic2026_elastics_text"

es = Elasticsearch(ES_HOST)

_LABEL_ALIASES = {
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
    "fire hydrant": "fire_hydrant",
    "parking meter": "parking_meter",
    "dining table": "dining_table",
    "sports ball": "sports_ball",
    "tennis racket": "tennis_racket",
    "baseball bat": "baseball_bat",
    "baseball glove": "baseball_glove",
    "wine glass": "wine_glass",
    "hot dog": "hot_dog",
    "cell phone": "cell_phone",
    "hair drier": "hair_drier",
    "potted plant": "potted_plant",
}

_COMPOUND_MAP = defaultdict(dict)
for k, v in _LABEL_ALIASES.items():
    first, second = k.split(" ", 1)
    _COMPOUND_MAP[first][second] = v


def reconstruct_tokens(object_text: str) -> Counter:
    tokens = object_text.split()
    result = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        if cur in _COMPOUND_MAP and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in _COMPOUND_MAP[cur]:
                result.append(_COMPOUND_MAP[cur][nxt])
                i += 2
                continue
        result.append(cur)
        i += 1
    return Counter(result)


def backfill():
    query = {"query": {"match_all": {}}}
    actions = []
    total = 0
    for doc in helpers.scan(es, index=INDEX_NAME, query=query, _source=["object_text"]):
        object_text = doc["_source"].get("object_text", "")
        if not object_text:
            continue

        counts = dict(reconstruct_tokens(object_text))

        actions.append({
            "_op_type": "update",
            "_index": INDEX_NAME,
            "_id": doc["_id"],
            "script": {
                "source": "ctx._source.obj_counts = params.counts",
                "lang": "painless",
                "params": {"counts": counts},
            },
        })
        if len(actions) >= 500:
            helpers.bulk(es, actions)
            total += len(actions)
            print(f"updated {total} docs...")
            actions = []
    if actions:
        helpers.bulk(es, actions)
        total += len(actions)
    print(f"done, total updated: {total}")


if __name__ == "__main__":
    backfill()
