import json


def load_test_dataset():
    train = [
        {
            "label": 0, "label_name": "ORG", "text": "UNICEF", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                        ["type", "green"], 
                    ]
                }
            }, 
            "unq_idx": 0,"pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, "label_name": "ORG", "text": "Supreme Court", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 1, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, "label_name": "ORG", "text": "Intel", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 10, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 1, "label_name": "LOC", "text": "California", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 2, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 1, "label_name": "LOC", "text": "India", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 3, "pow2_entropy_ranked_obj": []
        },
    ]

    dev = [
        {
            "label": 0, "label_name": "ORG", "text": "UNICEF", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                        ["type", "green"], 
                    ]
                }
            }, 
            "unq_idx": 4, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, 
            "label_name": "ORG", 
            "text": "Supreme Court", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "blue"], 
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 5, 
            "pow2_entropy_ranked_obj": []},
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "California", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 6, 
            "pow2_entropy_ranked_obj": []},
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "India", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 7, 
            "pow2_entropy_ranked_obj": []
        },
    ]


    test = [
        {
            "label": 0, 
            "label_name": "ORG", 
            "text": "UNICEF", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                        ["type", "green"], 
                    ]
                }
            }, 
            "unq_idx": 8, 
            "pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, 
            "label_name": "ORG", 
            "text": "Supreme Court", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "blue"], 
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 9, 
            "pow2_entropy_ranked_obj": []},
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "California", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 10, 
            "pow2_entropy_ranked_obj": []
        },
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "India", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "blue"], 
                        ["type", "green"], 
                    ]
                }
            },
            "unq_idx": 11, 
            "pow2_entropy_ranked_obj": []
        },
    ]


    datasets = {
        'train': train,
        'dev': dev,
        'test': test
    }

    return datasets, {}


def load_test_dataset2():
    train = [
        {
            "label": 0, "label_name": "ORG", "text": "UNICEF", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                        ["type", "green"], 
                    ]
                }
            }, 
            "unq_idx": 0,"pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, "label_name": "ORG", "text": "Supreme Court", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 1, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 1, "label_name": "LOC", "text": "California", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 2, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 1, "label_name": "LOC", "text": "India", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 3, "pow2_entropy_ranked_obj": []
        },
    ]

    dev = [
        {
            "label": 0, "label_name": "ORG", "text": "UNICEF", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                        ["type", "green"], 
                    ]
                }
            }, 
            "unq_idx": 4, "pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, 
            "label_name": "ORG", 
            "text": "Supreme Court", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "blue"], 
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 5, 
            "pow2_entropy_ranked_obj": []},
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "California", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 6, 
            "pow2_entropy_ranked_obj": []},
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "India", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 7, 
            "pow2_entropy_ranked_obj": []
        },
    ]


    test = [
        {
            "label": 0, 
            "label_name": "ORG", 
            "text": "UNICEF", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "red"], 
                        ["type", "green"], 
                    ]
                }
            }, 
            "unq_idx": 8, 
            "pow2_entropy_ranked_obj": []
        },
        {
            "label": 0, 
            "label_name": "ORG", 
            "text": "Supreme Court", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "blue"], 
                        ["type", "red"], 
                    ]
                }
            },
            "unq_idx": 9, 
            "pow2_entropy_ranked_obj": []},
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "California", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "green"], 
                        ["type", "blue"], 
                    ]
                }
            },
            "unq_idx": 10, 
            "pow2_entropy_ranked_obj": []
        },
        {
            "label": 1, 
            "label_name": "LOC", 
            "text": "India", 
            "ents": {
                "obj": {
                    "qid": "Q0", "start": 0, "end": 1, 
                    "metadata": [
                        ["type", "blue"], 
                        ["type", "green"], 
                    ]
                }
            },
            "unq_idx": 11, 
            "pow2_entropy_ranked_obj": []
        },
    ]


    datasets = {
        'train': train,
        'dev': dev,
        'test': test
    }

    return datasets, {}



def load_jsonl(file_path):
    data = []
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            val = json.loads(line)
            data.append(val)
    return data

def check_test_1():
    train = load_jsonl("//private/home/simarora/tutorial/data/metadata//test_011021_ranked/train_aug.json") 
    for item in train:
        print(item['text'])
        print(f"All metadata: {item['obj_metadata']}")
        print(f"Ranked: {item['pow2_entropy_ranked_obj']}")
        print()

    print("\n-----------------------------\n")
    test = load_jsonl("//private/home/simarora/tutorial/data/metadata//test_011021_ranked/test_aug.json") 
    for item in test:
        print(item['text'])
        print(f"All metadata: {item['ents']['obj']['metadata']}")
        print(f"Ranked: {item['pow2_entropy_ranked_obj']}")
        print()

if __name__ == "__main__":
    check_test_1()