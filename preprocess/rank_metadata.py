import argparse
import math
from scipy.stats import entropy
from collections import Counter, defaultdict
import spacy
from tqdm import tqdm

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

from load_dataset import load_fewrel, load_tacred, load_open_entity
from test_ranking import load_test_dataset, load_test_dataset2
from load_dataset import load_bootleg_resources, save_datasets_stats


def get_pmi_words(dataset, all_types_counter, subj_metadata_choices, obj_metadata_choices):
    mi_scores = compute_mutual_info(dataset, subj_metadata_choices, obj_metadata_choices)

    top_mi_words_by_class = defaultdict(dict)
    for term in mi_scores.keys():
        for label, score in mi_scores[term].items():
            if term in all_types_counter.keys():
                top_mi_words_by_class[label][term] = score
    return top_mi_words_by_class, mi_scores


# input a dataset with a list of entities and a list of tuples of metadata source and metadata piece
def get_type_entropy_ranks(datasets, subj_metadata_choices, obj_metadata_choices):
    all_types_counter = Counter()
    all_label_counter = Counter()

    # first precompute counts of types and over the training data
    for ex in tqdm(datasets['train']):

        labels = ex['label']
        if type(labels) != list:
            labels = [labels]
        for label in labels:
            all_label_counter[label] += 1

        subj_chosen_metadata, obj_chosen_metadata = get_ex_metadata(ex, subj_metadata_choices, obj_metadata_choices)
        if 'subj' in ex['ents']:
            # entity typing doesn't have this
            subj_chosen_metadata = []
            for metadata in subj_chosen_metadata:
                all_types_counter[metadata] += 1
        for metadata in obj_chosen_metadata:
            all_types_counter[metadata] += 1

    # compute p(y) for y in Y
    ty_probs = defaultdict(dict)
    frequencies = {}
    total_labels = sum([v for k, v in all_label_counter.items()])
    for label, cnt in all_label_counter.items():
        frequencies[label] = cnt / total_labels

    # for each type m, get the conditional prob in each class
    top_mi_words_by_class, mi_scores = get_pmi_words(datasets['train'], all_types_counter, subj_metadata_choices, obj_metadata_choices)
    for ty, _ in all_types_counter.items():
        total_value = 0
        for label, ty_scores in top_mi_words_by_class.items():
            pmi = ty_scores[ty]
            cond_prob = (math.pow(2, pmi)) * frequencies[label]
            ty_probs[ty][label] = cond_prob
            total_value += cond_prob

        # normalize scores
        if total_value:
            for label, score in ty_probs[ty].items():
                ty_probs[ty][label] = score / total_value

    # compute entropy scores
    ty2entropy = {}
    for ty, cls_dict in ty_probs.items():
        ty_ent = entropy(list(cls_dict.values()))
        ty2entropy[ty] = ty_ent

    # rank the metadata by entropy scores
    ranked_by_entropy = {k: v for k, v in sorted(ty2entropy.items(), key=lambda item: item[1])}

    return ranked_by_entropy, top_mi_words_by_class, mi_scores, all_types_counter


def compute_mutual_info(dataset, subj_metadata_choices, obj_metadata_choices):
    word_counts = Counter()
    class_counts = Counter()
    word_in_class = defaultdict(Counter)
    total_type_occurrences = 0
    total_label_occurrences = 0
    for ex in dataset:

        # collect types
        all_types = []

        # type meta info (note just 'obj' for entity-typing)
        subj_chosen_metadata, obj_chosen_metadata = get_ex_metadata(ex, subj_metadata_choices, obj_metadata_choices)
        if 'subj' in ex['ents']:
            for metadata in subj_chosen_metadata:
                value = metadata
                all_types.append(value)
        for metadata in obj_chosen_metadata:
            value = metadata
            all_types.append(value)
        total_type_occurrences += len(all_types)

        # collect class
        label = ex['label']
        if type(label) != list:
            label = [label]
        for cl in label:
            class_counts[cl] += 1
            total_label_occurrences += 1

        # compute joint occurrences
        for ty in set(all_types):
            word_counts[ty] += 1
            for cl in label:
                word_in_class[ty][cl] += 1

    mi_scores = defaultdict(dict)
    for wd in word_counts.keys():
        for cl in class_counts.keys():
            # p (type and class) = p(type | class) * p(class)
            num = (word_in_class[wd][cl]/class_counts[cl]) * class_counts[cl]/total_label_occurrences

            # p(type) * p(class) 
            denom = (word_counts[wd]/total_label_occurrences) * (class_counts[cl]/total_label_occurrences)

            # pmi = log( p (type and class ) / p(type) * p(class) )
            if num != 0:
                mi_scores[wd][cl] = math.log2(num / denom)
            # log(0) -> -inf
            else:
                mi_scores[wd][cl] = -10000
    return mi_scores
    

def get_ex_metadata(ex, subj_metadata_choices, obj_metadata_choices):
    if 'subj' in ex['ents']:
        # entity typing doesn't have this
        subj_metadata = ex['ents']['subj']['metadata']
    obj_metadata = ex['ents']['obj']['metadata']

    subj_chosen_metadata = []
    if 'subj' in ex['ents']:
        # entity typing doesn't have this
        subj_chosen_metadata = []
        for metadata_dict in subj_metadata:
            if metadata_dict[0] in subj_metadata_choices:
                subj_chosen_metadata.append(metadata_dict[1])

    obj_chosen_metadata = []
    for metadata_dict in obj_metadata:
        if metadata_dict[0] in obj_metadata_choices:
            obj_chosen_metadata.append(metadata_dict[1])

    return subj_chosen_metadata, obj_chosen_metadata


def get_modified_datasets(args, datasets, subj_metadata_choices, obj_metadata_choices, num_classes=-1):
    assert num_classes > 0, print("Invalid number of classes.")

    print("Collecting pmi scores for types.")
    ranked_by_entropy, _, _, all_types_counter = get_type_entropy_ranks(datasets, subj_metadata_choices, obj_metadata_choices)

    print("Ranking types per example.")
    modified_datasets = {}

    worst_entropy = math.log2(num_classes)
    print(f"Worst Entropy on this Datast: {worst_entropy}")
    thresh = args.thresh 
    for name, dataset in datasets.items():
        modified_dataset = []
        for ex in tqdm(dataset):

            if 'subj' in ex['ents']:
                # entity typing doesn't have this
                ex['subj_metadata'] = ex['ents']['subj']['metadata']
            ex['obj_metadata'] = ex['ents']['obj']['metadata']

            # Algorithmic selection
            subj_chosen_metadata, obj_chosen_metadata = get_ex_metadata(ex, subj_metadata_choices, obj_metadata_choices)

            if 'subj' in ex['ents']:
                # entity typing doesn't have this
                subj_types_scores = defaultdict(float)
                subj_types_popularities = defaultdict(float)
                for ty in subj_chosen_metadata:
                    if ty in ranked_by_entropy:
                        score = ranked_by_entropy[ty]
                        count = all_types_counter[ty]
                        if score < worst_entropy*thresh:
                            subj_types_scores[ty] = score
                        subj_types_popularities[ty] = count

                # sort from lowest to highest entropy
                entropy_ranked_subj = [k for k, v in sorted(subj_types_scores.items(),
                                                            key=lambda item: item[1])]

                # sort from highest to lowest popularity
                pop_ranked_subj = [k for k, v in sorted(subj_types_popularities.items(),
                                                        key=lambda item: item[1],
                                                        reverse=True)]
                ex['pow2_entropy_ranked_subj'] = entropy_ranked_subj.copy()
                ex['pop_ranked_subj'] = pop_ranked_subj.copy()

            obj_types_scores = defaultdict(float)
            obj_types_popularities = defaultdict(float)
            for ty in obj_chosen_metadata:
                if ty in ranked_by_entropy:
                    score = ranked_by_entropy[ty]
                    count = all_types_counter[ty]
                    if score < worst_entropy*thresh:
                        obj_types_scores[ty] = score
                    obj_types_popularities[ty] = count
            entropy_ranked_obj = [k for k, v in sorted(obj_types_scores.items(),
                                                       key=lambda item: item[1])]
            
            pop_ranked_obj = [k for k, v in
                              sorted(obj_types_popularities.items(),
                                     key=lambda item: item[1],
                                     reverse=True)]
            ex['pow2_entropy_ranked_obj'] = entropy_ranked_obj.copy()
            ex['pop_ranked_obj'] = pop_ranked_obj.copy()
            modified_dataset.append(ex)

        modified_datasets[name] = modified_dataset
    save_datasets_stats(modified_datasets, f"/{args.save_path}/{args.task}_{args.suffix}_ranked/")
    return modified_datasets


def task_specific(args, label_map, ex, types, relations):
    if args.task == "fewrel":
        label_names = [k for k, v in label_map.items()]
        relations = [rel for rel in relations if rel in label_names] 

    return types, relations


# Assumes entity-tagged text
def collect_metadata(args, datasets, label_map):
    new_datasets = {}
    print("Loading metadata files.")
    input_dir, emb_dir, entity_dump, types_wd, types_rel, kg_syms, a2q, qid2desc = load_bootleg_resources()

    print("Collecting metadata.")
    for name, dataset in datasets.items():
        new_dataset = []
        for ex in tqdm(dataset):
            doc = nlp(ex['text'])

            for key, ent_dict in ex['ents'].items():
                ent = ent_dict['qid']
                metadata = []

                char_idx = 0
                entity_span = list(range(ent[1], ent[2]))
                ner = []
                pos = []
                for token in doc:
                    token_span = list(range(char_idx, char_idx + len(token.text)))
                    if len(frozenset(entity_span).intersection(frozenset(token_span))) > 0:
                        ner = token.tag_
                        pos = token.pos_
                        break

                # not all subjects and objects have QIDs
                if ent[0] in ["Q-1", "UNK"]:
                    ex['ents'][key]['metadata'] = metadata
                    continue

                # wikidata
                types = types_wd.get_types(ent).copy()
                types = [('type', ty) for ty in types]
                relations = types_rel.get_types(ent).copy()
                types, relations = task_specific(args, label_map, ex, types, relations)
                relations = [('relation', rel) for rel in relations]

                # wikipedia (could add n-grams of this to the metadata list too)
                if ent in qid2desc:
                    desc = [('desc', qid2desc[ent])]
                else:
                    desc = []

                metadata.extend(types)
                metadata.extend(relations)
                metadata.extend(desc)
                metadata.extend(ner)
                metadata.extend(pos)
                ex['ents'][key]['metadata'] = metadata
            new_dataset.append(ex)
        new_datasets[name] = new_dataset

    save_datasets_stats(new_datasets, f"/{args.save_path}/{args.task}_{args.suffix}/")
    return new_datasets


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

     application_config.add_argument(
        "--save_path", type=str, required=True,
        help="Path to save data with annotations"
    )

    application_config.add_argument(
        "--task", type=str, required=True,
        help="Text classification tasks"
    )

    application_config.add_argument(
        "--suffix", type=str, required=True,
        help="Folder to save data suffix"
    )

    application_config.add_argument(
        "--thresh", type=float, required=True, default=0.5,
        help="PMI Threshold"
    )


# The goal of this script is to input the raw data and insert the metadata based
# on pointwise mutual information.
if __name__ == "__main__":

    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "DataShape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_application_args(parser)
    args = parser.parse_args()

    # Load the dataset in raw
    if args.task == "fewrel":
        datasets, label_map = load_fewrel(args.save_path)
        subj_metadata_choices = ['type', 'relation']
        obj_metadata_choices = ['type']
    elif args.task == "test":
        datasets, label_map = load_test_dataset()
        subj_metadata_choices = []
        obj_metadata_choices = ['type']
    elif args.task == "test2":
        datasets, label_map = load_test_dataset2()
        subj_metadata_choices = []
        obj_metadata_choices = ['type']
    else:
        exit
    print("Loaded Dataset.")

    has_metadata = 0
    for ex in datasets['train']:
        if ex['ents']:
            if 'metadata' in ex['ents']['obj']:
                print("Has metadata")
                has_metadata = 1
                break
            else:
                collect_metadata(args, datasets, label_map)
                break
        else:
            print(ex.keys())

    print("Running algorithm.")
    modified_datasets = get_modified_datasets(args, datasets, subj_metadata_choices, obj_metadata_choices, num_classes=len(label_map))


