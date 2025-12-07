import gc
import random
import torch
import os
import re
import itertools
import numpy as np

considered_aspects_tasks = {
    "tasd": ["aspect term", "aspect category", "sentiment polarity"],
    "asqp": ["aspect term", "aspect category", "sentiment polarity", "opinion term"]
}
stop_paraphrase = rf"[SSEP]\n"
stop_tuple = rf"]\n"

TOOLKIT_PATH = '/home/hellwig/absa-toolkit'


def phrases_with_signs(valid_phrases):
    valid_phrases_with_signs = []
    for p in valid_phrases:
        if "'" in p:
            valid_phrases_with_signs.append(rf'"' + p + rf'"')
            pass
        valid_phrases_with_signs.append(rf"'" + p + rf"'")
        
    return valid_phrases_with_signs

def parse_label_string(label_string, task):
    label_string = label_string.strip()
    label_string = label_string[1:-1]
    # check if array-based tuples
    if label_string.startswith("["):
        array_based = True
    elif label_string.startswith("("):
        array_based = False

    if array_based:
        tuples = label_string.split("], [")
    else:
        tuples = label_string.split("), (")

    tuples_list = []
    for t in tuples:
        t = t.strip()

        if array_based:
            if not t.startswith("["):
                t = "[" + t
            if not t.endswith("]"):
                t = t + "]"
        else:
            if not t.startswith("("):
                t = "(" + t
            if not t.endswith(")"):
                t = t + ")"

        if task == "tasd":
            if array_based:
                # Allow both single and double quotes for first element
                pattern = r"\[['\"](.+?)['\"], '(.+?)', '(.+?)'\]"
            else:
                pattern = r"\(['\"](.+?)['\"], '(.+?)', '(.+?)'\)"
        elif task == "asqp":
            if array_based:
                # Allow both single and double quotes for first and fourth element
                pattern = r"\[['\"](.+?)['\"], '(.+?)', '(.+?)', ['\"](.+?)['\"]\]"
            else:
                pattern = r"\(['\"](.+?)['\"], '(.+?)', '(.+?)', ['\"](.+?)['\"]\)"
        matches = re.match(pattern, t)
        if matches:
            tuples_list.append(matches.groups())
    # convert tuples to tuples
    tuples_list = [tuple(t) for t in tuples_list]

    return tuples_list

def get_dataset(dataset_name, split, task, base_path=TOOLKIT_PATH + "/data"):
    dataset_path = os.path.join(
        base_path, "datasets", task, dataset_name, f"{split}.txt")
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    dataset = []
    for line in lines:
        text, label = line.strip().split("####")
        label = parse_label_string(label, task)
        # strip all strings within label tuples
        if task == "tasd" and "rest" in dataset_name:
            text = text.replace(" '", "'").replace(" )", ")").replace(" (", "(").replace(" /", "/").replace(" .", ".").replace(" ,", ",").replace(' "', '"').replace(" +", "+").replace(" $", "$")
        label = [tuple(s.strip() for s in tup) for tup in label]
        dataset.append({
            "text": text,
            "label": label
        })
    return dataset


def get_fs_examples(dataset_name, task, n_shot):
    fs_examples_path = os.path.join(
        TOOLKIT_PATH, "data", "fs_examples", task, dataset_name, f"fs_{n_shot}", "examples.txt")
    with open(fs_examples_path, "r") as f:
        lines = f.readlines()
    fs_examples = []
    for line in lines[:n_shot]:
        text, label = line.strip().split("####")
        # text = text.replace("  (", " (").replace(" )", ")").replace("  $", " $").replace(" /", "/").replace(" +", "+")
        label = parse_label_string(label, task)
        fs_examples.append({
            "text": text,
            "label": label
        })
    return fs_examples


def get_fs_examples_new_with_seed(dataset_name, task, n_shot, seed, use_dev=False):
    fs_examples_path_train = f"/home/hellwig/absa-toolkit/data/datasets/{task}/{dataset_name}/train.txt"
    fs_examples_path_dev = f"/home/hellwig/absa-toolkit/data/datasets/{task}/{dataset_name}/dev.txt"

    lines_train = []
    if not use_dev:
        with open(fs_examples_path_train, "r") as f:
            lines_train = f.readlines()

        random.seed(seed)
        random.shuffle(lines_train)

    
    lines_dev = []
    if use_dev:
        with open(fs_examples_path_dev, "r") as f:
            lines_dev = f.readlines()

        random.seed(seed)
        random.shuffle(lines_dev)

    lines = lines_train + lines_dev

    fs_examples = []
    
    if len(lines) >= n_shot:
        lines = lines[:n_shot]
    
    for line in lines:
        text, label = line.strip().split("####")
        # text = text.replace("  (", " (").replace(" )", ")").replace("  $", " $").replace(" /", "/").replace(" +", "+")
        label = parse_label_string(label, task)
        fs_examples.append({
            "text": text,
            "label": label
        })
    return fs_examples


def get_unique_aspect_categories(dataset_name, task, base_path=TOOLKIT_PATH + "/data"):
    train = get_dataset(dataset_name, "train", task, base_path=base_path)
    test = get_dataset(dataset_name, "test", task, base_path=base_path)
    dev = get_dataset(dataset_name, "dev", task, base_path=base_path)

    unique_aspect_categories = []
    for split in [train, test, dev]:
        for item in split:
            for tuple in item["label"]:
                unique_aspect_categories.append(tuple[1])

    # uniqueify aspect categories
    unique_aspect_categories = list(set(unique_aspect_categories))

    # sort unique aspect categories
    unique_aspect_categories.sort()

    return unique_aspect_categories


def get_unique_aspect_categories_in_list(labels):
    unique_aspect_categories = []
    for label in labels:
        for tuple in label:
            unique_aspect_categories.append(tuple[1])

    # uniqueify aspect categories
    unique_aspect_categories = list(set(unique_aspect_categories))

    # sort aspect categories
    unique_aspect_categories.sort()

    return unique_aspect_categories


def get_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    n_tp, n_fp, n_fn = 0, 0, 0
    n_gold, n_pred = 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        # Compute True Positives and False Positives
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
            else:
                n_fp += 1

        # Compute False Negatives
        for t in gold_pt[i]:
            if t not in pred_pt[i]:
                n_fn += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision != 0 or recall != 0 else 0

    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'TP': n_tp,
        'FP': n_fp,
        'FN': n_fn
    }

    return scores


def compute_f1_macro(pred_pt, gold_pt):
    unique_aspect_categories = get_unique_aspect_categories_in_list(gold_pt)
    scores = {}
    for ac in unique_aspect_categories:
        pred_ac = []
        gold_ac = []
        for i in range(len(pred_pt)):
            gold_ac.append([t for t in gold_pt[i] if t[1] == ac])
            pred_ac.append([t for t in pred_pt[i] if t[1] == ac])
        scores[ac] = get_scores(pred_ac, gold_ac)
    f1_macro = sum([scores[ac]['f1'] for ac in unique_aspect_categories]
                   ) / len(unique_aspect_categories)
    return f1_macro


def get_all_scores(pred_pt, gold_pt):
    scores = get_scores(pred_pt, gold_pt)
    f1_macro = compute_f1_macro(pred_pt, gold_pt)
    scores['f1_macro'] = f1_macro
    return scores


def tuple_position_to_idx(considered_aspects=["aspect term", "aspect category", "sentiment polarity", "opinion term"]):
    tuple_idxs = []
    for considered_aspect in considered_aspects:
        if considered_aspect == "aspect term":
            tuple_idxs.append(0)
        elif considered_aspect == "aspect category":
            tuple_idxs.append(1)
        elif considered_aspect == "sentiment polarity":
            tuple_idxs.append(2)
        elif considered_aspect == "opinion term":
            tuple_idxs.append(3)
    return tuple_idxs


def encode_tuple(sentiment_elements, task, considered_aspects, given_position=None):
    if given_position is None:
        given_position = [0, 1, 2, 3] if task == "asqp" else [0, 1, 2]

    idx_convert = {
        0: "aspect term",
        1: "aspect category",
        2: "sentiment polarity",
        3: "opinion term"
    }

    # considered_aspects : zielvormat
    # given_position : position der tuple orientierend an idx_convert
    tuples = []
    for element in sentiment_elements:
        new_tuple = {}
        idx = 0
        for pos in given_position:
            new_tuple[idx_convert[pos]] = element[idx]
            idx += 1

        new_tuple_list = tuple([new_tuple[aspect]
                               for aspect in considered_aspects])
        tuples.append(new_tuple_list)

    return tuples


def encode_tuple_elsevier(sentiment_elements):
    tuples = []
    for element in sentiment_elements:
        # element[1] is aspect category, element[2] is sentiment polarity
        tuples.append((element[1], element[2]))
    return tuples


def encode_paraphrase(sentiment_elements, task, considered_aspects):
    paraphrases = []
    encode_sentiment_polarity = {
        "positive": "great",
        "negative": "bad",
        "neutral": "ok"
    }
    for tuple_raw in sentiment_elements:
        if task == "tasd":
            paraphrase = f"{tuple_raw[1]} is {encode_sentiment_polarity[tuple_raw[2]]} because {'it' if tuple_raw[0] == 'NULL' else tuple_raw[0]} is {encode_sentiment_polarity[tuple_raw[2]]} [SSEP]"
        elif task == "asqp":
            paraphrase = f"{tuple_raw[1]} is {encode_sentiment_polarity[tuple_raw[2]]} because {'it' if tuple_raw[0] == 'NULL' else tuple_raw[0]} is {tuple_raw[3]} [SSEP]"
        else:
            raise ValueError(f"Unknown task: {task}")
        paraphrases.append(paraphrase)
    return " ".join(paraphrases)


def get_prompt_elsevier(dataset_name, task, label_gen, examples=[], unique_aspect_categories=None,  polarities=["positive", "negative", "neutral"]):

    if unique_aspect_categories is None:
        unique_aspect_categories = get_unique_aspect_categories(
            dataset_name, task)
    unique_aspect_categories.sort()
    unique_aspect_categories_str = ", ".join(
        f"'{cat}'" for cat in unique_aspect_categories)
    polarities_str = ", ".join(f"'{pol}'" for pol in polarities)

    prompt = "You are given a list of tuples (=Sentiment Elements), each consisting of an aspect category and a sentiment polarity.\nEach tuple represents an opinion which is addressed in the text.\n\n"

    prompt += f"- The 'aspect category' refers to the category that aspect belongs to, and the available categories includes: {unique_aspect_categories_str}.\n"
    prompt += f"- The 'sentiment polarity' refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: {polarities_str}.\n"

    prompt += f"\nGiven a list of tuples of opinions, generate a single sentence associated with the given label.\n\n"

    # Add few shot examples
    for example in examples:
        prompt += f"Sentiment elements: {encode_tuple_elsevier(example['label'])}\nText: {example['text']}\n"

    # Add examples to be predicted
    prompt += f"Sentiment elements: {label_gen}\nText: "

    return prompt


def get_prompt(dataset_name, task, text_pred, examples=[], unique_aspect_categories=None, considered_aspects=None, polarities=["positive", "negative", "neutral"], approach="tuple"):
    if considered_aspects is None and task == "tasd":
        considered_aspects = considered_aspects_tasks["tasd"]
    elif considered_aspects is None and task == "asqp":
        considered_aspects = considered_aspects_tasks["asqp"]
    if unique_aspect_categories is None:
        unique_aspect_categories = get_unique_aspect_categories(
            dataset_name, task)
    unique_aspect_categories.sort()
    unique_aspect_categories_str = ", ".join(
        f"'{cat}'" for cat in unique_aspect_categories)
    polarities_str = ", ".join(f"'{pol}'" for pol in polarities)
    implicit_aspect = "NULL" if approach == "tuple" else "it"
    output_format_definition = ""

    if approach == "tuple":
        output_format_definition += "[(" + ", ".join(
            f"'{cat}'" for cat in considered_aspects) + "), ...]."
    elif approach == "paraphrase":
        if task == "tasd":
            output_format_definition += "<aspect category> is <sentiment polarity text> because <aspect term> is <sentiment polarity text> [SSEP] ... ."
        elif task == "asqp":
            output_format_definition += "<aspect category> is <sentiment polarity text> because <aspect term> is <opinion term> [SSEP] ... ."
        output_format_definition += " Aspects are seperated by a seperator token (`[SSEP]`).  \n<sentiment polarity text> is a textual representation of the sentiment polarity, 'positive' → 'great', 'negative' → 'bad', 'neutral' → 'ok'."

    prompt = "According to the following sentiment elements definition: \n\n"

    for considered_aspect in considered_aspects:
        if "aspect term" == considered_aspect:
            prompt += f"- The 'aspect term' is the exact word or phrase in the text that represents a specific feature, attribute, or aspect of a product or service that a user may express an opinion about, the aspect term might be '{implicit_aspect}' for implicit aspect.\n"
        if "aspect category" == considered_aspect:
            prompt += f"- The 'aspect category' refers to the category that aspect belongs to, and the available categories includes: {unique_aspect_categories_str}.\n"
        if "sentiment polarity" == considered_aspect:
            prompt += f"- The 'sentiment polarity' refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: {polarities_str}.\n"
        if "opinion term" == considered_aspect:
            prompt += f"- The 'opinion term' is the exact word or phrase in the text that refers to the sentiment or attitude expressed by a user towards a particular aspect or feature of a product or service, the aspect term might be '{implicit_aspect}' for implicit opinion.\n"

    prompt += f"\nRecognize all sentiment elements with their corresponding {', '.join(considered_aspects)} in the following text with the format of {output_format_definition}\n\n"

    # Add few shot examples
    for example in examples:
        if approach == "tuple":
            prompt += f"Text: {example['text']}\nSentiment elements: {encode_tuple(example['label'], task, considered_aspects)}\n"
        elif approach == "paraphrase":
            prompt += f"Text: {example['text']}\nSentiment elements: {encode_paraphrase(example['label'], task, considered_aspects)}\n"

    # Add examples to be predicted
    prompt += f"Text: {text_pred}\nSentiment elements: "

    return prompt


def replace_tuple_index(tup, index, value):
    """
    Replaces the element at 'index' in 'tup' with 'value' and returns a new tuple.

    Parameters:
    - tup (tuple): The original tuple.
    - index (int): The index to replace.
    - value: The new value to insert.

    Returns:
    - tuple: A new tuple with the value replaced.
    """

    if not isinstance(tup, tuple):
        raise TypeError("Input must be a tuple.")

    if index < 0 or index >= len(tup):
        raise IndexError("Index out of range.")

    # Replace using slicing
    return tup[:index] + (value,) + tup[index+1:]


def validate_tuple(
    output_str, text_to_label, task, dataset_name, unique_aspect_categories=None, considered_aspects=None
):
    if unique_aspect_categories is None:
        unique_aspect_categories = get_unique_aspect_categories(
            dataset_name, task)

    if considered_aspects is None and task == "tasd":
        considered_aspects = ["aspect term",
                              "aspect category", "sentiment polarity"]
    elif considered_aspects is None and task == "asqp":
        considered_aspects = [
            "aspect term", "aspect category", "sentiment polarity", "opinion term"]

    n_elements_task = {"tasd": 3, "asqp": 4}

    # filter list
    try:
        list_str = re.findall(r"\[.*\]", output_str)[0]
    except:
        return False
    # eval list
    try:
        output_list = eval(list_str)
    except:
        return False
    # check if list contains only tuples of length 3, inside a try catch
    try:
        for item in output_list:
            if not isinstance(item, tuple) or len(item) != n_elements_task[task]:
                return False
    except:
        return False

    # trim each item in the list
    output_list = [tuple(str(x).strip() for x in item) for item in output_list]
    invalid_tuples_idx = set()

    for idx, tuple_pred in enumerate(output_list):
        for idx_asp, considered_aspect in enumerate(considered_aspects):
            if considered_aspect == "aspect category":
                if tuple_pred[idx_asp] not in unique_aspect_categories:
                    invalid_tuples_idx.add(idx)
            elif considered_aspect == "sentiment polarity":
                if tuple_pred[idx_asp] not in ["positive", "negative", "neutral"]:
                    invalid_tuples_idx.add(idx)
            elif considered_aspect == "aspect term":
                if tuple_pred[idx_asp] == "":
                    invalid_tuples_idx.add(idx)
                elif tuple_pred[idx_asp] not in text_to_label and tuple_pred[idx_asp] != "NULL":
                    if tuple_pred[idx_asp].lower() in text_to_label.lower():
                        pos = text_to_label.lower().index(
                            tuple_pred[idx_asp].lower())
                        output_list[idx] = replace_tuple_index(
                            output_list[idx], idx_asp, text_to_label[pos:pos+len(tuple_pred[idx_asp])])
                    else:
                        invalid_tuples_idx.add(idx)
            elif considered_aspect == "opinion term":
                if tuple_pred[idx_asp] == "":
                    invalid_tuples_idx.add(idx)
                elif tuple_pred[idx_asp] not in text_to_label:
                    if tuple_pred[idx_asp].lower() in text_to_label.lower():
                        pos = text_to_label.lower().index(
                            tuple_pred[idx_asp].lower())
                        output_list[idx] = replace_tuple_index(
                            output_list[idx], idx_asp, text_to_label[pos:pos+len(tuple_pred[idx_asp])])
                    else:
                        invalid_tuples_idx.add(idx)

    if len(invalid_tuples_idx) > 0:
        output_list = [item for idx, item in enumerate(
            output_list) if idx not in invalid_tuples_idx]
        return False, output_list

    return output_list


def validate_paraphrase(
    output_str, text_to_label, task, dataset_name, unique_aspect_categories=None, considered_aspects=None
):
    if unique_aspect_categories is None:
        unique_aspect_categories = get_unique_aspect_categories(
            dataset_name, task)

    if considered_aspects is None and task == "tasd":
        considered_aspects = ["aspect term",
                              "aspect category", "sentiment polarity"]
    elif considered_aspects is None and task == "asqp":
        considered_aspects = [
            "aspect term", "aspect category", "sentiment polarity", "opinion term"]

    if "[SSEP]" not in output_str:
        return False, "Output does not contain the expected separator '[SSEP]'."

    output_list_raw = [output for output in output_str.split(
        "[SSEP]") if output.strip() != ""]

    all_tuples = []
    invalid_tuples_idx = set()

    for idx, item in enumerate(output_list_raw):
        try:

            category = re.split(r" is (ok|bad|great) because ", item)[
                0].strip()
            if category not in unique_aspect_categories:
                invalid_tuples_idx.add(idx)

            polarity = re.search(r"is (ok|bad|great) because", item)
            if polarity:
                polarity = polarity.group(1).strip()

                if polarity == "ok":
                    polarity = "neutral"
                elif polarity == "bad":
                    polarity = "negative"
                elif polarity == "great":
                    polarity = "positive"
                else:
                    invalid_tuples_idx.add(idx)
            else:
                invalid_tuples_idx.add(idx)

            aspect_term = re.search(r"because (.*) is", item)
            if aspect_term:
                aspect_term = aspect_term.group(1).strip()
            else:
                invalid_tuples_idx.add(idx)

            if task == "asqp":
                if "is" in item:
                    opinion_term = item.split("is")[-1].strip()
                else:
                    invalid_tuples_idx.add(idx)

            if aspect_term == "it":
                aspect_term = "NULL"
            elif aspect_term == "":
                invalid_tuples_idx.add(idx)
            elif aspect_term.lower() in text_to_label.lower():
                pos = text_to_label.lower().index(aspect_term.lower())
                aspect_term = text_to_label[pos:pos + len(aspect_term)].strip()

            if task == "asqp":
                if opinion_term == "":
                    invalid_tuples_idx.add(idx)
                elif opinion_term.lower() in text_to_label.lower():
                    pos = text_to_label.lower().index(opinion_term.lower())
                    opinion_term = text_to_label[pos:pos +
                                                 len(opinion_term)].strip()

            if task == "asqp":
                new_tuple = (aspect_term, category, polarity, opinion_term)
            if task == "tasd":
                new_tuple = (aspect_term, category, polarity)

            all_tuples.append(new_tuple)
        except:
            pass

    if len(invalid_tuples_idx) > 0:
        return False, [t for i, t in enumerate(all_tuples) if i not in invalid_tuples_idx]

    return all_tuples


def get_regex_pattern_paraphrase(unique_aspect_categories, task):
    category_pattern = "|".join(cat for cat in unique_aspect_categories)
    paraphrase_polarity = ["ok", "bad", "great"]
    paraphrase_polarity_pattern = "|".join(paraphrase_polarity)
    if task == "asqp":
        paraphrase_pattern_str = rf"({category_pattern}) is ({paraphrase_polarity_pattern}) because (it|(\w[^\n']+)) is ([^\n']+))"
    else:
        paraphrase_pattern_str = rf"({category_pattern}) is ({paraphrase_polarity_pattern}) because (it|(\w[^\n']+)) is ({paraphrase_polarity_pattern})"

    paraphrase_pattern_str = rf"{paraphrase_pattern_str}( [SSEP] {paraphrase_pattern_str})* [SSEP]\n"
    return paraphrase_pattern_str


def escape_except_space(text):
    return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', text)


def get_regex_pattern_tuple(unique_aspect_categories, polarities, considered_aspects, valid_phrases):
    # Tuple Patternteile
    category_pattern = "|".join(cat for cat in unique_aspect_categories)
    polarity_pattern = "|".join(polarities)

    # Tuple pattern
    tuple_pattern_parts = []
    for aspect in considered_aspects:
        if aspect == "aspect term":
            tuple_pattern_parts += [rf"('NULL'|{'|'.join(escape_except_space(rf""+p) for p in phrases_with_signs(valid_phrases))})"]
        elif aspect == "aspect category":
            tuple_pattern_parts += [rf"'({category_pattern})'"]
        elif aspect == "sentiment polarity":
            tuple_pattern_parts += [rf"'({polarity_pattern})'"]
        elif aspect == "opinion term":
            tuple_pattern_parts += [rf"({'|'.join(escape_except_space(rf""+p) for p in phrases_with_signs(valid_phrases))})"]

    tuple_pattern_str = rf"\(" + rf", ".join(tuple_pattern_parts) + rf"\)"
    tuple_pattern_str = rf"\[{tuple_pattern_str}(, {tuple_pattern_str})*\]\n"

    return tuple_pattern_str


def majority_vote(comparison_labels, majority_threshold=None):
    n_lists = len(comparison_labels)
    if majority_threshold is None:
        majority_threshold = n_lists // 2 + 1
    tuple_occurrences = {}

    for lst in comparison_labels:
        current_list_counts = {}
        for tuple_item in lst:
            current_list_counts[tuple_item] = current_list_counts.get(
                tuple_item, 0) + 1

        for tuple_item, count in current_list_counts.items():
            if tuple_item not in tuple_occurrences:
                tuple_occurrences[tuple_item] = []
            tuple_occurrences[tuple_item].append(count)

    result = []
    for tuple_item, occurrences_list in tuple_occurrences.items():
        if len(occurrences_list) >= majority_threshold:
            max_count = max(occurrences_list)
            for count in range(1, max_count + 1):
                lists_with_count = sum(
                    1 for occ in occurrences_list if occ >= count)
                if lists_with_count >= majority_threshold:
                    result.append(tuple_item)

    return result


def clear_memory(variables_to_clear=None, verbose=True):
    """
    Clear memory by deleting specified variables and freeing CUDA cache.

    Args:
        variables_to_clear (list, optional): List of variable names to remove from globals.
                                            Defaults to common ML variables.
        verbose (bool, optional): Whether to print memory status. Defaults to True.
    """
    # Default variables to clear if none specified
    if variables_to_clear is None:
        variables_to_clear = ["inputs", "model", "processor", "trainer",
                              "peft_model", "bnb_config"]

    # Delete specified variables if they exist in global scope
    g = globals()
    for var in variables_to_clear:
        if var in g:
            del g[var]

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Second garbage collection pass
        gc.collect()

        # Print memory status if verbose
        if verbose:
            print(
                f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(
                f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def get_token_confidences(output):
    """
    Returns confidence scores (0-1) for the chosen token at each position.

    Args:
        output: vLLM output object

    Returns:
        List of floats: confidence scores for each generated token
    """
    confidences = []

    for logprob_dict in output.outputs[0].logprobs:
        if logprob_dict:
            # Nehme das erste (rank 1) Token
            first_token = next(iter(logprob_dict.values()))
            confidence = np.exp(first_token.logprob)
            confidences.append(confidence)

    return confidences


def find_valid_phrases_list(text: str, max_chars_in_phrase: int | None = None) -> list[str]:
    """
    Extract all valid phrases from a given text based on punctuation and formatting rules.

    Args:
        text (str): The input text.
        max_chars_in_phrase (int, optional): Maximum number of characters allowed per phrase. 
                                              If None, no limit is applied.
    Returns:
        list[str]: List of cleaned, unique phrases.
    """
    text = f" {text.strip()} "
    phrases = []

    # Collect split positions
    split_positions = {0, len(text)}

    # Define patterns for splits (merged for clarity)
    split_patterns = [
        r'(?<=\w)(?=[,!?;:\(\)])',      # before punctuation (without hyphen and period)
        r'(?<=[,!?;:\(\)])(?=\w)',      # after punctuation (without period)
        r'(?<=\))(?=\.)',               # between ) and .
        r'(?<=\))(?=\,)',               # between ) and ,
        r'\s+',                          # spaces
        r'[\$"\'""\/…]',                 # before/after special chars
        r'(?<=[a-z])(?=[A-Z])',          # camel-case boundary
        r'[^\x00-\x7F]',                 # non-ASCII chars
        r'(?<=\w)(?=-)',                 # before hyphen
        r'(?<=-)(?=\w)',                 # after hyphen
        r'(?<=\w)(?=\*)',                # before asterisk
        r'(?<=\*)',                      # after asterisk
        r'(?<=\w)(?=\.)',                # before period
        r'(?<=\.)(?=\w)',                # after period
    ]

    # Collect split indices
    for pattern in split_patterns:
        for match in re.finditer(pattern, text):
            split_positions.update({match.start(), match.end()})

    split_positions = sorted(split_positions)

    # Set character limit
    if max_chars_in_phrase is None:
        max_chars_in_phrase = len(text)

    # Generate phrases
    for i, start in enumerate(split_positions):
        for end in split_positions[i + 1:]:
            phrase = text[start:end].strip()
            if not phrase:
                continue
            if len(phrase) <= max_chars_in_phrase:
                phrases.append(phrase)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_phrases = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            unique_phrases.append(phrase)

    return unique_phrases

def setup_gpu_environment():
    """Configure GPU environment for optimal performance."""
    print("Configuring GPU environment...")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()


def get_considered_aspects(task):
    """Get considered aspects based on task type."""
    if task == "asqp":
        return ["aspect term", "aspect category", "sentiment polarity", "opinion term"]
    elif task == "tasd":
        return ["aspect term", "aspect category", "sentiment polarity"]


def get_all_combinations(list_of_str):
    all_combinations = []
    for perm in itertools.permutations(list_of_str):
        all_combinations.append(list(perm))
    return all_combinations


def convert_to_list_of_probs_for_each_token(logprob_dict):
    token_probs = []
    # prob_0 is most likely token probability
    # list should look like [[<prob_0>, <prob_1>, ...], ...]
    for token_logprobs in logprob_dict:
        probs = []
        for rank in sorted(token_logprobs.keys(), key=int):
            logprob_obj = token_logprobs[rank]
            prob = np.exp(
                logprob_obj.logprob) if logprob_obj.logprob != -np.inf else 0.0
            probs.append(prob)
        token_probs.append(probs)
    
    # remove valus lower than 0.001
    # round values above
    token_probs = [[round(p, 3) for p in probs if p >= 0.001] for probs in token_probs]
    
    return token_probs


def calculate_entropy(probs):
    entropy = 0.0
    for prob in probs:
        if prob > 0:
            entropy -= prob * np.log(prob)
    return entropy


def calculate_average_entropy(logprobs):
    total_entropy = 0.0
    for token_probs in logprobs:
        entropy = calculate_entropy(token_probs)
        total_entropy += entropy
    average_entropy = total_entropy / \
        len(logprobs) if len(logprobs) > 0 else 0.0
    return average_entropy
