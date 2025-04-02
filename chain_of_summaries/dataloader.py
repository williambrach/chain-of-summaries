import json


def order_data(
    data: dict,
    synthetic_questions: list,
    train_questions: list,
    validation_questions: list
) -> dict:
    file_names = [tq["file_name"] for tq in data["sites"]]
    file_names = set(file_names)

    synthetic_questions_by_file = {}
    for tq in synthetic_questions:
        if not tq:
            continue
        file_name = tq["file_name"]
        if file_name not in synthetic_questions_by_file:
            synthetic_questions_by_file[file_name] = []
        synthetic_questions_by_file[file_name].append(tq)

    # Group questions by file_name
    train_questions_by_file = {}
    for tq in train_questions:
        if not tq:
            continue
        file_name = tq["file_name"]
        if file_name not in train_questions_by_file:
            train_questions_by_file[file_name] = []
        train_questions_by_file[file_name].append(tq)

    eval_questions_by_file = {}
    for eq in validation_questions:
        file_name = eq["file_name"]
        if file_name not in eval_questions_by_file:
            eval_questions_by_file[file_name] = []
        eval_questions_by_file[file_name].append(eq)

    # Filter files that have at least 3 eval questions
    valid_files = [
        file_name
        for file_name, questions in eval_questions_by_file.items()
        if len(questions) >= 3 and file_name in file_names
    ]

    # Now create the three lists in the same order
    ordered_sites = []
    ordered_train_questions = []
    ordered_synthetic_questions = []
    ordered_eval_questions = []

    for file_name in valid_files:
        # Add site data
        site_data = next(
            (site for site in data["sites"] if site["file_name"] == file_name), None
        )
        if site_data:
            ordered_sites.append(site_data)

        # Add train questions
        if file_name in synthetic_questions_by_file:
            ordered_synthetic_questions.append(synthetic_questions_by_file[file_name])
        else:
            ordered_synthetic_questions.append([])  # Empty list if no train questions


        # Add train questions
        if file_name in train_questions_by_file:
            ordered_train_questions.append(train_questions_by_file[file_name])
        else:
            ordered_train_questions.append([])  # Empty list if no train questions

        # Add eval questions
        if file_name in eval_questions_by_file:
            ordered_eval_questions.append(eval_questions_by_file[file_name])
        else:
            ordered_eval_questions.append([])  # Empty list if no eval questions

    # Update the data
    data["sites"] = ordered_sites
    train_questions = ordered_train_questions
    validation_questions = ordered_eval_questions
    synthetic_questions = ordered_synthetic_questions
    return data, synthetic_questions, train_questions, validation_questions

def load_raw_data(
        initial :str,
        synthetic :str,
        train :str,
        validation : str = "trivia_qa_validation"
) -> tuple:
    with open(initial) as f:
        data = json.load(f)
    # Load synthetic questions
    with open(synthetic) as f:
        synthetic_questions_raw = json.load(f)
        flatterned_synthetic_questions_raw = []
        for tq in synthetic_questions_raw:
            flatterned_synthetic_questions_raw.extend(tq)
        synthetic_questions_raw = flatterned_synthetic_questions_raw
    # load train questions
    with open(train) as f:
        train_questions_raw = json.load(f)
    # Load validation questions
    with open(validation) as f:
        eval_questions_raw = json.load(f)
    return data, synthetic_questions_raw, train_questions_raw, eval_questions_raw

