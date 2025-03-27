import dspy
import pandas as pd
import torch
from litellm import completion
from tqdm import tqdm
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    pipeline,
)

from src.chain_of_summaries.metrics import (
    calculate_bert_score,
    calculate_surprisal_score,
    check_answer_em,
    check_answer_f1,
    prepare_content,
)
from src.chain_of_summaries.qna import QnA, extract_qa_pairs, synthetic_qa_prompt
from src.chain_of_summaries.summary import RefineSummary, Summarize, enc


def build_llms_txt_file(
    data: list,
    item_max_tokens: int = None,
    description: bool = True,
    output_file: str = None,
) -> str:
    """
    Builds a markdown-formatted text file from structured LLM data.

    This function takes structured data about LLMs and creates a formatted markdown
    text document with hierarchical sections. Each section includes a title, URL,
    and optionally a summary description that can be token-limited.

    Args:
        data (list): A dictionary containing 'title' and 'sites' where 'sites' is a list
                     of dictionaries with 'title', 'url', and 'summary' keys.
        item_max_tokens (int, optional): Maximum number of tokens allowed for each summary.
                                         If None, no limit is applied. Defaults to None.
        description (bool, optional): Whether to include descriptions/summaries in the output.
                                      Defaults to True.
        output_file (str, optional): Path to save the formatted text. If None, the text
                                     is returned as a string instead. Defaults to None.

    Returns:
        str: Either the path to the saved file if output_file is provided,
             or the formatted text content as a string.
    """
    title = data["title"]
    sections = [f"# {title}"]

    # Process each site in the data
    for site in data["sites"]:
        site_title = site["file_name"]
        url = site["url"]
        # Format the URL as a markdown link
        url_line = f"- [{site_title}]({url})"
        summary = site["summary"]

        # Create a list for the current section's content
        section = [f"## {site_title}", url_line]

        # Add description if enabled
        if description:
            # Truncate summary to max tokens if specified
            if item_max_tokens is not None:
                tokens = enc.encode(summary)
                truncated_tokens = tokens[:item_max_tokens]
                summary = enc.decode(truncated_tokens)

            # Format each sentence of the summary as a markdown quote
            formatted_summary = "\n".join(
                f"> {line.strip()}" for line in summary.split(".") if line.strip()
            )
            section.append(formatted_summary)

        sections.append("\n".join(section))

    # Either save to file or return as string
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("\n\n".join(sections))
        return output_file
    else:
        return "\n\n".join(sections)


class LLMSProcessor:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2056,
        **kwargs,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.cache = kwargs.get("cache", True)
        self.num_threads = kwargs.get("num_threads", 20)
        self.display_progress = kwargs.get("display_progress", True)
        self.device = kwargs.get("device", "cpu")

    def __repr__(self) -> str:
        """Return a string representation of the processor."""
        return f"LLMSProcessor(model='{self.model}')"

    def test_completion_call(self) -> str:
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        response = completion(model=self.model, messages=messages, **self.kwargs)
        return response

    def _get_summaries(self, data: list[dspy.Example]) -> list:
        """
        Generate summaries for the provided data examples.

        Args:
            data: A list of dspy.Example objects to summarize

        Returns:
            A list of summary dictionaries
        """
        # Initialize the summarization program with the configured language model
        program = Summarize()
        program.set_lm(dspy.LM(self.model, max_tokens=self.max_tokens, **self.kwargs))

        # Run evaluation with a simple pass-through metric
        evaluation_results = dspy.Evaluate(
            devset=data,
            metric=lambda x, y: True,
            num_threads=self.num_threads,
            display_progress=self.display_progress,
            return_outputs=True,
        )(program)

        # Extract and convert the summary outputs to dictionaries
        summary_dicts = [dict(output_pair[1]) for output_pair in evaluation_results[1]]
        return summary_dicts

    def create_llms_txt_data(
        self,
        title: str,
        sites: list[dict],
        summary_model: str = None,
        iteration: int = 0,  # TODO maybe remove as param?
    ) -> list[dict]:
        """
        Generate text data with summaries and optional scoring metrics.
        Args:
            title: The title for the collection of sites
            sites: List of site dictionaries containing title, content, and url
            summary_model: Model to use for summarization (defaults to self.model)
            iteration: Current iteration number
            bert_score: Whether to calculate BERT scores
            suprisal_score: Whether to calculate surprisal scores
        Returns:
            Dictionary containing the title and processed site data with summaries and metrics
        """
        # Determine which model to use
        if summary_model is None:
            summary_model = self.model
        # Generate summaries based on model type
        if "pegasus" in summary_model:
            # Initialize Pegasus model
            model_name = "google/pegasus-large"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(
                device
            )
            # Generate summaries with Pegasus
            summaries = []
            for site in sites:
                content = site["content"]
                src_text = [content]

                batch = tokenizer(
                    src_text, truncation=True, padding="longest", return_tensors="pt"
                ).to(device)
                translated = model.generate(**batch)
                tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

                summary = {
                    "summary": tgt_text[0],
                    "tokens": len(enc.encode(tgt_text[0])),
                }
                summaries.append(summary)

        elif "brio" in summary_model:
            # Initialize BRIO summarizer
            summarizer = pipeline("summarization", model=summary_model)
            # Generate summaries with BRIO
            summaries = []
            for site in sites:
                content = site["content"]
                result = summarizer(content, do_sample=False)
                summary = {
                    "summary": result[0]["summary_text"],
                    "tokens": len(enc.encode(result[0]["summary_text"])),
                }
                summaries.append(summary)

        else:
            batch = []
            for site in sites:
                site_title = site["file_name"]
                content = site["content"]
                batch.append(
                    dspy.Example(
                        passage=content, id=site_title, iteration=iteration
                    ).with_inputs("passage", "id", "iteration")
                )
            summaries = self._get_summaries(batch)

        # Combine site data with summaries
        processed_sites = []
        for i, site in enumerate(sites):
            site_data = {
                "file_name": site["file_name"],
                "content": site["content"],
                "url": site["url"],
                "iteration": iteration,
            }
            if summaries:
                site_data.update(
                    {
                        "summary": summaries[i]["summary"],
                        "tokens": summaries[i]["tokens"],
                    }
                )
            else:
                site_data["summary"] = ""
                site_data["tokens"] = 0

            processed_sites.append(site_data)

        return {
            "title": title,
            "sites": processed_sites,
        }

    def _generate_k_qa_pairs(
        self,
        passage: str,
        model: str,
        existing_questions: list = None,
        k: int = 10,
        file_name: str = None,
    ) -> list:
        questions = set(existing_questions or [])
        data = []
        max_attempts = 10
        for _ in range(max_attempts):
            # Generate more QA pairs
            messages = synthetic_qa_prompt(
                passage=passage, existing_questions=questions, k=k
            )
            response = completion(model=model, messages=messages, **self.kwargs)
            response = response.choices[0].message.content
            qa_pairs = extract_qa_pairs(response, file_name=file_name)
            unique_pairs = []
            for pair in qa_pairs:
                question = pair["question"]
                # Check if this is a new question
                if question not in questions:
                    questions.add(question)
                    unique_pairs.append(pair)

            data.extend(unique_pairs)

            # If we've collected enough unique questions, we're done
            if len(questions) >= k:
                break
        return data

    def generate_qa_pairs(
        self, sites: list[dict], num_questions: int, model: str = None
    ) -> list[dict]:
        if model is None:
            model = self.model
        data = []
        for site in tqdm(sites):
            existing_questions = site.get("questions", None)
            passage = site["content"]
            qa_pairs = self._generate_k_qa_pairs(
                passage=passage,
                existing_questions=existing_questions,
                model=model,
                k=num_questions,
                file_name=site["file_name"],
            )
            data.append(qa_pairs)
        # Remove duplicates questions and merge answers
        return data

    def evaluate_summaries(self, data: tuple, model: str = None) -> pd.DataFrame:
        # data -> (site_data, qa_pairs)
        pass

    def _get_refined_summaries(
        self,
        data: list[dspy.Example],
        model: str,
        bert_score: bool = False,
        suprisal_score: bool = False,
    ) -> pd.DataFrame:
        program = RefineSummary()
        program.set_lm(dspy.LM(model, max_tokens=self.max_tokens, **self.kwargs))

        output = dspy.Evaluate(
            devset=data,
            metric=lambda x, y: True,
            num_threads=self.num_threads,
            display_progress=False,
            return_outputs=True,
        )(program)

        processed_sites = [dict(output_pair[1]) for output_pair in output[1]]

        # Calculate BERT scores if requested
        if bert_score:
            summaries_text = [site["summary"] for site in processed_sites]
            original_content = [
                prepare_content(site["content"]) for site in processed_sites
            ]
            bert_scores = calculate_bert_score(summaries_text, original_content)
            # Add scores to respective sites
            for site, score in zip(processed_sites, bert_scores["scores"]):
                site["bert_score"] = score

        # Calculate surprisal scores if requested
        if suprisal_score:
            for site in processed_sites:
                surprisal_results = calculate_surprisal_score(
                    summary=site["summary"],
                    model=self.suprisal_model,
                    tokenizer=self.suprisal_tokenizer,
                    device=self.device,
                    verbose=False,
                )
                # Update site with all surprisal metrics
                site.update(surprisal_results)
        return processed_sites

    def eval_questions(
        self, sites: list, qa_pairs: list, qna: dspy.Module, iteration: int = 0
    ) -> pd.DataFrame:
        answers = []
        for i, site in enumerate(sites):
            for qa in qa_pairs[i]:
                # Create example directly from the qa dictionary and site data
                example = dspy.Example(
                    question=qa["question"],
                    summary=site["summary"],
                    content=site["content"],
                    true=qa["answer"],
                    type="summary",
                    file_name=site["file_name"],
                    iteration=iteration,
                ).with_inputs(
                    "question",
                    "true",
                    "content",
                    "type",
                    "file_name",
                    "summary",
                    "iteration",
                )
                answers.append(example)
        output = dspy.Evaluate(
            devset=answers,
            metric=lambda x, y: True,  # noqa: E731
            num_threads=self.num_threads,
            display_progress=False,
            return_outputs=True,
        )(qna)
        qa_df = pd.DataFrame([dict(output_pair[1]) for output_pair in output[1]])

        qa_df["correct"] = qa_df.apply(check_answer_em, axis=1)
        qa_df["correct_f1"] = qa_df.apply(check_answer_f1, axis=1)
        # drop content column
        qa_df.drop(columns=["content"], inplace=True)
        return qa_df

    def improve_llms_txt(
        self,
        data: dict,
        train_questions: list[dict],
        eval_questions: list[dict] = None,
        model: str = None,
        iterations: int = 10,
    ) -> dict:
        """
        Updates the provided llms.txt with iterative summarization.
        Returns updated llms.txt data object.
        """
        if eval_questions is None:
            eval_questions = []
        if model is None:
            model = self.model

        index_iterations = [0] * (iterations + 1)
        index_iterations[0] = data["sites"]

        qna = QnA()
        qna.set_lm(dspy.LM(model, max_tokens=self.max_tokens, **self.kwargs))

        train_qa_df = self.eval_questions(index_iterations[0], train_questions, qna)
        eval_qa_df = self.eval_questions(index_iterations[0], eval_questions, qna)

        for iteration in tqdm(range(1, iterations + 1)):
            to_be_refined = []
            last_iteration = index_iterations[iteration - 1]
            for _, site in enumerate(last_iteration):
                file_name = site["file_name"]

                # find questions that are not answered correctly
                unanswered = train_qa_df[
                    (train_qa_df["file_name"] == file_name)
                    & (train_qa_df["correct"] == 0)
                    & (train_qa_df["iteration"] == iteration - 1)
                    # & (train_qa_df["correct_f1"] > 0.5)
                ].copy()

                # for site build a dspy refiner prompt with X sample selected questions
                sample_size = min(5, len(unanswered))  # TODO rewrite as hyperparam
                unanswered_questions = (
                    []
                    if unanswered.empty
                    else unanswered["question"].sample(sample_size)
                )
                to_be_refined.append(
                    dspy.Example(
                        summary=site["summary"],
                        passage=site["content"],
                        questions=unanswered_questions,
                        iteration=iteration,
                        file_name=file_name,
                    ).with_inputs(
                        "summary", "passage", "questions", "iteration", "file_name"
                    )
                )
            index_iterations[iteration] = self._get_refined_summaries(
                to_be_refined, model
            )
            # train eval
            # --------------------------------
            iteration_train_qa_df = self.eval_questions(
                index_iterations[iteration], train_questions, qna, iteration=iteration
            )
            train_qa_df = pd.concat(
                [train_qa_df, iteration_train_qa_df], ignore_index=True
            )
            training_loss = iteration_train_qa_df["correct"].mean()
            training_loss_f1 = iteration_train_qa_df["correct_f1"].mean()
            # eval eval
            # --------------------------------
            iteration_eval_qa_df = self.eval_questions(
                index_iterations[iteration], eval_questions, qna, iteration=iteration
            )
            eval_qa_df = pd.concat(
                [eval_qa_df, iteration_eval_qa_df], ignore_index=True
            )
            eval_loss = iteration_eval_qa_df["correct"].mean()
            eval_loss_f1 = iteration_eval_qa_df["correct_f1"].mean()

            # Update tqdm with the loss values
            tqdm.write(
                f"Iteration {iteration}: Train Loss = {training_loss:.4f}, Train F1 = {training_loss_f1:.4f}, Eval Loss = {eval_loss:.4f}, Eval F1 = {eval_loss_f1:.4f}"
            )

        # TODO fix to return only best or with metadata
        # TODO return latest best iteration
        return index_iterations[-1], index_iterations, train_qa_df, eval_qa_df
