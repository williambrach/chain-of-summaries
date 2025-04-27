import time

import dspy
import pandas as pd
import tiktoken
import torch
from litellm import batch_completion, completion
from tqdm import tqdm
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    pipeline,
)

from chain_of_summaries.metrics import (
    check_answer_em,
    check_answer_f1,
)
from chain_of_summaries.qna import (
    answer_prompt,
    extract_qa_pairs,
    synthetic_qa_prompt,
)
from chain_of_summaries.summary import Summarize, enc, refine_summary_prompt
from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration

def get_token_count(text: str, enc: object = None) -> int:
    if enc is None:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
    """Get the number of tokens in a string."""
    return len(enc.encode(text))


def truncate_text(text: str, max_tokens: int = 100_000, enc: object = None) -> str:
    if enc is None:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
    """Truncate a string to a maximum number of tokens."""
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)


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
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.cache = kwargs.get("cache", True)
        self.num_threads = kwargs.get("num_threads", 20)
        self.batch_size = batch_size
        if "num_threads" in kwargs:
            del kwargs["num_threads"]
        if "cache" in kwargs:
            del kwargs["cache"]
        self.display_progress = kwargs.get("display_progress", True)
        self.device = kwargs.get("device", "cpu")
        try:
            self.test_completion_call()
            print(f"LLMSProcessor initialized with model: {self.model}")
        except Exception as e:
            print(f"Error initializing LLMSProcessor: {e}")
            raise

    def __repr__(self) -> str:
        """Return a string representation of the processor."""
        return f"LLMSProcessor(model='{self.model}')"

    def test_completion_call(self) -> str:
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        response = completion(model=self.model, messages=messages, **self.kwargs)
        return response

    def completion_call(self, messages: list, model: str = None) -> str:
        if model is None:
            model = self.model
        response = completion(
            model=model,
            messages=messages,
            temperature=0,
            **self.kwargs,
        )
        return response

    def batch_completion_call(self, messages: list, model: str = None) -> str:
        batch_size = self.batch_size
        if model is None:
            model = self.model
        # Split messages into batches
        batches = [
            messages[i : i + batch_size] for i in range(0, len(messages), batch_size)
        ]
        responses = []
        for batch in tqdm(batches, total=len(batches)):
            response = batch_completion(
                model=model,
                messages=batch,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.kwargs,
            )
            # check if rate limit
            try:
                _ = response[0].choices[0].message.content
            except Exception as e:
                print(f"Error in batch completion: {e}")
                sleep_time = 60
                time.sleep(sleep_time)
                response = batch_completion(
                    model=model,
                    messages=batch,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **self.kwargs,
                )
            responses.extend(response)
        return responses

    def run_qna(
        self,
        question: str,
        summary: str,
        model: str,
        content: str = None,
        true: str = None,
        type: str = "summary",
        file_name: str = None,
        iteration: int = 0,
    ) -> str:
        prompt = answer_prompt(question=question, content=summary, idk=False)
        try:
            response = self.completion_call(prompt, model=model)
            return {
                "question": question,
                "summary": summary,
                "content": content,
                "true": true,
                "pred": response.choices[0].message.content,
                "type": type,
                "file_name": file_name,
                "iteration": iteration,
            }
        except Exception as e:
            print(prompt)
            print(f"Error in QnA: {e}")
            return {
                "question": question,
                "summary": summary,
                "content": content,
                "true": true,
                "pred": "I don't know",
                "type": type,
                "file_name": file_name,
                "iteration": iteration,
            }

    def _get_summaries(self, data: list[dspy.Example]) -> list:
        # TODO REWRITE IN PLAIN PROMPT
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
            print("Using Pegasus for summarization...")
            model_name = "google/pegasus-large"
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(
                device
            )
            # Generate summaries with Pegasus
            summaries = []
            for site in tqdm(sites):
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
            print("Using Brio for summarization...")
            
            model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
            tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
            max_length = 1024
            summaries = []
            for site in tqdm(sites):
                content = site["content"]
                article = sites[0]['content']
                inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"])
                result= tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                summary = {
                    "summary":result,
                    "tokens": len(enc.encode(result)),
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

    def evaluate_summaries(
        self, data: tuple, questions: list[dict], model: str = None, iteration: int = 0
    ) -> pd.DataFrame:
        """
        Evaluate the summaries using the provided model and questions.
        """
        if model is None:
            model = self.model
        index = data["sites"] if "sites" in data else data

        answers = []
        if self.batch_size == 1:
            for site, qa_pairs in tqdm(zip(index, questions), total=len(index)):
                for qa in qa_pairs:
                    response = self.run_qna(
                        question=qa["question"],
                        summary=site["summary"],
                        content=site["content"],
                        true=qa["answer"],
                        file_name=site["file_name"],
                        iteration=iteration,
                        model=model,
                    )
                    answers.append(response)
        else:
            input_data = [
                {
                    "question": qa["question"],
                    "summary": site["summary"],
                    "content": site["content"],
                    "true": qa["answer"],
                    "file_name": site["file_name"],
                    "iteration": iteration,
                    "model": model,
                }
                for site, qa_pairs in zip(index, questions)
                for qa in qa_pairs
            ]

            # Create all messages at once
            messages = [
                answer_prompt(
                    question=input["question"], content=input["summary"], idk=False
                )
                for input in input_data
            ]

            # Make the batch call
            results = self.batch_completion_call(messages, model=model)

            # Create and return answers
            answers = [
                {
                    "question": input_item["question"],
                    "summary": input_item["summary"],
                    "content": input_item["content"],
                    "true": input_item["true"],
                    "pred": response.choices[0].message.content,
                    "type": "summary",
                    "file_name": input_item["file_name"],
                    "iteration": input_item["iteration"],
                }
                for response, input_item in zip(results, input_data)
            ]
        qa_df = pd.DataFrame(answers)
        qa_df["correct"] = qa_df.apply(check_answer_em, axis=1)
        qa_df["correct_f1"] = qa_df.apply(check_answer_f1, axis=1)
        if "content" in qa_df.columns:
            qa_df.drop(columns=["content"], inplace=True)

        return qa_df

    def _get_refined_summaries(
        self,
        data: list[dspy.Example],
        model: str = None,
        cod: bool = False,
    ) -> pd.DataFrame:
        if model is None:
            model = self.model
        index = []
        if self.batch_size == 1:
            for example in tqdm(data, total=len(data), desc="Refining summaries"):
                example = example.toDict()
                summary = example["summary"]
                passage = example["passage"]
                questions = example["questions"]
                iteration = example["iteration"]
                file_name = example["file_name"]
                q_per_iteration = example["q_per_iteration"]
                prompt = refine_summary_prompt(
                    passage=passage,
                    existing_summary=summary,
                    questions=questions,
                    cod=cod,
                )
                response = self.completion_call(prompt, model=model)
                response = (
                    response.choices[0]
                    .message.content.split("Summary")[-1]
                    .strip(":")
                    .strip("*")
                    .strip()
                )
                index.append(
                    {
                        "summary": response,
                        "file_name": file_name,
                        "tokens": len(enc.encode(response)),
                        "iteration": iteration,
                        "content": passage,
                        "questions": questions,
                        "q_per_iteration": q_per_iteration,
                        "cod": cod,
                    }
                )
        else:
            batch_size = self.batch_size
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                prompts = []
                batch_data = []
                for example in batch:
                    example = example.toDict()
                    summary = example["summary"]
                    passage = example["passage"]
                    questions = example["questions"]
                    iteration = example["iteration"]
                    file_name = example["file_name"]
                    q_per_iteration = example["q_per_iteration"]
                    prompt = refine_summary_prompt(
                        passage=passage,
                        existing_summary=summary,
                        questions=questions,
                        cod=cod,
                    )
                    prompts.append(prompt)
                    batch_data.append(example)
                responses = self.batch_completion_call(prompts, model=model)
                for response, example in zip(responses, batch_data):
                    content = response.choices[0].message.content
                    summary = content.split("Summary")[-1].strip(":").strip("*").strip()
                    index.append(
                        {
                            "summary": summary,
                            "file_name": example["file_name"],
                            "tokens": len(enc.encode(summary)),
                            "iteration": example["iteration"],
                            "content": example["passage"],
                            "questions": example["questions"],
                            "q_per_iteration": example["q_per_iteration"],
                            "cod": cod,
                        }
                    )
        return index

    # def eval_questions(
    #     self, sites: list, qa_pairs: list, qna: dspy.Module, iteration: int = 0
    # ) -> pd.DataFrame:
    #     answers = []
    #     for i, site in enumerate(sites):
    #         for qa in qa_pairs[i]:
    #             # Create example directly from the qa dictionary and site data
    #             example = dspy.Example(
    #                 question=qa["question"],
    #                 summary=site["summary"],
    #                 content="",
    #                 true=qa["answer"],
    #                 type="summary",
    #                 file_name=site["file_name"],
    #                 iteration=iteration,
    #             ).with_inputs(
    #                 "question",
    #                 "true",
    #                 "content",
    #                 "type",
    #                 "file_name",
    #                 "summary",
    #                 "iteration",
    #             )
    #             answers.append(example)
    #     output = dspy.Evaluate(
    #         devset=answers,
    #         metric=lambda x, y: True,  # noqa: E731
    #         num_threads=self.num_threads,
    #         display_progress=True,
    #         return_outputs=True,
    #     )(qna)
    #     qa_df = pd.DataFrame([dict(output_pair[1]) for output_pair in output[1]])
    #     qa_df["correct"] = qa_df.apply(check_answer_em, axis=1)
    #     qa_df["correct_f1"] = qa_df.apply(check_answer_f1, axis=1)
    #     # drop content column
    #     if "content" in qa_df.columns:
    #         qa_df.drop(columns=["content"], inplace=True)
    #     return qa_df

    def improve_llms_txt(
        self,
        data: dict,
        train_questions: list[dict],
        eval_questions: list[dict] = None,
        model: str = None,
        iterations: int = 10,
        iteration_questions: int = 5,
        cod: bool = False,
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

        print("Evaluation of initial train data")
        train_qa_df = self.evaluate_summaries(
            index_iterations[0], train_questions, model=model
        )
        print("Evaluation of initial eval data")
        eval_qa_df = self.evaluate_summaries(
            index_iterations[0], eval_questions, model=model
        )

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
                ].copy()

                # for site build a dspy refiner prompt with X sample selected questions
                sample_size = min(iteration_questions, len(unanswered))
                unanswered_questions = (
                    []
                    if unanswered.empty
                    else unanswered["question"].sample(sample_size).tolist()
                )
                to_be_refined.append(
                    dspy.Example(
                        summary=site["summary"],
                        passage=site["content"],
                        questions=unanswered_questions,
                        iteration=iteration,
                        file_name=file_name,
                        q_per_iteration=len(unanswered_questions),
                    ).with_inputs(
                        "summary", "passage", "questions", "iteration", "file_name"
                    )
                )
            index_iterations[iteration] = self._get_refined_summaries(
                to_be_refined, model, cod=cod
            )
            # train eval
            # --------------------------------
            iteration_train_qa_df = self.evaluate_summaries(
                index_iterations[iteration],
                train_questions,
                model=model,
                iteration=iteration,
            )
            train_qa_df = pd.concat(
                [train_qa_df, iteration_train_qa_df], ignore_index=True
            )
            training_loss = iteration_train_qa_df["correct"].mean()
            training_loss_f1 = iteration_train_qa_df["correct_f1"].mean()
            # eval eval
            # --------------------------------
            iteration_eval_qa_df = self.evaluate_summaries(
                index_iterations[iteration],
                eval_questions,
                model=model,
                iteration=iteration,
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
