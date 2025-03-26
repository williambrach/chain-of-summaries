import re

import dspy


class QnASignature(dspy.Signature):
    """
    Given a question and content answer question simple as possible.
    Answers are mostly words and phrases. Don't asnwer as full sentences.
    """

    question = dspy.InputField(desc="a question to answer")
    content = dspy.InputField(desc="content that contains the answer")
    answer: str = dspy.OutputField(desc="answer for question")


class QnA(dspy.Module):
    def __init__(self) -> None:
        self.answer = dspy.ChainOfThought(QnASignature)

    def forward(
        self,
        question: str,
        summary : str,
        content: str,
        true: str = None,
        type: str = None,
        file_name: str = None,
        iteration: int = 0,
    ) -> dspy.Example:
        response = self.answer(question=question, content=summary)
        return dspy.Example(
            question=question,
            summary=summary,
            content=content,
            true=true,
            pred=response.answer,
            type=type,
            file_name=file_name,
            iteration=iteration,
        )

def synthetic_qa_prompt(
    passage: str,
    existing_questions: list = None,
    k: int = 10,
) -> list:
    """
    Generate messages for creating synthetic question-answer pairs from a text passage.
    Reference: https://github.com/Azure/synthetic-qa-generation
    """
    # Create system message with context
    messages = [{"content": f"Here is content of the file:\n\n{passage}", "role": "system"}]

    # Add constraint for existing questions if provided
    # Add main instruction message
    messages.append(
        {
            "content": f"Generate {k} diverse and specific questions in Q: format based on the content. Do not include question numbers. Each question should target important information from the text. Each answer should be concise (word or short phrase) and directly address the question.",
            "role": "user",
        }
    )
    if existing_questions:
        messages.append(
            {
                "content": f"The following questions already exist and must not be duplicated: {', '.join(existing_questions)}.",
                "role": "system",
            }
        )
    # Add format instruction message
    messages.append(
        {
            "content": "Format should be:\nQ: <question>\nA: <answer>\n\nEnsure answers are brief (a word or short phrase, not a full sentence) and factually accurate based on the text.",
            "role": "user",
        }
    )

    return messages


def extract_qa_pairs(response : str, file_name : str = None) -> list:
    questions = [q.strip() for q in re.findall(r"Q:(.*?)$", response, re.MULTILINE)]
    answers = [q.strip() for q in re.findall(r"A:(.*?)$", response, re.MULTILINE)]
    qa_pairs = []
    for q, a in zip(questions, answers):
        r = {"question": q, "answer": a}
        if file_name is not None:
            r["file_name"] = file_name
        qa_pairs.append(r)
    return qa_pairs
