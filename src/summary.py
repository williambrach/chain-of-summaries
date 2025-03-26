import dspy
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")


class SummarizeSignature(dspy.Signature):
    """
    Given a passage, generate a short summary. Maximum 1-2 sentences.
    """

    passage: str = dspy.InputField(desc="a passage to summarize")
    summary: str = dspy.OutputField(
        desc="a concise summary of the passage in plain text as paragraph"
    )


class Summarize(dspy.Module):
    def __init__(self) -> None:
        self.summarize = dspy.Predict(SummarizeSignature)

    def forward(self, passage: str, id: str, iteration: int) -> dspy.Example:
        try:
            summary = self.summarize(passage=passage)
        except Exception:
            try:
                summary = self.summarize(passage=passage[:512])
            except Exception:
                return dspy.Example(
                    summary=passage[:128],
                    file_name=id,
                    tokens=len(enc.encode(passage[:128])),
                    iteration=iteration,
                )

        return dspy.Example(
            summary=summary.summary,
            file_name=id,
            tokens=len(enc.encode(summary.summary)),
            iteration=iteration,
        )


class RefineSummarySignature(dspy.Signature):
    """Refine an existing text summary to address specific user questions.

    Take a summary and user questions as input, then generates a focused
    version that answers those questions while preserving information that is already in summary.

    Key requirements:
    - Include information that directly answers the user's questions
    - Preserve relevant key points from the original summary
    - Return the original summary unchanged if it already contains all necessary information
    - Return the original summary if the questions are not relevant to the text

    """

    passage: str = dspy.InputField(
        desc="The knowledge base passage that the summary is based on"
    )

    existing_summary: str = dspy.InputField(
        desc="The current summary that needs to be refined"
    )

    questions: list[str] = dspy.InputField(
        desc="Frequantly asked questions that the summary should address"
    )

    summary = dspy.OutputField(
        desc="Updated summary addressing the questions while maintaining the also informational content of the original summary. Try to keep summary short and concise."
    )


class RefineSummary(dspy.Module):
    def __init__(self) -> None:
        self.summarize = dspy.ChainOfThought(RefineSummarySignature)

    def forward(
        self,
        summary: str,
        passage: str,
        questions: list,
        iteration: int,
        file_name: str,
    ) -> dspy.Example:
        if len(questions) == 0:
            return dspy.Example(
                summary=summary,
                file_name=file_name,
                tokens=len(enc.encode(summary)),
                iteration=iteration,
                content=passage,
                questions=[],
            )
        generated = self.summarize(
            passage=passage, questions=questions, existing_summary=summary
        )
        return dspy.Example(
            summary=generated.summary,
            file_name=file_name,
            content=passage,
            tokens=len(enc.encode(generated.summary)),
            iteration=iteration,
            questions=questions,
        )
