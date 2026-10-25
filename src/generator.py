
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        self.model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )
        self.client = Groq(api_key=api_key)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        if not chunks:
            return {
                "answer": (
                    "I could not find this information "
                    "in the provided documents."
                ),
                "citations": [],
            }

        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[CHUNK {i}]\n"
                f"Source: {chunk.get('source', 'Unknown')}\n"
                f"Page: {chunk.get('page', 'Unknown')}\n"
                f"{chunk.get('text', '')}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an enterprise knowledge assistant.

Answer using ONLY the provided context.
Do not invent facts or citations.

Return only valid JSON in this format:
{{"answer": "your answer", "citations": [0]}}

The citations must be integer chunk indexes that directly
support the answer.

If the answer is unavailable, return:
{{"answer": "I could not find this information in the provided documents.", "citations": []}}

QUESTION:
{query}

CONTEXT:
{context}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer from document evidence only. "
                        "Return valid JSON without Markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty response.")

        result = json.loads(content)

        if not isinstance(result.get("answer"), str):
            raise ValueError("Model response is missing a valid answer.")
        if not isinstance(result.get("citations"), list):
            raise ValueError("Model response has invalid citations.")

        valid_citations = [
            index
            for index in result["citations"]
            if isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(chunks)
        ]

        return {
            "answer": result["answer"],
            "citations": valid_citations,
        }
