import os

from openai import OpenAI
from tqdm.auto import tqdm

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm(prompt, model='gpt-4o-mini'):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = openai_client.responses.create(
        model=model,
        input=messages
    )

    return response.output_text


prompt_template = """
Split the provided document into logical sections
that make sense for a Q&A system.

Each section should be self-contained and cover
a specific topic or concept.

<DOCUMENT>
{document}
</DOCUMENT>

Use this format:

## Section Name

Section content with all relevant details

---

## Another Section Name

Another section content

---
""".strip()



def intelligent_chunking(text):
    prompt = prompt_template.format(document=text)
    response = llm(prompt)
    sections = response.split('---')
    sections = [s.strip() for s in sections if s.strip()]
    return sections


def _iter_with_progress(docs):
    return tqdm(docs)

def intelligent_chunk_documents(docs, content_key="content"):
    chunks_out = []

    for doc in _iter_with_progress(docs):
        doc_copy = doc.copy()
        doc_content = doc_copy.pop(content_key)

        sections = intelligent_chunking(doc_content)
        for section in sections:
            section_doc = doc_copy.copy()
            section_doc['section'] = section
            chunks_out.append(section_doc)

    return chunks_out
