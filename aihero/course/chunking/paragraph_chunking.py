from __future__ import annotations


import re

def split_markdown_by_level(text, level=2):
    """
    Split markdown text by a specific header level.
    
    :param text: Markdown text as a string
    :param level: Header level to split on
    :return: List of sections as strings
    """
    # This regex matches markdown headers
    # For level 2, it matches lines starting with "## "
    header_pattern = r'^(#{' + str(level) + r'} )(.+)$'
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)
    
    sections = []
    for i in range(1, len(parts), 3):
        # We step by 3 because regex.split() with
        # capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i+1]  # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i+2 < len(parts):
            content = parts[i+2].strip()

        if content:
            section = f'{header}\n\n{content}'
        else:
            section = header
        sections.append(section)
    
    return sections

def paragraph_chunk_documents(docs, content_key="content", size=2000, step=1000):
    chunks_out = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop(content_key)
        sections = split_markdown_by_level(doc_content, level=2)
        for section in sections:
            section_doc = doc_copy.copy()
            section_doc['section'] = section
            chunks_out.append(section_doc)

    return chunks_out

