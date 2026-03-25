try:
    from .chunking import simple_chunk_documents
    from .data_loading import read_repo_data
    from .search import FAQHybridSearch, filter_documents_by_filename
    from .agents.agent import CourseFAQAgent
except ImportError:
    # Allow running this file directly: `python main.py`
    from chunking import simple_chunk_documents
    from data_loading import read_repo_data
    from search import FAQHybridSearch, filter_documents_by_filename
    from agents.agent import CourseFAQAgent


DEFAULT_TRACK = "data-engineering"
DEFAULT_LIMIT = 5


dtc_faq = read_repo_data('DataTalksClub', 'faq')
dtc_faq_track = filter_documents_by_filename(dtc_faq, DEFAULT_TRACK)
dtc_faq_chunks = simple_chunk_documents(dtc_faq_track, content_key="content")

faq_hybrid_search = FAQHybridSearch.from_documents(
    dtc_faq,
    track=DEFAULT_TRACK,
    show_progress=True,
)

course_agent = CourseFAQAgent(search_fn=faq_hybrid_search.lexical)

question = "I just discovered the course, can I join now?"
result = course_agent.run(question)
print(course_agent.result_text(result))
print(course_agent.result_messages(result))




