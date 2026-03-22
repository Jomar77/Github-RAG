query = 'Can I join the course now?'

text_results = faq_index.search(query, num_results=5)

q = embedding_model.encode(query)
vector_results = faq_vindex.search(q, num_results=5)

final_results = text_results + vector_results
