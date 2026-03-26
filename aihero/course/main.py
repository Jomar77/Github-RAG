from dotenv import load_dotenv

# Load .env before importing modules that initialize OpenAI clients.
load_dotenv()

try:
    from .chunking import simple_chunk_documents
    from .data_loading import read_repo_data
    from .search import FAQHybridSearch, filter_documents_by_filename
    from .agents.agent import build_course_agent, result_text
    from .evals.evals import (
        build_eval_agent,
        evaluate_log_record_sync,
        load_log_file,
        log_interaction_to_file,
        collect_ai_generated_v2_logs,
        evaluate_log_set,
        eval_results_to_dataframe,
        eval_dataframe_stats
    )
except ImportError:
    # Allow running this file directly: `python main.py`
    from chunking import simple_chunk_documents
    from data_loading import read_repo_data
    from search import FAQHybridSearch, filter_documents_by_filename
    from agents.agent import build_course_agent, result_text
    from evals.evals import (
        build_eval_agent,
        evaluate_log_record_sync,
        load_log_file,
        log_interaction_to_file,
        collect_ai_generated_v2_logs,
        evaluate_log_set,
        eval_results_to_dataframe,
        eval_dataframe_stats
    )


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

course_agent = build_course_agent(search_fn=faq_hybrid_search.lexical)

question = "can I join late and get a certificate?"
result = course_agent.run_sync(question)
print(result_text(result))

if hasattr(result, "new_messages"):
    messages = result.new_messages()
elif hasattr(result, "all_messages"):
    messages = result.all_messages()
else:
    messages = []

log_path = log_interaction_to_file(course_agent, messages, source="user")
print(f"Logged interaction to: {log_path}")

log_record = load_log_file(log_path)
eval_agent = build_eval_agent()
checklist = evaluate_log_record_sync(log_record, eval_agent=eval_agent)

print("Evaluation summary:", checklist.summary)
for check in checklist.checklist:
    print(f"- {check.check_name}: {check.check_pass} ({check.justification})")

import asyncio

async def run_batch_eval():
    eval_agent = build_eval_agent()
    eval_set = collect_ai_generated_v2_logs()
    
    if not eval_set:
        print("No logs found for evaluation. Please ensure that AI-generated interactions have been logged.")
        return
    
    eval_results = await evaluate_log_set(eval_agent, eval_set)
    df = eval_results_to_dataframe(eval_results)
    stats = eval_dataframe_stats(df)
    print(f"Evaluated records: {len(df)}")
    print(df.head())
    print("Evaluation Stats:")
    print(stats)
    
asyncio.run(run_batch_eval())




