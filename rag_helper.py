from ingest import load_faq_data, build_index

documents = load_faq_data()
index = build_index(documents)

INSTRUCTIONS = """
    Your task is to answer questions from the course participants
    based on the provided context.

    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = '''
        Question:
        {question}

        Context:
        {context}
    '''

def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict, 
        num_results=5
    )

def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()

def build_prompt(question_local, search_result):
    context_local = build_context(search_result)
    USER_PROMPT_TEMPLATE = f"""
        Question:
        {question_local}

        Context:
        {context_local}
    """
    
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question_local,
        context_local=context_local
    )
    return prompt.strip()