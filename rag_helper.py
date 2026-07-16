from ingest import load_faq_data, build_index
from google.genai import types

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

class RAGBase:
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS, 
                 prompt_template=USER_PROMPT_TEMPLATE, course="llm-zoomcamp",model="gemini-2.5-flash"):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, question, course="llm-zoomcamp"):
        boost_dict = {"question": 2.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict, 
            num_results=5
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, question_local, search_result):
        context_local = self.build_context(search_result)
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
    
    def llm(self, instructions, user_prompt, model):
        # message_history = [
        #     {"role": "developer", "content": instructions},
        #     {"role": "user", "content": user_prompt}
        # ]
        response = self.llm_client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=instructions
            )
        )
        
        return response.text
    
    def rag(self, query, model="gemini-2.5-flash"):
        search_results = self.search(query)
        prompt_local = self.build_prompt(query, search_results)
        answer = self.llm(INSTRUCTIONS, prompt_local, model)
        return answer