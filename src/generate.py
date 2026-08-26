import os
import yaml
from typing import List, Dict, Any
from retrieve import Retriever


PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question ONLY using the provided context chunks below.
Cite chunk sources using [chunk_id].
If the context does not contain sufficient information to answer, state "I don't know based on the provided context."

CONTEXT CHUNKS:
{context_str}

QUESTION: {query}

ANSWER:"""


class Generator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        db_dir = self.config["paths"]["vector_db_dir"]
        model_name = self.config["embedding"]["model_name"]
        self.top_k = self.config["retrieval"]["top_k"]
        self.retriever = Retriever(model_name=model_name, db_dir=db_dir)

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer grounded in retrieved context chunks."""
        if not context_chunks:
            return {
                "answer": "I don't know based on the provided context.",
                "citations": []
            }

        # Build context string
        context_lines = []
        citations = []
        for chunk in context_chunks:
            cid = chunk["chunk_id"]
            text = chunk["text"]
            source = chunk["metadata"].get("source_file", "unknown")
            page = chunk["metadata"].get("page_number", 1)
            context_lines.append(f"[{cid}] (Source: {source}, Page: {page}): {text}")
            citations.append({"chunk_id": cid, "source_file": source, "page_number": page})

        context_str = "\n\n".join(context_lines)
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query=query)

        # Call LLM or standard mock answer engine
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                model_name = self.config["generation"].get("model_name", "gemini-3.5-flash-lite")
                if model_name in ("mock-llm", ""):
                    model_name = "gemini-3.5-flash-lite"
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                answer = (response.text or "").strip()
            except Exception as e:
                answer = f"Error calling Gemini API: {str(e)}"
        elif openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error calling OpenAI API: {str(e)}"
        else:
            # Deterministic mock response for grounded testing without external API key
            top_chunk = context_chunks[0]
            answer = f"Based on context [{top_chunk['chunk_id']}], {top_chunk['text'][:150]}..."

        return {
            "answer": answer,
            "citations": citations,
            "prompt_used": prompt
        }

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Pipeline entry point: retrieval -> generation."""
        retrieved_chunks = self.retriever.retrieve(query, k=self.top_k)
        result = self.generate_answer(query, retrieved_chunks)
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": result["answer"],
            "citations": result["citations"]
        }


if __name__ == "__main__":
    generator = Generator()
    sample_query = "What is the primary topic of the ingested files?"
    res = generator.answer_question(sample_query)
    print(f"Query: {res['query']}")
    print(f"Answer: {res['answer']}")
    print(f"Citations: {res['citations']}")
