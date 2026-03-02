"""
EduPredict RAG Query
Query the Chroma vector database for relevant documents and generate answers.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain imports
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

# HuggingFace LLM
try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. LLM generation disabled.")

# Paths
CHROMA_DIR = Path(__file__).parent.parent.parent / "models" / "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default LLM (small, free, local)
DEFAULT_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small, fast, decent quality
# Alternative small models:
# "microsoft/DialoGPT-medium" - conversational
# "distilgpt2" - very small, fast but limited
# "gpt2" - original small model


def get_vectorstore():
    """Initialize Chroma vector store."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma DB not found at {CHROMA_DIR}. Run ingest.py first.")
    
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )


def get_llm(model_name: str = DEFAULT_LLM_MODEL, device: str = "cpu"):
    """
    Initialize HuggingFace LLM for text generation.
    
    Args:
        model_name: HuggingFace model to use
        device: 'cpu' or 'cuda' (if GPU available)
    
    Returns:
        HuggingFacePipeline LLM or None if not available
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: transformers not available")
        return None
    
    try:
        print(f"Loading LLM: {model_name}")
        
        # For small models, we can load directly
        # Use text-generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            device=-1 if device == "cpu" else 0,  # -1 for CPU
            truncation=True
        )
        
        llm = HuggingFacePipeline(pipeline=text_pipeline)
        print(f"  LLM loaded successfully")
        return llm
        
    except Exception as e:
        print(f"  Error loading LLM: {e}")
        print("  Falling back to simple answer generation")
        return None


def query_rag(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the RAG system for relevant documents.
    
    Args:
        question: The question to ask
        k: Number of documents to retrieve
    
    Returns:
        List of retrieved documents with metadata
    """
    vectorstore = get_vectorstore()
    
    # Retrieve similar documents
    docs = vectorstore.similarity_search(question, k=k)
    
    results = []
    for doc in docs:
        # Calculate a simple relevance score (1.0 is perfect match)
        # Chroma doesn't return scores directly, so we use rank
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", "Untitled"),
            "doc_type": doc.metadata.get("doc_type", "unknown")
        })
    
    return results


def format_context(docs: List[Dict]) -> str:
    """Format retrieved documents into context for the LLM."""
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "unknown")
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        
        # Truncate long content
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        context_parts.append(f"""Document {i}:
Source: {source}
Title: {title}
Content: {content}
---""")
    
    return "\n\n".join(context_parts)


def generate_answer(question: str, docs: List[Dict], use_llm: bool = True) -> Dict[str, Any]:
    """
    Generate an answer from retrieved documents.
    
    Args:
        question: The user's question
        docs: Retrieved documents
        use_llm: Whether to use LLM for generation (vs. simple extraction)
    
    Returns:
        Dict with answer, sources, and confidence
    """
    if not docs:
        return {
            "answer": "No relevant documents found to answer this question.",
            "sources": [],
            "confidence": 0.0,
            "method": "none"
        }
    
    if use_llm and TRANSFORMERS_AVAILABLE:
        try:
            return generate_llm_answer(question, docs)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            # Fall back to simple extraction
    
    return generate_simple_answer(question, docs)


def generate_llm_answer(question: str, docs: List[Dict]) -> Dict[str, Any]:
    """Generate answer using HuggingFace LLM."""
    llm = get_llm()
    
    if llm is None:
        return generate_simple_answer(question, docs)
    
    context = format_context(docs)
    
    # Create prompt
    prompt = f"""You are an expert advisor on AI education programs. Answer the following question based ONLY on the provided documents. Be concise and factual.

Documents:
{context}

Question: {question}

Answer:"""
    
    try:
        # Generate answer
        answer = llm.predict(prompt)
        
        # Clean up answer
        answer = answer.strip()
        
        # Extract sources
        sources = list(set(d.get("source", "unknown") for d in docs))
        
        # Calculate confidence based on number and diversity of sources
        confidence = min(0.95, 0.5 + (len(docs) * 0.1) + (len(sources) * 0.1))
        
        return {
            "answer": answer,
            "sources": sources,
            "documents_used": len(docs),
            "confidence": round(confidence, 2),
            "method": "llm"
        }
        
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return generate_simple_answer(question, docs)


def generate_simple_answer(question: str, docs: List[Dict]) -> Dict[str, Any]:
    """
    Generate a simple answer by extracting and combining relevant info.
    """
    # Extract key sentences
    all_content = "\n\n".join(d.get("content", "") for d in docs)
    
    # Simple extraction - find sentences containing keywords from question
    question_words = set(w.lower() for w in question.split() if len(w) > 3)
    
    sentences = all_content.replace("\n", ". ").split(". ")
    relevant_sentences = []
    
    for sent in sentences:
        sent_lower = sent.lower()
        # Check if sentence contains any question keywords
        matches = sum(1 for w in question_words if w in sent_lower)
        if matches >= 1 and len(sent) > 20:
            relevant_sentences.append(sent.strip())
    
    # Limit sentences
    relevant_sentences = relevant_sentences[:5]
    
    # Build answer
    if relevant_sentences:
        answer = "Based on the available data:\n\n" + "\n\n".join(f"• {s}" for s in relevant_sentences)
    else:
        # Fallback - use first doc summary
        first_doc = docs[0].get("content", "")[:500]
        answer = f"Based on available information: {first_doc}..."
    
    sources = list(set(d.get("source", "unknown") for d in docs))
    
    # Confidence based on match quality
    confidence = min(0.85, 0.4 + (len(docs) * 0.1))
    
    return {
        "answer": answer,
        "sources": sources,
        "documents_used": len(docs),
        "confidence": round(confidence, 2),
        "method": "extraction"
    }


def answer_question(question: str, top_k: int = 5, use_llm: bool = True) -> Dict[str, Any]:
    """
    Complete RAG pipeline: retrieve documents and generate answer.
    
    Args:
        question: User's question
        top_k: Number of documents to retrieve
        use_llm: Whether to use LLM for answer generation
    
    Returns:
        Complete response with answer and metadata
    """
    try:
        # Retrieve documents
        docs = query_rag(question, k=top_k)
        
        # Generate answer
        result = generate_answer(question, docs, use_llm=use_llm)
        
        # Add retrieved documents for reference
        result["retrieved_documents"] = [
            {
                "title": d.get("title", "Untitled"),
                "source": d.get("source", "unknown"),
                "doc_type": d.get("doc_type", "unknown"),
                "preview": d.get("content", "")[:200] + "..."
            }
            for d in docs[:3]  # Only include top 3 in response
        ]
        
        return result
        
    except FileNotFoundError as e:
        return {
            "error": str(e),
            "answer": "RAG database not initialized. Please run the data pipeline first.",
            "sources": [],
            "confidence": 0.0
        }
    except Exception as e:
        return {
            "error": str(e),
            "answer": "An error occurred while processing your question.",
            "sources": [],
            "confidence": 0.0
        }


def get_suggested_questions() -> List[str]:
    """Get suggested questions for the RAG system."""
    return [
        "What are current AI program enrollment trends?",
        "Which universities recently added AI programs?",
        "What are job market projections for AI roles?",
        "What factors predict program success?",
        "What is the average salary for AI engineers?",
        "How many students are enrolled in AI programs?",
        "What education level is needed for AI jobs?",
        "What is the job growth rate for data scientists?"
    ]


def format_rag_response(response: Dict) -> str:
    """Format RAG response for display."""
    output = []
    output.append("=" * 60)
    output.append("RAG Query Result")
    output.append("=" * 60)
    
    if "error" in response:
        output.append(f"\n⚠ Error: {response['error']}")
    
    output.append(f"\nAnswer:\n{response.get('answer', 'No answer generated')}")
    
    output.append(f"\nConfidence: {response.get('confidence', 0) * 100:.0f}%")
    output.append(f"Method: {response.get('method', 'unknown')}")
    output.append(f"Documents used: {response.get('documents_used', 0)}")
    
    sources = response.get('sources', [])
    if sources:
        output.append(f"\nSources: {', '.join(sources)}")
    
    docs = response.get('retrieved_documents', [])
    if docs:
        output.append("\nRetrieved Documents:")
        for i, doc in enumerate(docs, 1):
            output.append(f"  {i}. {doc.get('title', 'Untitled')} ({doc.get('source', 'unknown')})")
    
    output.append("=" * 60)
    return "\n".join(output)


if __name__ == "__main__":
    print("=" * 60)
    print("EduPredict RAG Query System")
    print("=" * 60)
    
    # Show available questions
    print("\nSuggested questions:")
    for i, q in enumerate(get_suggested_questions(), 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-" * 60)
    
    # Test with example questions
    test_questions = [
        "What are current AI program enrollment trends?",
        "What is the average salary for AI engineers?",
        "What education level is needed for AI jobs?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 40)
        
        try:
            result = answer_question(question, top_k=5, use_llm=False)  # Fast mode
            print(format_rag_response(result))
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("To use LLM generation (slower but better answers):")
    print("  result = answer_question(question, use_llm=True)")
    print("=" * 60)
