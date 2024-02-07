# -*- coding: utf-8 -*-
"""
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
!pip install pypdf
!pip install -q pyngrok Flask flask-cors
!pip install -q llama-index==0.9.42
!pip install -q langchain==0.0.353
!pip install -q InstructorEmbedding==1.0.1 sentence-transformers==2.2.2
!pip install rank-bm25 pymupdf
!mkdir data
!wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
"""

import nest_asyncio
from huggingface_hub import hf_hub_download
from typing import List, Optional, Sequence
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import PromptTemplate
from tqdm.asyncio import tqdm
from llama_index.retrievers import BM25Retriever
from llama_index.schema import NodeWithScore
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List
from llama_index.schema import NodeWithScore
from llama_index.response.notebook_utils import display_source_node
from llama_index.query_engine import RetrieverQueryEngine


nest_asyncio.apply()

model_name_or_path = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
model_basename = "capybarahermes-2.5-mistral-7b.Q4_0.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>" # Prompt Template of the model

BOS, EOS = "", ""
B_INST, E_INST = "<|im_start|>user\n\n", "<|im_end|>\n<|im_start|>assistant"
B_SYS, E_SYS = "<|im_start|>system\n\n", "\n<|im_end|>\n\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

        if i == 0:
            # make sure system prompt is included at the start
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            # end previous user-assistant interaction
            string_messages[-1] += f" {EOS}"
            # no need to include system prompt
            str_message = f"{BOS} {B_INST} "

        # include user message content
        str_message += f"{user_message.content} {E_INST}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS} {B_INST} {B_SYS} {system_prompt_str.strip()} {E_SYS} "
        f"{completion.strip()} {E_INST}"
    )


llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 60},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)


documents = SimpleDirectoryReader("./data").load_data()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model, chunk_size=1024
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)


query_str = "How do the models developed in this work compare to open-source chat models based on the benchmarks tested?"

query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)


def generate_queries(llm, query_str: str, num_queries: int = 4):
    fmt_prompt = query_gen_prompt.format(num_queries=num_queries - 1, query=query_str)
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries


queries = generate_queries(llm, query_str, num_queries=4)
queries = [item for item in queries if item != ""]


async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict


## vector retriever
vector_retriever = index.as_retriever(similarity_top_k=2)

## bm25 retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

results_dict = await run_queries(queries, [vector_retriever, bm25_retriever])


def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]


final_results = fuse_results(results_dict)


for n in final_results:
    display_source_node(n, source_length=500)


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(llm, query_str, num_queries=4)
        results = run_queries(queries, [vector_retriever, bm25_retriever])
        final_results = fuse_results(
            results_dict, similarity_top_k=self._similarity_top_k
        )

        return final_results


fusion_retriever = FusionRetriever(
    llm, [vector_retriever, bm25_retriever], similarity_top_k=2
)

# query_engine = RetrieverQueryEngine(fusion_retriever)

query_engine_base = RetrieverQueryEngine.from_args(
    fusion_retriever, service_context=service_context
)

response = query_engine_base.query(query_str)

print(str(response))