# -*- coding: utf-8 -*-

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
!pip install pypdf
!pip install -q pyngrok Flask flask-cors
!pip install -q llama-index==0.9.42
!pip install -q langchain==0.0.353
!pip install -q InstructorEmbedding==1.0.1 sentence-transformers==2.2.2

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
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from llama_index.readers import SimpleDirectoryReader
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from langchain.embeddings import HuggingFaceInstructEmbeddings
from flask import Flask, request, jsonify, Response,stream_with_context
from pyngrok import ngrok
from flask_cors import CORS

embeddings = HuggingFaceInstructEmbeddings(
        model_name='hkunlp/instructor-large',
        encode_kwargs={'normalize_embeddings': True},
        query_instruction="Represent the query for retrieval: "
    )

model_name_or_path = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
model_basename = "capybarahermes-2.5-mistral-7b.Q4_0.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>" # Prompt Template of the model

BOS, EOS = "", ""
B_INST, E_INST = "<|im_start|>\n", "<|im_end|>\n<|im_start|>assistant"
B_SYS, E_SYS = "<|im_start|>\n", "\n<|im_end|>\n\n"

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

!wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "llama2.pdf"

!mkdir data

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embeddings,
)

class ChunkProcessor:
    def __init__(self, file_path:str, service_context=None):
        self.file_path = file_path
        self.service_context = service_context
        self.sub_chunk_sizes = [256, 512]
        self.sub_node_parsers = [
            SimpleNodeParser.from_defaults(chunk_size=c) for c in self.sub_chunk_sizes
        ]
        self.all_nodes = []
        self.load_format_docs()
        self.create_retriever()

    def load_format_docs(self):
        documents_0 = SimpleDirectoryReader(self.file_path).load_data()
        doc_text = "\n\n".join([d.get_content() for d in documents_0])
        docs = [Document(text=doc_text)]

        for base_node in SimpleNodeParser.from_defaults(chunk_size=1024).get_nodes_from_documents(docs):
            for n in self.sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                self.all_nodes.extend(sub_inodes)

            # also add original node to node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            self.all_nodes.append(original_node)

    def create_retriever(self):
        all_nodes_dict = {n.node_id: n for n in self.all_nodes}
        vector_index_chunk = VectorStoreIndex(
            self.all_nodes, service_context=self.service_context
        )
        vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=2)
        self.retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever_chunk},
            node_dict=all_nodes_dict,
            verbose=True,
        )

    def query_prompt(self, prompt):
        query_engine_chunk = RetrieverQueryEngine.from_args(
            self.retriever, service_context=self.service_context
        )
        response = query_engine_chunk.query(prompt)
        return str(response)

    def process_chunks_and_query(self, prompt):
        response = self.query_prompt(prompt)
        return response

chunk_processor = ChunkProcessor(file_path="./data", service_context=service_context)

!curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok

!ngrok authtoken "Your-Ngrok-API"



app = Flask(__name__)
CORS(app)
@app.route("/index")
def index():
    return "Hello"

@app.route('/generate', methods=['POST'])
def generate():
    inp = request.get_json().get("prompt")
    print(f"======Input:{inp}=======")
    response = chunk_processor.process_chunks_and_query(prompt=inp)

    return jsonify({'generated_text': str(response)})

if __name__ == '__main__':
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)
    app.run()

