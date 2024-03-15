##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
import logging
from typing import List
from libs.base import BaseModel
from transformers import AutoTokenizer
from langchain_community.llms.vllm import VLLMOpenAI





##### Loggers #####
LC_LOGGER = logging.getLogger("LangChain")
LC_LOGGER.setLevel(logging.DEBUG)
LC_HANDLER = logging.StreamHandler()
LC_HANDLER.setLevel(logging.DEBUG)
LC_HANDLER.setFormatter(logging.Formatter('\n'+os.environ["LOG_FMT"], datefmt=os.environ["LOG_DATE_FMT"]))
LC_LOGGER.addHandler(LC_HANDLER)





##### Classes #####
class VllmDockerModel(BaseModel):
    def __init__(self, model_name: str, max_tokens: int, port: int) -> None:
        super().__init__()
                
        class MyVLLMOpenAI(VLLMOpenAI):

            @property
            def max_context_size(self) -> int:
                """Get max context size for this model."""
                return max_tokens
            
            def get_token_ids(self, text: str) -> List[int]:
                """Get the token IDs using AutoTokenizer."""
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                token_ids = tokenizer.encode(text)
                LC_LOGGER.debug(f"The token length of the input text is {len(token_ids)}.")
                return token_ids

        self.model = MyVLLMOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=-1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=1,
            best_of=1,
            model_kwargs={"stop": [ self.stopping_sign ]},
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{port}/v1",
            batch_size=20,
            timeout=None,  # float | Tuple[float, float] | Any | None
            max_retries=2,
            streaming=False,
            allowed_special=set(),     # AbstractSet[str] | Literal['all']
            disallowed_special="all",  #  Collection[str] | Literal['all']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_response(self, message: str) -> str:
        response = self.model.invoke(message)
        res_token_len = len(self.tokenizer.encode(response))
        LC_LOGGER.debug(
            f"The token length of the response text is {res_token_len}.")
        return response