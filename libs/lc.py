##### Libraries #####
import os
import logging
from typing import List
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
class LcVllmDockerBaseModel(object):
    def __init__(self) -> None:
        self.stopping_sign = "User:"
        # self.SOU = "<|StartOfUser|>"  # Start Of User
        # self.EOU = "<|EndOfUser|>"    # End Of User
        # self.SOY = "<|StartOfYou|>"   # Start Of You
        # self.EOY = "<|EndOfYou|>"     # End Of You
    
    def apply_template(self, message):
        return f"User: {message}\nYou: "
    
    def __call__(self, message: str) -> str:
        # BM_LOGGER.info(f"message: {message}")
        msg_tpl = self.apply_template(message)
        # BM_LOGGER.info(f"msg_tpl:\n\n{msg_tpl}")
        response = self.generate_response(msg_tpl)
        # BM_LOGGER.info(f"Generated response:\n\n{response}")
        response = response.removeprefix(msg_tpl).removesuffix(self.stopping_sign)
        response = response.strip()
        # BM_LOGGER.info(f"Proccessed response:\n\n{response}")
        return response

    def generate_response(self, message: str) -> str:
        raise NotImplementedError



class VllmDockerModel(LcVllmDockerBaseModel):
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