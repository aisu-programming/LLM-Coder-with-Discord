##### Libraries #####
import torch
import logging
from libs.base import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig





##### Loggers #####
HF_LOGGER = logging.getLogger("Hugging Face")
HF_LOGGER.setLevel(logging.DEBUG)
HF_HANDLER = logging.StreamHandler()
HF_HANDLER.setLevel(logging.DEBUG)
HF_HANDLER.setFormatter(logging.Formatter(
    "\n[%(levelname)s] (%(name)s) %(asctime)s | %(filename)s: %(funcName)s: %(lineno)03d | %(message)s",
    datefmt="%m-%d %H:%M:%S"
))
HF_LOGGER.addHandler(HF_HANDLER)





##### Classes #####
class Zephyr7bBeta(BaseModel):
    def __init__(
            self,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            load_in_4bit: bool = False,
            load_in_8bit: bool = True
        ) -> None:
        super().__init__()

        assert not (load_in_4bit and load_in_8bit), \
            "Parameters load_in_4bit and load_in_8bit cannot both be True."
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        self.model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",
            device_map=device,
            quantization_config=BitsAndBytesConfig(load_in_4bit=load_in_4bit,
                                                   load_in_8bit=load_in_8bit),
        )
        HF_LOGGER.info(f"Model \"HuggingFaceH4/zephyr-7b-beta\" successfully loaded!")
        self.eos_token_id = \
            self.tokenizer.encode(self.stopping_sign, add_special_tokens=False)[-1]
    
    def generate_response(self, msg: str) -> str:
        HF_LOGGER.debug("msg:\n\n", msg)
        msg_pt: torch.Tensor = \
            self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(
            msg_pt,
            max_new_tokens=500,
            eos_token_id=self.eos_token_id,
            repetition_penalty=1.2
        )
        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        HF_LOGGER.debug("Generated response:\n", response)
        return response



class DeepseekCoder33bInstruct():
    def __init__(
            self,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            load_in_4bit: bool = True,
            load_in_8bit: bool = False,
        ) -> None:
        super().__init__()

        assert not (load_in_4bit and load_in_8bit), \
            "Parameters load_in_4bit and load_in_8bit cannot both be True."
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-33b-instruct",
            device_map=device,
            quantization_config=BitsAndBytesConfig(load_in_4bit=load_in_4bit,
                                                   load_in_8bit=load_in_8bit),
            trust_remote_code=True,
        )

    def __call__(self, msg: str) -> str:
        msg = [
            { "role": "system", "content": "You are a coding agent who always response to any questions." + \
                                           "If the user ask you to generate codes, reply only with codes." + \
                                           "Do not reply any other sentences." },
            { "role": "user",   "content": msg }
        ]
        inputs = self.tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False,
                                      # top_p=0.95,
                                      top_k=50, num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][len(inputs[0]):],
                                     skip_special_tokens=True)