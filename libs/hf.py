##### Libraries #####
import torch
import logging
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
class HfBaseModel(object):
    def __init__(self) -> None:
        pass



class Zephyr7bBeta(HfBaseModel):
    def __init__(
            self,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            load_in_8bit: bool = True,
            load_in_4bit: bool = False,
        ) -> None:

        assert not (load_in_8bit and load_in_4bit), \
            "Parameters 'load_in_8bit' and 'load_in_4bit' cannot both be True."
        
        model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device,
            quantization_config=BitsAndBytesConfig(load_in_8bit=load_in_8bit,
                                                   load_in_4bit=load_in_4bit),
        )
        HF_LOGGER.info(f"Model \"{model_name}\" successfully loaded!")
        self.eos_token_id = \
            self.tokenizer.encode("User:", add_special_tokens=False)[-1]
    

    def apply_template(self, current_msg):
        return f"User: {current_msg}\nYou: "
    

    def inference(self, msg_tpl: str) -> str:
        HF_LOGGER.debug("msg_tpl:\n\n", msg_tpl)
        msg_tpl_pt: torch.Tensor = \
            self.tokenizer(msg_tpl, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(
            msg_tpl_pt,
            max_new_tokens=512,
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
    

    def __call__(self, current_msg: str) -> str:
        msg_tpl = self.apply_template(current_msg)
        response = self.inference(msg_tpl)
        response = response.removeprefix(msg_tpl).removesuffix(self.stopping_sign)
        response = response.strip()
        return response



class Qwen(HfBaseModel):
    def __init__(
            self,
            param_num: float = 14,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            load_in_8bit: bool = False,
            load_in_4bit: bool = True,
        ) -> None:
        super().__init__()

        assert param_num in [ 0.5, 1.8, 4, 7, 14 ], \
            "Parameter 'param_num' invalid."
        assert not (load_in_8bit and load_in_4bit), \
            "Parameters 'load_in_8bit' and 'load_in_4bit' cannot both be True."

        model_name = f"Qwen/Qwen1.5-{param_num}B-Chat"
        if   load_in_8bit: model_name += "-GPTQ-Int8"
        elif load_in_4bit: model_name += "-GPTQ-Int4"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=device)
        HF_LOGGER.info(f"Model \"{model_name}\" successfully loaded!")


    def inference(self, messages: str) -> str:
        msg_tpl = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([msg_tpl], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


    def classify_message(self, current_msg: str) -> bool:
        content = \
f"""Please tell me whether the given message is a request for generating codes or not.
Please reply with "True" or "False" only.

Example 1:
User: Hi! My name is Aisu.
You: False

Example 2:
User: Please write me a code of quick sort.
You: True

Given message: {current_msg}"""
        messages = [{ "role": "user", "content": content }]
        is_code_gen = self.inference(messages)=="True"
        return is_code_gen


    def __call__(self, current_msg: str) -> str:
        return self.classify_message(current_msg)



class DeepseekCoderInstruct(HfBaseModel):
    def __init__(
            self,
            param_num: float = 6.7,
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            load_in_8bit: bool = False,
            load_in_4bit: bool = False,
        ) -> None:
        super().__init__()

        assert param_num in [ 6.7, 33 ], \
            "Parameter 'param_num' invalid."
        assert not (load_in_8bit and load_in_4bit), \
            "Parameters 'load_in_8bit' and 'load_in_4bit' cannot both be True."
        
        model_name = f"deepseek-ai/deepseek-coder-{param_num}b-instruct"
        if param_num == 33:
            load_in_8bit, load_in_4bit = False, True
            HF_LOGGER.info(f"Loading {model_name}, using 4bit quantization.")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=BitsAndBytesConfig(load_in_8bit=load_in_8bit,
                                                   load_in_4bit=load_in_4bit),
            trust_remote_code=True,
        )
        HF_LOGGER.info(f"Model \"{model_name}\" successfully loaded!")


    def inference(self, messages: str) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=False,
                                      # temperature=0.1, top_p=0.95,
                                      top_k=50, num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][len(inputs[0]):],
                                     skip_special_tokens=True)


    def __call__(self, current_msg: str) -> str:
        messages = [{ "role": "user", "content": current_msg }]
        return self.inference(messages)