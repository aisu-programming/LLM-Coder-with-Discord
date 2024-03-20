# LLM-Coder-with-Discord

A discord bot that can call LLMs using either Hugging Face or vLLM on the Windows platform. <br>
Combined with function calling and RAG.

# Current Using Model
- Qwen1.5-14B-Chat-GPTQ-Int8 (w/ context windows length only 4096 due to 24G VRAM)
- Qwen1.5-14B-Chat-GPTQ-Int4
- Qwen1.5-7B-Chat-GPTQ-Int8

## Environment

1. Dependency

   After creating and activating a virtual environment. <br>
   Use either `./setup.sh` or `pip install -r requirements.txt` with some manual install (see more in the file).<br>
   To do customized adjustments, you can edit the above files.

2. Environment Variables

   Edit the content in the file "_.env_sample_" and rename it into "_.env_".

## vLLM Server

If you are not using Hugging Face pipelines/models but the vLLM server:

- For Windows:

  Since vLLM does not currently support Windows, we have to install vLLM using Docker.

  1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

  2. Build the docker service (image) and create the docker container by:
     ```
     docker-compose create
     ```

  3. (Optional) Start the container either using the GUI, discord bot command, or the following command:
     ```
     docker-compose start
     ```

- For other platforms:
  
  See "_docker-compose.yml_" for your reference.

## Execution

```
python ./main.py
```

## Now-implemented Tools
- My Web Extractor
- File Operator (My Storage)
- My Code Executor

## Some Demo Cases

> ![image](https://github.com/aisu-programming/LLM-Coder-with-Discord/assets/66176726/4093682f-08ed-4e51-b5c4-695acc7698a6)

> ![image](https://github.com/aisu-programming/LLM-Coder-with-Discord/assets/66176726/7917c76c-c607-4dfa-a523-f114127a4e69)

> ![image](https://github.com/aisu-programming/LLM-Coder-with-Discord/assets/66176726/c9c186c9-a514-4d85-8c8f-88295cd857b8)
