# LLM-Coder-with-Discord

A discord bot which can call LLMs using either huggingface or vllm on Windows platform.

## Environment

1. Dependency

   After created and activated a virtual environment. <br>
   Using either `./setup.sh` or `pip install -r requirements.txt` with some manual install (see more in the file).<br>
   To do customized adjustment, you can edit the aboved files.

2. Environment Variables

   Edit the content in file "_.env_sample_" and rename it into "_.env_".

## LLM Server

If you are not using Hugging Face but vLLM server:

1. (For Windows) Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

2. Build the docker service (image) and create the docker container by:
   ```
   docker-compose create
   ```

3. (Optional) Start the container either using the GUI, discord bot command, or the following command:
   ```
   docker-compose start
   ```

## Execution

```
python ./main.py
```