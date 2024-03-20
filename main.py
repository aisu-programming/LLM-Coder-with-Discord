##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
os.environ["HF_HOME"] = os.getenv("HF_HOME")
import time
import json
import json5
import openai
import typing
import asyncio
import discord
import logging
import functools
import subprocess
from discord.threads import Thread
from typing import Union, List, Dict
from qwen_agent.utils.utils import extract_code
from libs import (
    HfBaseModel,
    HfZephyr7bBeta,
    HfQwen,
    HfDeepseekCoderInstruct,
    VllmDockerLcModel,
    VllmDockerQwenAgent,
)
from discord.channel import (
    TextChannel,
    DMChannel,
    GroupChannel,
    PartialMessageable,
    VoiceChannel,
    StageChannel,
)
MessageableChannel = Union[TextChannel, VoiceChannel, StageChannel, Thread,
                           DMChannel, PartialMessageable, GroupChannel]
VllmDockerModel    = Union[VllmDockerLcModel, VllmDockerQwenAgent]




##### Parameters #####
MODEL_NAME    : str = str(os.getenv("MODEL_NAME"))
MAX_MODEL_LEN : int = int(os.getenv("MAX_MODEL_LEN"))
VLLM_PORT     : int = int(os.getenv("VLLM_PORT"))
DISCORD_TOKEN : str = str(os.getenv("DISCORD_TOKEN"))
DC_LOG_LEVEL  : int = logging.WARNING
MAIN_LOG_LEVEL: int = logging.INFO
# MAIN_LOG_LEVEL: int = logging.DEBUG





##### Loggers #####
DC_LOGGER = logging.getLogger("discord")
DC_LOGGER.addHandler(logging.StreamHandler())
DC_LOGGER.setLevel(DC_LOG_LEVEL)

MAIN_LOGGER = logging.getLogger("Main")
MAIN_LOGGER.setLevel(MAIN_LOG_LEVEL)
MAIN_HANDLER = logging.StreamHandler()
MAIN_HANDLER.setLevel(MAIN_LOG_LEVEL)
MAIN_HANDLER.setFormatter(logging.Formatter('\n'+os.environ["LOG_FMT"], datefmt=os.environ["LOG_DATE_FMT"]))
MAIN_LOGGER.addHandler(MAIN_HANDLER)





##### Functions #####
def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


async def log_and_send(
        channel: MessageableChannel,
        message: str,
        level: int = logging.INFO
    ) -> None:
    MAIN_LOGGER.log(msg=message, level=level)
    await channel.send(f"**[SYSTEM]** *{message}*")
    return


def check_server_is_started(model: VllmDockerModel) -> bool:
    try:
        model(":)")
        return True
    except openai.APIConnectionError:
        return False


@to_thread
def report_server_started(model: VllmDockerModel) -> None:
    first_sleep_time = 330
    MAIN_LOGGER.debug(f"First time waiting for the docker to start... sleep for {first_sleep_time} secs.")
    time.sleep(first_sleep_time)
    while True:
        try:
            model(":)")
            return
        except openai.APIConnectionError:
            MAIN_LOGGER.debug("Docker is still starting... sleep for 5 secs.")
            time.sleep(5)


async def start_docker(
        model: VllmDockerModel,
        channel: MessageableChannel
    ) -> None:
    await log_and_send(channel, "Checking the docker is started or not...")
    if check_server_is_started(model):
        await log_and_send(channel, "The docker is already started!")
    else:
        await log_and_send(channel, "Starting the docker... This takes about 6 minutes.")
        await channel.send("**[SYSTEM]** *I will notice you when the docker is successfully started.*")
        subprocess.check_call(args=[ "docker-compose", "-f", "docker-compose.yml", "start" ])
        await report_server_started(model)
        await log_and_send(channel, "The docker has successfully started!")


async def restart_docker(
        model: VllmDockerModel,
        channel: MessageableChannel
    ) -> None:
    await log_and_send(channel, "Checking the docker is started or not...")
    if check_server_is_started(model):
        await log_and_send(channel, "Restarting the docker... This takes about 6 minutes.")
        await channel.send("**[SYSTEM]** *I will notice you when the docker is successfully restarted.*")
        subprocess.check_call(args=[ "docker-compose", "-f", "docker-compose.yml", "restart" ])
        await report_server_started(model, channel)
        await log_and_send(channel, "The docker has successfully restarted!")
    else:
        await log_and_send(channel, "The docker isn't started yet, please use the command \"!Start\" instead.")


async def force_restart_docker(
        model: VllmDockerModel,
        channel: MessageableChannel
    ) -> None:
    await log_and_send(channel, "Force restarting the docker... This takes about 6 minutes.\n" + \
                                        "I will notice you when the docker is successfully restarted.")
    subprocess.check_call(args=[ "docker-compose", "-f", "docker-compose.yml", "restart" ])
    await report_server_started(model, channel)
    await log_and_send(channel, "The docker has successfully restarted!")


async def stop_docker(
        model: VllmDockerModel,
        channel: MessageableChannel
    ) -> None:
    await log_and_send(channel, "Stopping the docker...")
    subprocess.check_call(args=[ "docker-compose", "-f", "docker-compose.yml", "stop" ])
    await log_and_send(channel, "The docker has successfully stopped!")


def split_message(message: str) -> List[str]:
    if len(message) <= 1900: return [ message ]
    split_messages = [ '' ]
    in_markdown = False
    markdown_python = False
    for msg in message.split('\n'):
        if len(split_messages[-1]) + len(msg) <= 1900:
            split_messages[-1] += msg+'\n'
            if "```" in msg:
                in_markdown = not in_markdown
                markdown_python = in_markdown and "```py" in msg
        else:
            if in_markdown:
                split_messages[-1] += "```"
                msg = f"```py\n{msg}" if markdown_python else f"```{msg}"
            split_messages.append(msg+'\n')
    MAIN_LOGGER.debug(f"Response was splitted into {len(split_messages)} slices.")
    return split_messages


def process_qwen_response_list(response_list: List[Dict]) -> List[Dict]:
    adjusted_response_list = []
    for response in response_list:
        content: str = response.get("content")
        if content:
            # content = content.replace("stdout:", '').strip()
            if response["role"] == "assistant":
                role = "Bot"
            elif response["role"] == "function":
                function_name: str = response["name"]
                function_name_split = function_name.split('_')
                role = ' '.join([ n.capitalize() for n in function_name_split ])
            else:
                role = response["role"]
        elif "function_call" in response:
            role = "Function Call"
            called_function = response['function_call']['name']

            # Temp
            if called_function == "code_interpreter":
                content = f"Called function: {called_function}\n"
                content += "Arguments: Skipped due to parsing problem."

            else:
                content = f"Called function: {called_function}\n"
                # # print('\n\n', response['function_call']['arguments'], '\n\n')
                # arguments = json.loads(response['function_call']['arguments'])
                # if len(arguments["content"]) > 30:
                #     arguments["content"] = arguments["content"][:30] + "..."
                # content += f"Arguments:\n```{json.dumps(arguments, indent=4)}```"
        adjusted_response_list.append((role, content))
    return adjusted_response_list





##### Classes #####
class DiscordBot(discord.Client):
    def __init__(
            self,
            model: HfBaseModel | VllmDockerModel,
            intents: discord.Intents,
            **options: dotenv.Any
        ) -> None:
        super().__init__(intents=intents, **options)
        self.model = model

    async def on_ready(self) -> None:
        MAIN_LOGGER.info(f"Discord bot \"{self.user}\" connected!")

    async def on_message(self, dc_msg: discord.message.Message) -> None:
        # Prevent the bot from replying its own message
        if dc_msg.author.id != int(os.getenv("USER_ID")): return
        MAIN_LOGGER.debug(f"dc_msg: {dc_msg}")
        message = dc_msg.content
        message_pruned = message[:20] + "..." if len(message) > 20 else message
        MAIN_LOGGER.info(f"Received message: \"{message_pruned}\" from \"{dc_msg.author.name}\".")

        if type(self.model) is VllmDockerModel:
            if message == "!Start":
                await start_docker(self.model, dc_msg.channel)
                return
            elif message == "!Restart":
                await restart_docker(self.model, dc_msg.channel)
                return
            elif message == "!ForceRestart":
                await force_restart_docker(self.model, dc_msg.channel)
                return
            elif message == "!Stop":
                await stop_docker(self.model, dc_msg.channel)
                return

        if type(self.model) is not VllmDockerQwenAgent:
            response = self.model(dc_msg.content)
            MAIN_LOGGER.debug(f"Generated response: \"{response}\".")
            msg = await dc_msg.channel.send(response)
            response_pruned = response[:20] + "..." if len(response) > 20 else response
            MAIN_LOGGER.info(f"Replied: \"{response_pruned}\".")
        else:
            response_list = self.model(dc_msg.content)
            response_list = process_qwen_response_list(response_list)
            for role, content in response_list:
                split_messages = split_message(f"# {role}:\n{content}")
                for message in split_messages:
                    msg = await dc_msg.channel.send(message)
                    time.sleep(1)
                content_pruned = content[:20] + "..." if len(content) > 20 else content
                MAIN_LOGGER.info(f"Replied: \"{content_pruned}\".")





##### Execution #####
if __name__ == "__main__":
    # model: VllmDockerLcModel = VllmDockerLcModel(MODEL_NAME, MAX_MODEL_LEN, VLLM_PORT)
    model: VllmDockerQwenAgent = VllmDockerQwenAgent(MODEL_NAME, VLLM_PORT)
    bot = DiscordBot(model=model, intents=discord.Intents.default())
    bot.run(DISCORD_TOKEN)