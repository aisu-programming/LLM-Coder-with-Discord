##### Libraries #####
import dotenv
dotenv.load_dotenv(".env")
import os
os.environ["HF_HOME"] = os.getenv("HF_HOME")
import time
import openai
import typing
import asyncio
import discord
import logging
import functools
import subprocess
from libs import (
    HfBaseModel,
    Zephyr7bBeta,
    Qwen,
    DeepseekCoderInstruct,
    VllmDockerModel,
)
from typing import Union, List
from discord.channel import (
    TextChannel,
    DMChannel,
    GroupChannel,
    PartialMessageable,
    VoiceChannel,
    StageChannel,
)
from discord.threads import Thread
MessageableChannel = Union[TextChannel, VoiceChannel, StageChannel, Thread,
                           DMChannel, PartialMessageable, GroupChannel]





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


def split_response(response: str) -> List[str]:
    if len(response) <= 2000: return [ response ]
    response_slices = response.split('.')  # 暫時用的邏輯
    MAIN_LOGGER.debug(f"Response was splitted into {len(response_slices)} slices.")
    response_slices
    return response_slices





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
        message_pruned = message[:10] + "..." if len(message) > 10 else message
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

        response = self.model(dc_msg.content)
        MAIN_LOGGER.debug(f"Generated response: \"{response}\".")
        msg = await dc_msg.channel.send(response)
        response_pruned = response[:10] + "..." if len(response) > 10 else response
        MAIN_LOGGER.info(f"Replied: \"{response_pruned}\".")
        # response_slices = split_response(response)
        # for rid, response in enumerate(response_slices):
        #     response = response.strip()
        #     if response != '':
        #         msg = await dc_msg.channel.send(response)
        #         MAIN_LOGGER.info(f"Replied ({rid+1}/{len(response_slices)}): \"{response}\".")
        #         time.sleep(3)
        #         # await msg.add_reaction("👍")
        #         # await msg.add_reaction("👎")

    # async def on_reaction_add(
    #     self,
    #     reaction: discord.reaction.Reaction,
    #     user: discord.user.User,
    # ) -> None:
    #     # Check the reaction to see what the user responded with
    #     print(reaction, user, self.user)
    #     if user != self.user:
    #         if reaction.emoji == "👍":
    #             print("Get")
    #             await reaction.message.channel.send(f"{user.name} liked this message!")
    #         elif reaction.emoji == "👎":
    #             await reaction.message.channel.send(f"{user.name} did not like this message!")


def main():
    # model: HfBaseModel = Qwen()
    print(MODEL_NAME)
    model: VllmDockerModel = VllmDockerModel(MODEL_NAME, MAX_MODEL_LEN, VLLM_PORT)
    intents = discord.Intents.default()
    # intents.messages = True
    # intents.reactions = True
    bot = DiscordBot(model=model, intents=intents)
    bot.run(DISCORD_TOKEN)





##### Execution #####
if __name__ == "__main__":
    main()