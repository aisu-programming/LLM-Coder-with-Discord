##### Libraries #####
import os
import json5
import logging
import requests
import subprocess
from bs4 import BeautifulSoup
from typing import Union, Optional, Dict, List
from qwen_agent.agents import Assistant
from qwen_agent.utils.utils import extract_code
from qwen_agent.llm.base import ModelServiceError
from qwen_agent.tools.base import BaseTool, register_tool





##### Parameters #####
LOG_LEVEL: int = logging.INFO





##### Loggers #####
LOGGER = logging.getLogger("Qwen")
LOGGER.setLevel(LOG_LEVEL)
HANDLER = logging.StreamHandler()
HANDLER.setLevel(LOG_LEVEL)
HANDLER.setFormatter(logging.Formatter('\n'+os.environ["LOG_FMT"], datefmt=os.environ["LOG_DATE_FMT"]))
LOGGER.addHandler(HANDLER)





##### Classes #####
@register_tool("my_web_extractor")
class MyWebExtractor(BaseTool):
    description = "A tool to extract information from provided URL."
    parameters = [{
        "name": "url",
        "type": "string",
        "description": "The URL of the website",
        "required": True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        url = params["url"]
        headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, features="html.parser")
            for script in soup(["script", "style"]): script.extract()  # rip it out
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        else:
            return ''



@register_tool("project_manager")
class ProjectManager(BaseTool):
    description = "A tool can either do operations on codes/docs, or scan the workspaces/directories."
    parameters = [{
        "name": "operate",
        "type": "string",
        "description": "Operation type. " + \
                       "Option includes: ['create', 'install', 'save', 'read', 'update', 'delete', 'walk'] " + \
                       "for creating a new project, installing packages, " + \
                       "saving/reading/updating/deleting codes/docs, " + \
                       "or walking through projects.",
        "required": True
    }, {
        "name": "project name",
        "type": "string",
        "description": "The name of the current doing project" + \
                       "Required when creating a new project " + \
                       "or saving/reading/updating/deleting the codes/docs." + \
                       "Optional when walking through projects." + \
                       "Cannot be empty string.",
    }, {
        "name": "install command",
        "type": "string",
        "description": "The command line to pip install packages, it's okay to be complicated." + \
                       "E.g. 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'." + \
                       "Only required when installing packages."
    }, {
        "name": "filename",
        "type": "string",
        "description": "The name of the file to save, read, update, or delete." + \
                       "Required when saving/reading/updating/deleting the codes/docs." + \
                       "Cannot be empty string.",
    }, {
        "name": "content",
        "type": "string",
        "description": "Complete content of the runnable program/code or document. " + \
                       "Required when saving/updating the programs/documents." + \
                       "Should not be empty string.",
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.root = "projects"
        os.makedirs(self.root, exist_ok=True)
        self.args_format = "Content应为Markdown代码块。"

    def call(self, params: Union[str, dict], **kwargs) -> str:

        try:
            # Read, Delete, and Walk should be able to parse
            params = json5.loads(params)
        except Exception:
            # For Save and Update
            content = extract_code(params)
            params = params.replace(content, '')
            params = params[:params.index(', "content":')] + '}'
            params = json5.loads(params)
            params["content"] = content

        operate = params["operate"]
        assert operate in ["create", "install", "save", "read", "update", "delete", "walk"], \
            "Parameter 'operate' invalid."
        
        if operate == "create":
            assert "project name" in params, "Parameter 'project name' is necessary."
            return self.create(os.path.join(self.root, params["project name"]))
        
        elif operate == "install":
            assert "project name"    in params, "Parameter 'project name' is necessary."
            assert "install command" in params, "Parameter 'install command' is necessary."
            return self.install(os.path.join(self.root, params["project name"]), params["install command"])
        
        elif operate in ["save", "read", "update", "delete"]:

            assert "project name" in params, "Parameter 'project name' is necessary."
            assert "filename"     in params, "Parameter 'filename' is necessary."
        
            project_path = os.path.join(self.root, params["project name"])
            if project_path.startswith('/'): project_path = project_path[1:]
            file_path = os.path.join(project_path, params["filename"])

            if operate == "save":
                assert "content" in params, "Parameter 'content' is necessary."
                return self.save(file_path, params["content"])
            elif operate == "read":
                return self.read(file_path)
            elif operate == "update":
                assert "content" in params, "Parameter 'content' is necessary."
                # Temp
                return self.save(file_path, params["content"])
                return self.update(file_path, params["content"])
            elif operate == "delete":
                return self.delete(file_path)
        
        elif operate == "walk":
            path = os.path.join(self.root, params["project name"]) \
                if "project name" in params else self.root
            return self.walk(path)
    
    def create(self, project_path: str) -> str:
        os.makedirs(project_path, exist_ok=True)
        command = ["python", "-m", "venv", "venv4W"]
        completed_process = subprocess.run(command, text=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            cwd=project_path)
        if completed_process.returncode != 0:
            return completed_process.stderr
        if os.path.exists(f"{project_path}/requirements.txt"):
            command = ["cmd.exe", "/c", "venv4W\\bin\\activate", "&&",
                       "pip", "install", "-r", "requirements.txt"]
            completed_process = subprocess.run(command, text=True,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                                cwd=project_path)
        if completed_process.returncode != 0:
            return completed_process.stderr
        else:
            return "SUCCESS"
        
    def install(self, project_path: str, command: str) -> str:
        command = ["cmd.exe", "/c", "venv4W\\Scripts\\activate", "&&"] + command.split(' ')
        completed_process = subprocess.run(command, text=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            cwd=project_path)
        if completed_process.returncode != 0:
            return completed_process.stderr
        else:
            return "SUCCESS"

    def save(self, file_path: str, value: str) -> str:
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(value)
        return "SUCCESS"

    def read(self, file_path: str) -> str:
        with open(file_path, 'r', encoding="utf-8") as file:
            file_content = file.read()
        return file_content

    def delete(self, file_path: str) -> str:
        os.remove(file_path)
        return "SUCCESS"

    def walk(self, path: str) -> str:
        dir_info = list(os.walk(path))
        # Temp logic
        dir_info = list(filter(lambda t: "venv" not in t[0] and len(t[1]) < 10 and len(t[2]) < 10, dir_info))
        print(dir_info)
        return str(dir_info)



@register_tool("my_code_executor")
class MyCodeExecutor(BaseTool):
    description = "A tool to execute existing Python codes file with virtual environment and get the result." + \
                  "Unlike the original 'code_interpreter', this tool doesn't require inputting codes."
    parameters = [{
        "name": "project name",
        "type": "string",
        "description": "The name of the current doing project.",
        "required": True
    }, {
        "name": "filename",
        "type": "string",
        "description": "The name of the file contains Python code to execute.",
        "required": True
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.root = "projects"

    def call(self, params: Union[str, dict], timeout: Optional[int] = 30, **kwargs) -> str:
        params = json5.loads(params)
        assert "project name" in params, "Parameter 'project name' is necessary."
        assert "filename"     in params, "Parameter 'filename' is necessary."
        project_name, filename = params["project name"], params["filename"]

        command = ["cmd.exe", "/c", "venv4W\\Scripts\\activate", "&&", "python", filename ]
        completed_process = subprocess.run(command, text=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            cwd=os.path.join(self.root, project_name))
        
        if completed_process.returncode != 0:
            return completed_process.stderr
        else:
            result = completed_process.stdout
            return result if result.strip() else "Finished execution."



class VllmDockerQwenAgent(Assistant):
    def __init__(self, model_name, vllm_port):
        llm_cfg = {
            "model": model_name,
            "model_server": f"http://localhost:{vllm_port}/v1",
            "api_key": "EMPTY",
            "generate_cfg": { "top_p": 0.9 }
        }
        tools = ["my_code_executor", "my_web_extractor", "project_manager"]
        super().__init__(
            llm=llm_cfg,
            function_list=tools,
            # system_message=system,
            # files=[ os.path.abspath("doc.pdf") ],
        )
        # self.history_messages = [{
        #     "role": "system",
        #     "content": "When generating code, " + \
        #                "use '#' to write comment. " + \
        #                "Do not use triple quotes (\"\"\") to write comment."
        # }]
        self.history_messages = []
        
    def __call__(self, msg: str) -> List[Dict]:
        self.history_messages.append({ "role": "user", "content": msg })
        if sum(len(m["content"]) for m in self.history_messages) > 5000:
            self.remove_long_message()
        while True:
            try:
                for response_list in self.run(messages=self.history_messages):
                    pass
                break
            except ModelServiceError as ex:
                print(type(ex), ex)
                print(type(ex.message), ex.message)
                if "max_tokens must be at least 1" in ex.message:
                    if self.remove_long_message():
                        LOGGER.info("The length of history messages is too long. Removed some messages.")
                    else:
                        raise Exception("Special case?: ", ex)
                else:
                    raise ex

        for response in response_list:

            print('\n', type(response), response, '\n')

            if response.get("name", '') == "my_web_extractor":
                response["content"] = "Deleted for saving memory. Extract again if needed."
            if response.get("name", '') == "project_manager":
                response["content"] = f"```python\n{response['content']}```"
                # if len(response["content"]) > 100:
                #     response["content"] = "Deleted for saving memory."
            self.history_messages.append(response)
        return response_list
    

    # Temp logic
    def remove_long_message(self) -> None:
        for msg_id in range(len(self.history_messages)):
            if msg_id <= 1: pass
            if self.history_messages[msg_id]["role"] != "user" and \
               len(self.history_messages[msg_id]["content"]) > 500:
                self.history_messages[msg_id]["content"] = "Deleted for saving memory."
                return
        for _ in range(5): self.history_messages.pop(2)
        self.history_messages.insert(
            2, { "role": "system", "content": "5 messages are deleted for saving memory." })
        return



# ##### Execution #####
# if __name__ == "__main__":
#     agent = QwenAgent()
#     while True: agent(input("Say something: "))