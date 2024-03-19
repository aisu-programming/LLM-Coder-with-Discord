##### Libraries #####
import os
import json5
import requests
from bs4 import BeautifulSoup
from typing import Union, Optional, Dict, List
from qwen_agent.agents import Assistant
from qwen_agent.utils.utils import extract_code
from qwen_agent.tools.base import BaseTool, register_tool





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



@register_tool("file_operator")
class FileOperator(BaseTool):
    description = "A tool can either do operations on codes/docs, or scan the workspaces/directories."
    parameters = [{
        "name": "operate",
        "type": "string",
        "description": "Operation type. " + \
                       "Option includes: ['save', 'read', 'update', 'delete', 'walk'] " + \
                       "for saving/reading/updating/deleting codes/docs, " + \
                       "or walking through the workspaces/directories.",
        "required": True
    }, {
        "name": "project name",
        "type": "string",
        "description": "The name of the current doing project" + \
                       "Required when saving/reading/updating/deleting the codes/docs." + \
                       "Optional when walking through the workspaces/directories." + \
                       "Cannot be empty.",
    }, {
        "name": "filename",
        "type": "string",
        "description": "The name of the file to save, read, update, or delete." + \
                       "Required when saving/reading/updating/deleting the codes/docs." + \
                       "Cannot be empty.",
    }, {
        "name": "content",
        "type": "string",
        "description": "Content of the codes or docs. " + \
                       "Required when saving/updating the codes/docs."
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
        assert operate in ["save", "read", "update", "delete", "walk"], \
            "Parameter 'operate' invalid."
        
        if operate in ["save", "read", "update", "delete"]:

            assert "project name" in params, "Parameter 'project name' missing."
            assert "filename"     in params, "Parameter 'filename' missing."
        
            project_path = os.path.join(self.root, params["project name"])
            os.makedirs(project_path, exist_ok=True)
            if project_path.startswith('/'): project_path = project_path[1:]
            file_path = os.path.join(project_path, params["filename"])

            if operate == "save":
                assert "content" in params, "Parameter content missing."
                return self.save(file_path, params["content"])
            elif operate == "read":
                return self.read(file_path)
            elif operate == "update":
                assert "content" in params, "Parameter content missing."
                # Temp
                return self.save(file_path, params["content"])
                return self.update(file_path, params["content"])
            elif operate == "delete":
                return self.delete(file_path)
        
        elif operate == "walk":
            path = os.path.join(self.root, params["project name"]) \
                if "project name" in params else self.root
            return self.walk(path)

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
        dir_info = list(filter(lambda _, dn, fn: len(dn) < 10 and len(fn) < 10, dir_info))
        return str(dir_info)



class VllmDockerQwenAgent(Assistant):
    def __init__(self, model_name, vllm_port):
        llm_cfg = {
            "model": model_name,
            "model_server": f"http://localhost:{vllm_port}/v1",
            "api_key": "EMPTY",
            "generate_cfg": { "top_p": 0.9 }
        }
        tools = ["code_interpreter", "my_web_extractor", "file_operator"]
        super().__init__(
            llm=llm_cfg,
            function_list=tools,
            # system_message=system,
            # files=[ os.path.abspath("doc.pdf") ],
        )
        self.history_messages = [{
            "role": "system",
            "content": "When generating code, " + \
                       "use '#' to write comment. " + \
                       "Do not use triple quotes (\"\"\") to write comment."
        }]
        
    def __call__(self, msg: str) -> List[Dict]:
        self.history_messages.append({ "role": "user", "content": msg })
        for response_list in self.run(messages=self.history_messages):
            pass
        for response in response_list:
            if response.get("name", '') == "my_web_extractor":
                response["content"] = "Deleted for saving memory."
        # response = list(filter(lambda r: r.get("name", '') != "my_web_extractor", response))
        print("bot response:", response_list)
        self.history_messages.extend(response_list)
        return response_list
    




# ##### Execution #####
# if __name__ == "__main__":
#     agent = QwenAgent()
#     while True: agent(input("Say something: "))