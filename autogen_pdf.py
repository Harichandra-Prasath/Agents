import os
import dotenv
from langchain.pydantic_v1 import BaseModel,Field
from langchain.tools import BaseTool
import convertapi
import autogen
from typing import Type
from pypdf import PdfReader
import itertools
import json

dotenv.load_dotenv(".env")

convertapi.api_secret = os.environ["CONVERT_API_KEY"]

class MakerInput(BaseModel):
    directory_name: str = Field(description="should contain the name of the directory to be created")

class Maker(BaseTool):
    name ="directory_maker"
    description = "Used by agents to create a fresh directory that can be used further"
    args_schema: Type[BaseModel] = MakerInput

    def _run(self,directory_name):
        os.mkdir(directory_name)
        return directory_name

class SplitterInput(BaseModel):
    pdf_name: str = Field(description="should contain the name of the pdf")
    output_directory: str = Field(description="should contain the name of the output directory where the splitted pages should be stored")

class Splitter(BaseTool):
    name = "pdf_splitter"
    description = "Used by agents to split the given pdf into seperate pages and store them in the given output directory"
    args_schema: Type[BaseModel] = SplitterInput

    def _run(self,pdf_name,output_directory):
        convertapi.convert('split',{
              'File':pdf_name
         },from_format='pdf').save_files(output_directory)

class CounterInput(BaseModel):
    count_directory: str = Field(description="should contain the name of the directory")

class Counter(BaseTool):
    name = "word_counter"
    description = "For all pdf files present in the given directory,count the top 5 frequent words in each pdf"
    args_schema: Type[BaseModel] = CounterInput

    def _run(self,count_directory):
        count = {}
        i = 1
        for file in os.listdir(count_directory):
            
            reader = PdfReader(f"results/"+file)
            text = reader.pages[0].extract_text()
            _count = {}
            for word in text.split():
                _count[word] = _count.get(word,0)+1
            sorted(_count,key=_count.get,reverse=True)
            count[str(i)] = dict(itertools.islice(_count.items(),5))
            i+=1
        
        return count

class StorerInput(BaseModel):
    json_: dict = Field(description="should contain the json string to be stored")

class Storer(BaseTool):
    name = "json_storer"
    description = "For given dict, you should be able to create a json file and store the dict in that json file"
    args_schema = StorerInput

    def _run(self,json_):
        with open("results.json","w") as f:
            json.dump(json_,f)
        


def generate_llm_config(tool: BaseTool):
    schema = {
        "name":tool.name,
        "description":tool.description,
        "parameters":{
            "type":"object",
            "properties":{},
            "required":[]
        }
    }

    if tool.args is not None:
        schema["parameters"]["properties"] = tool.args
    
    return schema

maker = Maker()
splitter = Splitter()
counter = Counter()
storer = Storer()

llm_config = {
    "functions": [
        generate_llm_config(maker),
        generate_llm_config(splitter),
        generate_llm_config(counter),
        generate_llm_config(storer)
    ],
    "config_list":[{"model":"gpt-4","api_key":os.environ["OPENAI_API_KEY"]}],
    "timeout":120,
}

user = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir":"coding",
        "use_docker":False
    },
    human_input_mode="TERMINATE"
)

user.register_function(
    function_map={
        maker.name: maker._run,
        splitter.name: splitter._run,
        counter.name: counter._run,
        storer.name: storer._run
    }
)

pdfmaster = autogen.AssistantAgent(
    name="PDF Master",
    system_message="""You are an expert at working with pdfs.
    You  are assigned to Extensively work with pdf.
    You are capable of taking decisions on which tool to use from your diverse set of tools""",
    llm_config=llm_config
)

user.initiate_chat(
    pdfmaster,
    message="""Please split the pdf named 'Safari.pdf' into seperate pages.
    Make sure to create a fresh directory named 'results' and use that directory to store the splitted pdf pages.
    Additonaly please count the top 5 most frequent words of the splitted pdf pages stored in the created directory.
    Write the counting results in a json file.""",
    llm_config=llm_config
)