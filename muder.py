import os
import dotenv
from langchain.pydantic_v1 import BaseModel,Field
from langchain.tools import BaseTool
import convertapi
import autogen
from typing import Type
import json
from llama_parse import LlamaParse
import nest_asyncio
import pypdf




nest_asyncio.apply()

dotenv.load_dotenv(".env")
llamaparser = LlamaParse(
    api_key=os.environ["LLAMA_KEY"],
    result_type="markdown",
    num_workers=1,
    verbose=True,
    language="en"
)

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
        reader = pypdf.PdfReader(pdf_name)
        for i in range(len(reader.pages)):
            writer = pypdf.PdfWriter()
            writer.add_page(reader._get_page(i))
            with open(f"{output_directory}/page-{i+1}.pdf",'wb') as f:
                writer.write(f)

class ExtractorInput(BaseModel):
    input_directory: str = Field(description="should contain the name of the directory of the pdfs")
    output_directory: str = Field(description="should contain the name of output directory where the markdown to be stored")
 

class Extractor(BaseTool):
    name = "page_extractor"
    description = "Used by agents to parse and extract content of all page into markdown from a given directory and store it in the directory;"
    def _run(self,input_directory,output_directory,):
        for file in os.listdir(input_directory):
            page_path = input_directory+"/"+file
            file_name = file.split(".")[0]
            result = llamaparser.load_data(
                file_path=page_path,
            )
            with open(f'{output_directory}/{file_name}.md','w') as f:
                f.write(result[0].text)

class WriterInput(BaseModel):
    agent: str = Field(description="should contain which agent is using this")
    inference: str = Field(description="should contain the inference the agent got")
    page_no: str = Field(description="page no")


class Writer(BaseTool):
    name = "write_inference"
    description = "Used by agents to write their inference as txt file that can be useful later;"
    def _run(self,agent_name,inference,page_no):
        with open(f"inferences/{agent_name}_{page_no}-inference.txt",'w') as f:
            f.write(inference)

class ReaderInput(BaseModel):
    markdown_path: str = Field(description="should contain the path of the markdown file")

class Reader(BaseTool):
    name = "markdown_reader"
    description = "Used by agent to get the text presented in the given markdown file;"

    def _run(self,markdown_path):
        with open(markdown_path,'r') as f:
            result = f.read()
        return result

class Retriever(BaseTool):
    name = "master_inference"
    description = "Used by LeadDetective agent to get all the inferences written by detectives and investigator"

    def _run(self):
        result = ""
        for file in os.listdir('inferences'):
            print(file)
            with open(f'inferences/{file}','r') as f:
                result+=f.read()
        
        return result

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
reader = Reader()
writer = Writer()
extractor = Extractor()
inference = Retriever()

config_list_memgpt = [
    {
        "model": "gpt-4",
        "preset": "memgpt_chat",
        "model_wrapper": None,
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1",
        "context_window": 8192,  # gpt-4 context window
        "openai_key": os.environ["OPENAI_API_KEY"],
    },
]

interface_kwargs = {
    "debug": False,
    "show_inner_thoughts": True,
    "show_function_outputs": False,
}

llm_config_memgpt = {
    "config_list":config_list_memgpt,
    "seed":42,
    "functions": [
        generate_llm_config(maker),
        generate_llm_config(splitter),
        generate_llm_config(reader),
        generate_llm_config(writer),
        generate_llm_config(extractor),
        generate_llm_config(inference)
    ],
    
}

llm_config = {
    "functions": [
        generate_llm_config(maker),
        generate_llm_config(splitter),
        generate_llm_config(reader),
        generate_llm_config(writer),
        generate_llm_config(extractor),
        generate_llm_config(inference)
    ],
    "config_list":[{"model":"gpt-4","api_key":os.environ["OPENAI_API_KEY"]}],
    "timeout":120,

}
llm_config_manager = llm_config.copy()
llm_config_manager.pop("functions",None)
llm_config_manager.pop("tools",None)

user = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir":"coding",
        "use_docker":False
    },
    system_message="Keeps the group on track by reminding everyone of what needs to be done next, repeating instructions/code if necessary. Reply TERMINATE if the original task is done.",
    human_input_mode="TERMINATE",
)

user.register_function(
    function_map={
        maker.name: maker._run,
        splitter.name: splitter._run,
        reader.name: reader._run,
        writer.name: writer._run,
        extractor.name: extractor._run,
        inference.name: inference._run
    }
)


pdfmaster = autogen.AssistantAgent(
    "PDFMaster",
    llm_config=llm_config,
    system_message="""You are an expert at working with pdfs.
    You  are assigned to Extensively work with pdf.
    To create directories , use 'directory_maker' tool. To split pdfs, use 'pdf_splitter' tool. To extract content, use 'page_extractor'""",
    
)

indicator = autogen.AssistantAgent(
    "Indicator",
    llm_config=llm_config,
    system_message="""You are a reminder for the Detective and Investigator.You should remember the last page no of investigation and should tell the detective and investigator the next page no to investigate. Investigations starts from you because you are the one to tell them the page no.You have to start from 1 and end at 13.""",
)



detective = autogen.AssistantAgent(
    "Juniordetective",
    system_message="""You are an expert at solving cases.
    You are keen in observing and analysing people's behaviour by their statements.
    You keep track of all the events related to the case.
    You will finally find the truth of the case.
    You are good with managing people under you to solve the cases.
    To read a case page content, use 'text_reader' tool.
    To write your inference, use 'write_inference'""",
    llm_config=llm_config,


)

# investigator = autogen.AssistantAgent(
#     "JuniorInvestigator",
#     system_message="""You are an expert at investigation.
#     You are good with solving problems and collecting evidences.
#     You are capable of deducing the underlying thing by people official statements
#     You are good with cooperation and can work very well under a lead.
#     You will interact with lead detective and will share your infered things and evidences.
#     To read a case page content, use 'text_reader' tool.
#     To write your inference and evidences related to the case, use 'write_inference'""",
#     llm_config=llm_config,


# )
Lead_Detective = autogen.AssistantAgent(
    "LeadDetective",
    system_message="""You are an lead detective.
    You are an expert in deducing solutions from inferences written by junor detectives.
    Use 'master_inference' tool to get all the inference written and try your best to deduce the solution""",
    llm_config=llm_config,


)

groupchat = autogen.GroupChat(agents=[user,indicator,detective,Lead_Detective], messages=[],max_round=100,speaker_selection_method="auto")
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config_manager)

user.initiate_chat(
    manager,
    message="""
    You have already extracted content in 'extracted_content' directory with each page named 'page-' followed by page no and .md extension.You dont need to extract anything.
    Indicator agent will inform  junior detective  the page to investigate and they should try their best to understand the case and should write their own inference of that page.
    make a directory called inferences to store the inferences by detective.
    Only come to an conclusion after you read all the pages , dont skip any page.
    The lead detective has to understand the inference written by junior detective and deduce the solution.
    You have to answer the following questions.

WHO KILLED CATHERINE FOX?
Can you prove which grandchild blackmailed Catherine Fox? Prove it below to solve the case.
What TWO documents can best prove the identity of the murderer?*

Newspaper Article

Catherine's Letter to Police

Ransom Note

Crime Scene Photo of Dead Body

Photo of Suspect Lineup

Photo of Catherine Fox

Photo of Charlotte Marple

Witness Report of Charlotte Marple

Person of Interest - Alfred Christoff

Person of Interest - Edgar Christoff

Person of Interest - Gina Chesterson

Use all the agents and tools you have in your sleeve at the best to solve the case.""",
llm_config=llm_config
)