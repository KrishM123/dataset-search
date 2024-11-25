from composio_langchain import ComposioToolSet, App
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

from langchain.tools import BaseTool, Tool
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import subprocess
import sys
from typing import Optional, List
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
import ast

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import subprocess
import sys
from typing import Optional, List, Dict
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
import ast

from langchain.agents import Tool
from langchain.agents import AgentType, initialize_agent
import os
from datetime import datetime
import shutil

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class JupyterNotebookExecutor:
    """Class for executing code in Jupyter notebooks and managing dependencies."""
    
    def __init__(self, query: str = "auto_ml"):
        """
        Initialize with a query that will be used to name the notebook.
        Creates a new notebook with timestamp for each run.
        """
        # Create notebooks directory if it doesn't exist
        os.makedirs('notebooks', exist_ok=True)
        
        # Generate unique notebook name using timestamp and query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_query = "".join(x for x in query[:30] if x.isalnum() or x in (' ', '_')).strip()
        sanitized_query = sanitized_query.replace(' ', '_')
        self.notebook_path = f"notebooks/{timestamp}_{sanitized_query}.ipynb"
        
        self.shell = InteractiveShell.instance()
        self._init_notebook(query)
        
        # Add a data directory for uploaded files
        self.data_dir = 'notebooks/data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _init_notebook(self, query: str):
        """Initialize a new notebook with metadata and initial markdown."""
        self.notebook = nbformat.v4.new_notebook()
        self.notebook.metadata = {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8'
            }
        }
        
        # Add initial markdown cells with context
        self.add_markdown_cell("# Auto ML Notebook\n" + 
                             f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n" +
                             f"Query: {query}")
        self._save_notebook()
    
    def _save_notebook(self):
        """Save the current state of the notebook."""
        with open(self.notebook_path, 'w') as f:
            nbformat.write(self.notebook, f)
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements and identify required packages."""
        required_packages = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        required_packages.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        required_packages.add(node.module.split('.')[0])
        except:
            pass
        return list(required_packages)
    
    def _install_dependencies(self, packages: List[str]) -> Dict[str, bool]:
        """
        Install required packages using pip.
        Returns a dictionary of package installation results.
        """
        results = {}
        for package in packages:
            try:
                # Check if package is already installed
                __import__(package)
                results[package] = True
            except ImportError:
                try:
                    print(f"Installing {package}...")
                    subprocess.check_call([
                        sys.executable, 
                        "-m", 
                        "pip", 
                        "install", 
                        package,
                        "--quiet"
                    ])
                    results[package] = True
                except subprocess.CalledProcessError:
                    results[package] = False
        return results
    
    def _execute_code(self, code: str) -> Dict[str, any]:
        """
        Execute code and return the result with metadata.
        Returns a dictionary containing execution results and metadata.
        """
        try:
            # Execute the code
            result = self.shell.run_cell(code)
            
            # Prepare the output
            output = {
                'success': True,
                'output': str(result.result) if result.result else '',
                'error': None,
                'execution_count': len(self.notebook.cells),
                'timestamp': result.execution_count
            }
            
            # Handle any errors
            if result.error_before_exec or result.error_in_exec:
                output['success'] = False
                output['error'] = str(result.error_before_exec or result.error_in_exec)
            
            return output
        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': str(e),
                'execution_count': len(self.notebook.cells),
                'timestamp': None
            }
    
    def add_markdown_cell(self, content: str):
        """Add a markdown cell to the notebook."""
        cell = new_markdown_cell(content)
        self.notebook.cells.append(cell)
        self._save_notebook()
    
    def add_code_cell(self, code: str, execution_result: Optional[Dict] = None):
        """Add a code cell to the notebook with optional execution results."""
        cell = new_code_cell(code)
        
        if execution_result:
            if execution_result['success']:
                if execution_result['output']:
                    # Create proper nbformat output structure
                    output = nbformat.v4.new_output(
                        output_type='execute_result',
                        data={'text/plain': execution_result['output']},
                        metadata={},
                        execution_count=execution_result.get('execution_count', None)
                    )
                    cell.outputs = [output]
            else:
                # Create proper error output structure
                output = nbformat.v4.new_output(
                    output_type='error',
                    ename='ExecutionError',
                    evalue=str(execution_result['error']),
                    traceback=[str(execution_result['error'])]
                )
                cell.outputs = [output]
        
        self.notebook.cells.append(cell)
        self._save_notebook()
    
    def _clean_code(self, code: str) -> str:
        """Clean code by removing markdown code block syntax and extra whitespace."""
        # Remove markdown code block syntax
        code = code.replace('```python', '').replace('```', '')
        # Remove leading/trailing whitespace while preserving indentation
        code = code.strip()
        return code
    
    def execute_code(self, code: str, install_deps: bool = True) -> Dict[str, any]:
        """Execute code and manage the notebook."""
        # Clean the code first
        code = self._clean_code(code)
        
        if install_deps:
            required_packages = self._extract_imports(code)
            if required_packages:
                self.add_markdown_cell(
                    "### Installing Dependencies\n" + 
                    "Installing the following packages: " + ", ".join(required_packages)
                )
                install_results = self._install_dependencies(required_packages)
                
                results_md = "\n".join([
                    f"- {pkg}: {'✅ Success' if status else '❌ Failed'}"
                    for pkg, status in install_results.items()
                ])
                self.add_markdown_cell(results_md)
        
        execution_result = self._execute_code(code)
        self.add_code_cell(code, execution_result)
        
        return execution_result
    
    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to the notebook's data directory.
        Returns the new path of the file in the data directory.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get just the filename from the path
        filename = os.path.basename(file_path)
        destination = os.path.join(self.data_dir, filename)
        
        # Copy the file to the data directory
        shutil.copy2(file_path, destination)
        
        return destination

def create_jupyter_tool(query: str) -> Tool:
    """Create a LangChain tool for Jupyter notebook execution."""
    executor = JupyterNotebookExecutor(query)
    
    return Tool(
        name="Jupyter Code Executor",
        func=lambda code: executor.execute_code(code),
        description="Execute Python code in a Jupyter notebook. The code will be added as a new cell and executed. "
                   "Dependencies will be automatically installed."
    )

def setup_agent(query: str):
    """Setup the LangChain agent with the Jupyter tool."""
    # Initialize the LLM
    llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key = OPENAI_API_KEY,
        model_name = "gpt-4o"
    )
    
    # Create tools with the query context
    jupyter_tool = create_jupyter_tool(query)
    tools = [jupyter_tool]
    
    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

if __name__ == "__main__":
    # First, create a notebook executor and upload the file
    executor = JupyterNotebookExecutor("data_analysis")
    file_path = "your_file_name.csv"  # Replace with your CSV file name
    new_path = executor.upload_file(file_path)
    print(f"File uploaded to: {new_path}")
    
    # Now create and run the agent
    queries = [
        "Please analyse the data given in ./Males.csv and summarise what the data is about.",
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        agent = setup_agent(query)
        result = agent.run(query)
        print(f"Result: {result}")
        print(f"Notebook saved in notebooks directory")





