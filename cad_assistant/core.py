from typing import Any, Dict, Optional, List, Union
from langchain_experimental.utilities import PythonREPL
import importlib.util
import types
from .logger import CADAssistantLogger
from pathlib import Path
import importlib
import inspect
import os, json
class CADAssistantCore:
    def __init__(self, planner_chain, prompt_name: str = "default", project_file: str = 'test.FCStd'):
        self.planner_chain = planner_chain
        self.history = []
        self.prompt_name = prompt_name
        self.project_file = project_file
        
        # Initialize logging for this run
        self.logger = CADAssistantLogger(project_file=project_file)
        self.logger.log_initialization(prompt_name, project_file)
        self.planner_chain.logdir = self.logger.log_dir

        base = {"__builtins__": __builtins__}
        self.python_repl  = PythonREPL(_globals=base, _locals=base)

        # **patch the instance right after it is created**
        self.python_repl.locals = self.python_repl.globals  

        self.env_init_response = self._run_module_initialization(project_file)
    
    
    def _run_module_initialization(self, project_file: str = None):
        """Execute a Python module's code in the REPL environment."""
        try:

            # Step 1: Execute the init_freecad module
            module_path = "cad_assistant/init_freecad.py"
            with open(module_path, 'r') as f:
                code_to_execute = f.read()
            
            # Execute the module code (imports, classes, functions)
            if code_to_execute:
                env_init_response = self.python_repl.run(code_to_execute)
                
                # Log initialization code for notebook
                self.logger.log_executed_code(
                    code=code_to_execute,
                    output=env_init_response,
                    step_info={"user_input": "System Initialization - FreeCAD Imports and Utilities"}
                )
            

            # This is used for 2D sketch question answering as the isconstruction primitive property is
            # not natevely supported from FreeCAD
            isconstruction_jsonfile = project_file[:-6] + "_isconstruction.json"


            # Step 2: Set global variables in a separate cell
            global_vars_code = ''.join((
                "# Set global configuration variables\n"
                f"project_file = r'{project_file}' if '{project_file}' else None\n"
                f"logdir = r'{self.logger.log_dir}'\n"
                "step = 0\n",
                "is_construction_dict = {}\n"
                f"if os.path.exists('{isconstruction_jsonfile}'):\n"
                f"  with open('{isconstruction_jsonfile}') as file:\n"
                "       is_construction_dict = json.load(file)\n"
            ))

            global_vars_response = self.python_repl.run(global_vars_code)


            # Log global variables setup
            self.logger.log_executed_code(
                code=global_vars_code,
                output=global_vars_response,
                step_info={"user_input": "System Initialization - Global Variables Setup"}
            )
            
            # Step 3: Initialize the FreeCAD environment in another cell
            init_code = (
            "doc, sketch = initialize_freecad_environment(project_file)"
            )
            init_response = self.python_repl.run(init_code)
            
            # Log FreeCAD environment initialization
            self.logger.log_executed_code(
                code=init_code,
                output=init_response,
                step_info={"user_input": "System Initialization - FreeCAD Environment Setup"}
            )
            
            # Combine responses for logging
            if env_init_response and init_response:
                env_init_response = env_init_response + "\n" + init_response
                    
            # Step 4: Load and register tools from cad_tools directory
            self.load_and_register_tools()
                    
        except Exception as e:
            print(f"❌ Failed to run module initialization: {e}")

        return env_init_response
    
    def load_and_register_tools(self) -> str:
        """Load and register tools from the cad_tools directory based on config."""
        import os
        
        tools_dir = "cad_assistant/cad_tools"
        if not os.path.exists(tools_dir):
            raise Exception(f"Tools directory not found: {tools_dir}")
        
        # Get tools from config
        config = self.planner_chain.config
        registered_tools = config.get("tools", [])
        
        if not registered_tools:
            return "# No tools registered in config"
        
        tools_loaded = []
        
        # Load each registered tool
        for tool_name in registered_tools:
            code_file = os.path.join(tools_dir, f"{tool_name}_code.py")
            doc_file = os.path.join(tools_dir, f"{tool_name}_doc.txt")
            
            # Validate both files exist
            if not os.path.exists(code_file):
                raise Exception(f"Tool code file not found: {code_file}")
            if not os.path.exists(doc_file):
                raise Exception(f"Tool documentation file not found: {doc_file}")
            
            # Load the tool code
            try:
                with open(code_file, 'r') as f:
                    tool_code = f.read()
                
                # Execute the tool code in the REPL
                response = self.python_repl.run(tool_code)
                tools_loaded.append(tool_name)
                
                # Log the tool loading
                self.logger.log_executed_code(
                    code=tool_code,
                    output=response,
                    step_info={"user_input": f"Tool Loading - {tool_name}"}
                )
                
            except Exception as e:
                raise Exception(f"Failed to load tool {tool_name}: {e}")
        
        if tools_loaded:
            return f"# Loaded tools: {', '.join(tools_loaded)}"
        return ""
    
    def step(self, user_input: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Execute one user step with iterative planning and action execution.
        
        Args:
            user_input: The user's input/request
            max_iterations: Maximum number of planning iterations to prevent infinite loops
            
        Returns:
            Dictionary containing the final step record with all iterations
        """
        iteration_count = 0
        current_input = user_input
        iteration_history = []
        final_plan = ""
        final_action = ""
        final_output = ""
        
        while iteration_count < max_iterations:
            # Create context for this iteration
            context = {
                "user_input": current_input,
                "env_init_response": self.env_init_response
            }
            
            # Get plan and action from planner
            response = self.planner_chain(context, iteration_history)
            plan = response.get("plan", "")
            action = response.get("action", "")
            
            # Store final values for return
            final_plan = plan
            final_action = action
            
            # Create step info for logging
            step_info_for_logging = {
                'user_input': current_input if iteration_count == 0 else f"Iteration {iteration_count + 1}",
                'plan': plan
            }
            
            output = self.execute_action(action, step_info_for_logging)
            
            final_output = output
            
            # Record this iteration
            iteration_record = {
                "iteration": iteration_count + 1,
                "plan": plan,
                "action": action,
                "output": output
            }
            iteration_history.append(iteration_record)
            
            iteration_count += 1
            
            # Check for termination
            if "TERMINATE" in plan:
                if hasattr(self.planner_chain, 'reset_steps'):
                    self.planner_chain.reset_steps()

                self.postprocessing()
                break
            
            # For next iteration, the input becomes the execution result
            current_input = f"Previous action result: {output}"
        
        # If we hit max iterations without terminating
        if iteration_count >= max_iterations and "TERMINATE" not in final_plan:
            final_plan = f"MAX_ITERATIONS_REACHED: {final_plan}"
            # Run the same postprocessing as when terminating normally
            if hasattr(self.planner_chain, 'reset_steps'):
                self.planner_chain.reset_steps()
            self.postprocessing()
        
        # Record this step in main history
        step_record = {
            "user_input": user_input,
            "plan": final_plan,
            "action": final_action,
            "output": final_output,
            "iterations": iteration_history,
            "total_iterations": iteration_count
        }
        self.history.append(step_record)
        
        # Log this step
        self.logger.log_step(step_record)
        
        return step_record
    

    def postprocessing(self):

        final_project_file = str(Path(self.logger.log_dir) / 'final.FCStd')
        postprocessing_action = f'doc.FileName = "{final_project_file}"\n'
        postprocessing_action += "doc.save()"

        self.python_repl.run(postprocessing_action)
        
        # Generate final reports when session terminates
        self.generate_session_reports()


    def execute_action(self, action, step_info: Dict[str, Any] = None):
        # Execute the action if provided
        output = ""
        if action.strip():
            try:
                output = self.python_repl.run(action)
                
                # Log the executed LLM-generated code for notebook generation
                self.logger.log_executed_code(
                    code=action,
                    output=output,
                    step_info=step_info
                )
                
            except Exception as e:
                output = f"ERROR: {str(e)}"
                
                # Log the failed code as well
                self.logger.log_executed_code(
                    code=action,
                    output=output,
                    step_info=step_info
                )
        else:
            # No action to execute, but still log the plan if provided
            if step_info and step_info.get('plan'):
                self.logger.log_executed_code(
                    code="",  # No code
                    output="",  # No output
                    step_info=step_info
                )
        return output


    def load_initialization(self, initialization: Union[List[str], str, types.ModuleType]):
        """Load additional initialization code at runtime."""
        self._run_initialization(initialization, self.project_file)
    
    def run_until_terminate(self, user_input: str, max_steps: int = 10) -> List[Dict[str, Any]]:
        """Run the assistant until it returns TERMINATE plan, with automatic iteration."""
        steps = []
        current_input = user_input
        
        # Reset the planner's step counter
        if hasattr(self.planner_chain, 'reset_steps'):
            self.planner_chain.reset_steps()
        
        for step_num in range(max_steps):
            result = self.step(current_input if step_num == 0 else "")
            steps.append(result)
            
            if result["plan"] == "TERMINATE":
                break
        
        # Generate session reports
        self.generate_session_reports()
        
        return steps 
    
    def get_log_directory(self) -> str:
        """Get the log directory path for this run."""
        return self.logger.get_log_dir()
    
    def get_run_id(self) -> str:
        """Get the unique run ID for this session."""
        return self.logger.get_run_id()
    
    def generate_session_reports(self) -> None:
        """
        Generate all session reports (notebook, HTML, JSON history).
        
        This method can be called manually to generate reports at any time,
        or it's automatically called when the session terminates.
        """
        try:
            # Generate notebook report
            total_steps = len(self.history)
            self.logger.log_completion(total_steps)
            
            print(f"📁 Session report generated in: {self.get_log_directory()}")
            print(f"  📓 Jupyter notebook: session.ipynb")
            
        except Exception as e:
            print(f"❌ Failed to generate session reports: {e}")
    
    def force_save_logs(self) -> str:
        """
        Force save all logs immediately, useful for debugging or manual saves.
        
        Returns:
            Path to the log directory
        """
        self.generate_session_reports()
        return self.get_log_directory() 