"""
CAD Assistant Logging Module
============================

Provides logging functionality for CAD Assistant runs including:
- Creating unique log directories for each run
- Copying project files to log directories
- Setting up logging configuration
- Generating Jupyter notebooks with executed code
- Creating HTML reports with formatted plans and actions
"""

import os
import shutil
import logging
import json
import nbformat as nbf
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class CADAssistantLogger:
    """
    Logger for CAD Assistant that manages log directories and file copying.
    """
    
    def __init__(self, base_log_dir: str = "logs", project_file: Optional[str] = None):
        """
        Initialize the logger for a new CAD Assistant run.
        
        Args:
            base_log_dir: Base directory for all logs (default: "logs")
            project_file: Path to project file to copy (if provided)
        """
        self.base_log_dir = Path(base_log_dir)
        self.project_file = project_file
        self.run_id = self._generate_run_id()
        self.log_dir = self._create_log_directory()
        self.logger = self._setup_logging()
        
        # Initialize notebook tracking
        self.notebook_cells = []
        self.executed_code_history = []
        self.step_counter = 0  # Track step numbers for image detection
        
        # Copy project file if provided
        if self.project_file:
            self._copy_project_file()
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp, process ID, and random component."""
        import os
        import random
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        process_id = os.getpid()
        random_component = random.randint(1000, 9999)
        return f"run_{timestamp}_{process_id}_{random_component}"
    
    def _create_log_directory(self) -> Path:
        """Create a unique log directory for this run."""
        log_dir = self.base_log_dir / self.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def _setup_logging(self) -> logging.Logger:
        """Set up Python logging to write to the log directory."""
        logger = logging.getLogger(f"cad_assistant_{self.run_id}")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        log_file = self.log_dir / "cad_assistant.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _copy_project_file(self) -> Optional[str]:
        """
        Copy the project file to the log directory.
        
        Returns:
            Path to the copied file in log directory, or None if copy failed
        """
        if not self.project_file:
            return None
            
        project_path = Path(self.project_file)
        
        if not project_path.exists():
            self.logger.warning(f"Project file does not exist: {self.project_file}")
            return None
        
        try:
            # Copy to log directory with same filename
            copied_file = self.log_dir / project_path.name
            shutil.copy2(project_path, copied_file)
            
            self.logger.info(f"✅ Project file copied: {self.project_file} -> {copied_file}")
            return str(copied_file)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to copy project file: {e}")
            return None
    
    def log_initialization(self, prompt_name: str, project_file: Optional[str]) -> None:
        """Log the initialization of a CAD Assistant run."""
        self.logger.info("🚀 CAD Assistant Run Started")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Log Directory: {self.log_dir}")
        self.logger.info(f"Prompt Name: {prompt_name}")
        self.logger.info(f"Project File: {project_file or 'None'}")
    
    def log_executed_code(self, code: str, output: str = "", step_info: Dict[str, Any] = None) -> None:
        """
        Log executed code for notebook generation.
        
        Args:
            code: Python code that was executed
            output: Output from the code execution
            step_info: Additional step information (plan, user_input, etc.)
        """
        code_entry = {
            "timestamp": datetime.now().isoformat(),
            "code": code,
            "output": output,
            "step_info": step_info or {}
        }
        self.executed_code_history.append(code_entry)
        
        # Add code cell to notebook
        if code.strip():
            # Check if this is a user step (not initialization)
            is_user_step = step_info and not step_info.get('user_input', '').startswith('System Initialization')
            self._add_notebook_cell(code, output, step_info, is_user_step)
        elif step_info and step_info.get('plan'):
            # No code but we have plan info - add plan-only cell
            is_user_step = step_info and not step_info.get('user_input', '').startswith('System Initialization')
            if is_user_step:
                self._add_plan_only_markdown_cell(step_info)
    
    def _add_notebook_cell(self, code: str, output: str = "", step_info: Dict[str, Any] = None, is_user_step: bool = False) -> None:
        """Add a cell to the notebook structure."""
        
        # Add markdown cell with step information if provided
        if step_info and step_info.get("user_input"):
            markdown_content = f"Step: {step_info['user_input']}\n\n"
            if step_info.get("plan"):
                markdown_content += f"**Plan:** {step_info['plan']}\n\n"
            
            markdown_cell = nbf.v4.new_markdown_cell(markdown_content)
            self.notebook_cells.append(markdown_cell)
        
        # Add code cell
        code_cell = nbf.v4.new_code_cell(code)
        
        # Add output if present
        if output.strip():
            # Determine if output is text or contains images/plots
            if "matplotlib" in output.lower() or "plot" in output.lower():
                # For display outputs (plots, etc.)
                code_cell.outputs = [
                    nbf.v4.new_output(
                        output_type="display_data",
                        data={"text/plain": output}
                    )
                ]
            else:
                # For regular text output
                code_cell.outputs = [
                    nbf.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text=output
                    )
                ]
        
        self.notebook_cells.append(code_cell)
        
        # Check for and add step image if it exists (only for user steps)
        if is_user_step:
            self._add_step_image_if_exists()
    
    def _check_for_step_image(self, step_number: int) -> Optional[str]:
        """
        Check if an image exists for the given step number.
        
        Args:
            step_number: The step number to check for
            
        Returns:
            Path to the image file if it exists, None otherwise
        """
        # Check for common image formats
        image_formats = ['png', 'jpg', 'jpeg', 'svg']
        
        for fmt in image_formats:
            image_path = os.path.join(self.log_dir, f"sketch_step_{step_number}.{fmt}")
            if os.path.exists(image_path):
                return image_path
        
        return None
    
    def _load_image_as_base64(self, image_path: str) -> Optional[Dict[str, str]]:
        """
        Load an image file and encode it as base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image data and format, or None if loading fails
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                # Determine image format from file extension
                ext = os.path.splitext(image_path)[1].lower().lstrip('.')
                if ext == 'jpg':
                    ext = 'jpeg'
                
                return {
                    'data': base64_data,
                    'format': ext,
                    'path': image_path
                }
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return None
    
    def _add_step_image_if_exists(self) -> None:
        """Check for and add step image to notebook if it exists."""
        image_path = self._check_for_step_image(self.step_counter)
        
        if image_path:
            image_data = self._load_image_as_base64(image_path)
            
            if image_data:
                # Create markdown cell with image title
                image_title = f"### 📊 Step {self.step_counter} Output"
                title_cell = nbf.v4.new_markdown_cell(image_title)
                self.notebook_cells.append(title_cell)
                
                # Create code cell that displays the image
                step_num = self.step_counter
                img_format = image_data['format']
                img_data = image_data['data']
                
                display_code = f"""# Display step {step_num} image
from IPython.display import Image, display
import base64

# Load and display the image
image_data = base64.b64decode('''{img_data}''')
display(Image(data=image_data, format='{img_format}'))"""
                
                image_cell = nbf.v4.new_code_cell(display_code)
                
                # Add the image as output
                image_cell.outputs = [
                    nbf.v4.new_output(
                        output_type="display_data",
                        data={
                            f"image/{image_data['format']}": image_data['data']
                        },
                        metadata={"image/png": {"width": 600, "height": 400}}
                    )
                ]
                
                self.notebook_cells.append(image_cell)
                print(f"📊 Added step {self.step_counter} image to notebook: {os.path.basename(image_path)}")
        
        # Increment step counter for next call
        self.step_counter += 1
    
    def _add_plan_only_markdown_cell(self, step_info: Dict[str, Any]) -> None:
        """Add only a markdown cell for plan-only steps (called from execute_action)."""
        if step_info and step_info.get("user_input") and step_info.get("plan"):
            markdown_content = f"Step: {step_info['user_input']}\n\n"
            markdown_content += f"**Plan:** {step_info['plan']}\n\n"
            markdown_content += f"*No code execution required*\n\n"
            markdown_content += f"*Logged at: {datetime.now().strftime('%H:%M:%S')}*"
            
            markdown_cell = nbf.v4.new_markdown_cell(markdown_content)
            self.notebook_cells.append(markdown_cell)
            
            print(f"📝 Added plan-only markdown cell: {step_info['user_input']}")
    
    def _add_plan_only_cell(self, step_data: Dict[str, Any]) -> None:
        """
        Add a markdown cell for plans that don't have associated code execution.
        This is useful for TERMINATE plans, reasoning steps, etc.
        """
        user_input = step_data.get('user_input', '')
        plan = step_data.get('plan', '')
        
        # Create descriptive title based on plan content
        if "TERMINATE" in plan:
            title = "🏁 Session Complete"
            plan_type = "**Completion:**"
        elif "MAX_ITERATIONS_REACHED" in plan:
            title = "⏰ Maximum Iterations Reached"
            plan_type = "**Status:**"
        else:
            title = f"💭 Planning Step"
            plan_type = "**Plan:**"
        
        # Create markdown content
        markdown_content = f"## {title}\n\n"
        
        if user_input and not user_input.startswith("Previous action result"):
            markdown_content += f"**Request:** {user_input}\n\n"
        
        markdown_content += f"{plan_type} {plan}\n\n"
        
        # Add any iteration information if present
        if step_data.get('iterations'):
            markdown_content += f"**Iterations:** {len(step_data['iterations'])}\n\n"
        
        # Add timestamp
        markdown_content += f"*Logged at: {datetime.now().strftime('%H:%M:%S')}*"
        
        # Create and add the markdown cell
        markdown_cell = nbf.v4.new_markdown_cell(markdown_content)
        self.notebook_cells.append(markdown_cell)
        
        print(f"📝 Added plan-only cell: {title}")
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """
        Log a step execution with all its details.
        
        Args:
            step_data: Dictionary containing step information (user_input, plan, action, output)
        """
        self.logger.info("=" * 50)
        self.logger.info(f"STEP: {step_data.get('user_input', 'N/A')}")
        self.logger.info(f"PLAN: {step_data.get('plan', 'N/A')}")
        
        if step_data.get('action'):
            self.logger.info(f"ACTION:\n{step_data['action']}")
        
        if step_data.get('output'):
            self.logger.info(f"OUTPUT:\n{step_data['output']}")
        
        self.logger.info("=" * 50)
        
        # Add special plan cells to notebook (TERMINATE, MAX_ITERATIONS, or plan-only steps)
        plan = step_data.get('plan', '')
        action = step_data.get('action', '').strip()
        
        # Add plan cell if it's a special plan (TERMINATE/MAX_ITERATIONS) or has no action
        should_add_plan_cell = (
            "TERMINATE" in plan or 
            "MAX_ITERATIONS_REACHED" in plan or 
            (plan and not action)
        )
        
        if should_add_plan_cell:
            self._add_plan_only_cell(step_data)
        
        # Note: Code execution is now logged directly in execute_action method
        # This avoids duplicate notebook cells for steps with code
    
    def log_completion(self, total_steps: int) -> None:
        """Log the completion of a CAD Assistant run."""
        self.logger.info(f"✅ CAD Assistant Run Completed")
        self.logger.info(f"Total Steps: {total_steps}")
        self.logger.info(f"Run ID: {self.run_id}")
        
        # Generate and save notebook only
        self.generate_notebook()
    
    def generate_notebook(self) -> str:
        """
        Generate a Jupyter notebook with all executed code.
        
        Returns:
            Path to the generated notebook file
        """
        try:
            # Create notebook
            nb = nbf.v4.new_notebook()
            
            # Add title cell
            title_cell = nbf.v4.new_markdown_cell(
                f"# CAD Assistant Session\n\n"
                f"**Run ID:** {self.run_id}  \n"
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
                f"**Project File:** {self.project_file or 'None'}  \n\n"
                f"This notebook contains all the code executed during the CAD Assistant session."
            )
            nb.cells.append(title_cell)
            
            # Add all recorded cells
            nb.cells.extend(self.notebook_cells)
            
            # Save notebook
            notebook_path = self.log_dir / "session.ipynb"
            with open(notebook_path, 'w') as f:
                nbf.write(nb, f)
            
            self.logger.info(f"📓 Jupyter notebook saved: {notebook_path}")
            return str(notebook_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate notebook: {e}")
            return ""
    
    def get_log_dir(self) -> str:
        """Get the path to the log directory for this run."""
        return str(self.log_dir)
    
    def get_run_id(self) -> str:
        """Get the unique run ID for this session."""
        return self.run_id 