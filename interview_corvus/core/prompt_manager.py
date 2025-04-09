"""Manager for generating prompts for different tasks."""

from interview_corvus.config import PromptTemplates, settings


class PromptManager:
    """Manager for getting and updating prompt templates."""

    def __init__(self):
        """Initialize the prompt manager."""
        self.templates = settings.prompts.templates

    def get_template(self, template_name) -> str:
        """
        Get a specific template.

        Args:
            template_name: The name of the template to get

        Returns:
            The template text

        Raises:
            ValueError: If template not found
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def get_nvidia_code_solution_prompt(self, problem_description: str, language: str) -> str:
        """Special prompt format optimized for NVIDIA models.
        
        Args:
            problem_description: Description of the problem
            language: Programming language to use
            
        Returns:
            Formatted prompt
        """
        template = """Please solve this programming problem and provide a solution.

PROBLEM:
{problem_description}

INSTRUCTIONS:
1. Provide your solution in {language} programming language
2. Include a step-by-step explanation of your approach
3. Analyze the time complexity (Big O notation)
4. Analyze the space complexity (Big O notation)
5. List possible edge cases and how they're handled
6. Your solution must be in valid JSON format with these fields:
   - code: The complete code solution
   - explanation: Your step-by-step explanation
   - time_complexity: Time complexity analysis
   - space_complexity: Space complexity analysis
   - edge_cases: Array of edge cases (even if empty)
"""
        return template.format(
            problem_description=problem_description,
            language=language
        )

    def get_prompt(self, template_name, **kwargs) -> str:
        """
        Get a prompt with variables replaced.

        Args:
            template_name: The name of the template to get
            **kwargs: Variables to replace in the template

        Returns:
            The formatted prompt

        Raises:
            ValueError: If template not found
        """
        # Special case for NVIDIA models with code_solution template
        if (template_name == "code_solution" and 
            "nvidia" in settings.llm.model and 
            "nemotron" in settings.llm.model and
            "problem_description" in kwargs and
            "language" in kwargs):
            return self.get_nvidia_code_solution_prompt(
                problem_description=kwargs["problem_description"],
                language=kwargs["language"]
            )
            
        template = self.get_template(template_name)
        return template.format(**kwargs)

    def update_template(self, template_name, new_template) -> None:
        """
        Update a specific template.

        Args:
            template_name: The name of the template to update
            new_template: The new template text

        Raises:
            ValueError: If template not found
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        self.templates[template_name] = new_template
        settings.save_user_settings()

    def get_all_template_names(self) -> list:
        """
        Get all available template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())
