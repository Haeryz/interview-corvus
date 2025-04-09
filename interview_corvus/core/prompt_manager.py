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
        # Using named placeholders instead of positional to avoid formatting errors
        template = """You are an expert {language} programmer. Solve this programming problem with clean, complete code:

PROBLEM:
{problem_description}

IMPORTANT: Your solution MUST be presented in this exact format:

1. FIRST, write your complete code solution in a SINGLE CODE BLOCK with language tag
2. Then list Time Complexity
3. Then list Space Complexity
4. Then write your Explanation
5. Finally list Edge Cases

DO NOT split your code across multiple blocks. Keep all your code together in ONE CONTINUOUS BLOCK.
DO NOT include any empty dictionaries or interruptions in your code.

FORMAT EXAMPLE:
```{language}
def twoSum(nums, target):
    num_map = {{}}  # Using double braces to escape curly braces
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
```

Time Complexity: O(n)
Space Complexity: O(n)

Explanation:
This solution uses a hash map to store values we've seen so far and their indices.

Edge Cases:
- Empty array
- No solution exists
- Multiple valid solutions

Remember to write your COMPLETE CODE in a SINGLE uninterrupted block.
"""
        # Use safe_format to handle any problematic placeholders
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
