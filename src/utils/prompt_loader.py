import os


def load_prompt(filename: str) -> str:
    """Load prompt text from file.

    Args:
        filename (str): Name of the prompt file (without path)

    Returns:
        str: Content of the prompt file
    """
    # Navigate from src/utils/ to project root, then to prompts/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    prompt_path = os.path.join(project_root, "prompts", filename)
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def get_detailed_summary_prompt(section: str, context: str) -> str:
    """Generate detailed summary prompt for a specific section.

    Args:
        section (str): Section name to generate prompt for
        context (str): Context information

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("detailed_summary_prompt.txt")
    return template.format(section=section, context=context)


def get_three_point_summary_prompt() -> str:
    """Generate 3-point summary prompt.

    Returns:
        str: Formatted prompt text for 3-point summary
    """
    return load_prompt("three_point_summary_prompt.txt")


def get_system_prompt() -> str:
    """Get system prompt for AI assistant.

    Returns:
        str: System prompt text
    """
    return load_prompt("system_prompt.txt")


def get_three_point_system_prompt() -> str:
    """Get system prompt for 3-point summary.

    Returns:
        str: System prompt text for 3-point summary
    """
    return load_prompt("three_point_system_prompt.txt")
