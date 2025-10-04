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



# Workflow prompt loaders

def get_paper_analysis_system_prompt() -> str:
    """Get system prompt for paper analysis phase.

    Returns:
        str: System prompt text for paper analysis
    """
    return load_prompt("paper_analysis_system.txt")


def get_paper_analysis_user_prompt(section: str) -> str:
    """Generate paper analysis user prompt for a specific section.

    Args:
        section (str): Section name to analyze

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("paper_analysis_user.txt")
    return template.format(section=section)


def get_article_writer_system_prompt() -> str:
    """Get system prompt for article writing phase.

    Returns:
        str: System prompt text for article writing
    """
    return load_prompt("article_writer_system.txt")


def get_article_writer_user_prompt(section: str, analysis: str, context: str) -> str:
    """Generate article writer user prompt.

    Args:
        section (str): Section name to write about
        analysis (str): Analysis result from paper analyzer
        context (str): Context from previous sections

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("article_writer_user.txt")
    return template.format(section=section, analysis=analysis, context=context)


def get_article_revision_user_prompt(previous_article: str, feedback: str, analysis: str) -> str:
    """Generate article revision user prompt.

    Args:
        previous_article (str): Previous version of the article
        feedback (str): Feedback from evaluator
        analysis (str): Analysis result from paper analyzer

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("article_revision_user.txt")
    return template.format(previous_article=previous_article, feedback=feedback, analysis=analysis)


def get_evaluator_system_prompt() -> str:
    """Get system prompt for article evaluation phase.

    Returns:
        str: System prompt text for evaluation
    """
    return load_prompt("evaluator_system.txt")


def get_evaluator_user_prompt(article: str, analysis: str) -> str:
    """Generate evaluator user prompt.

    Args:
        article (str): Article to evaluate
        analysis (str): Analysis result from paper analyzer (for reference)

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("evaluator_user.txt")
    return template.format(article=article, analysis=analysis)



def get_evaluator_whole_article_system_prompt() -> str:
    """Get system prompt for whole article evaluation phase.

    Returns:
        str: System prompt text for whole article evaluation
    """
    return load_prompt("evaluator_whole_article_system.txt")


def get_evaluator_whole_article_user_prompt(article: str, analysis: str) -> str:
    """Generate evaluator user prompt for whole article evaluation.

    Args:
        article (str): Whole article to evaluate
        analysis (str): Analysis result from paper analyzer (for reference)

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("evaluator_whole_article_user.txt")
    return template.format(article=article, analysis=analysis)


def get_article_revision_whole_user_prompt(previous_article: str, feedback: str, analysis: str) -> str:
    """Generate article revision user prompt for whole article.

    Args:
        previous_article (str): Previous version of the whole article
        feedback (str): Feedback from evaluator
        analysis (str): Analysis result from paper analyzer

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("article_revision_whole_user.txt")
    return template.format(previous_article=previous_article, feedback=feedback, analysis=analysis)


def get_heading_generator_system_prompt() -> str:
    """Get system prompt for heading generation phase.

    Returns:
        str: System prompt text for heading generation
    """
    return load_prompt("heading_generator_system.txt")


def get_heading_generator_user_prompt(section: str, section_content: str) -> str:
    """Generate heading generator user prompt.

    Args:
        section (str): Section name to generate heading for
        section_content (str): Content of the section

    Returns:
        str: Formatted prompt text
    """
    template = load_prompt("heading_generator_user.txt")
    return template.format(section=section, section_content=section_content)
