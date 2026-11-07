"""
prompts module for Layer 1 prompt templates.
"""

import os
from jinja2 import Environment, FileSystemLoader

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True
)

def load_prompt(name: str, **kwargs) -> str:
    """
    Renders a prompt template by filename (e.g. 'system.j2' or 'system')
    with the provided keyword arguments.
    """
    template_name = name
    if not (template_name.endswith(".j2") or template_name.endswith(".jinja2")):
        candidate = f"{template_name}.j2"
        if os.path.exists(os.path.join(_PROMPTS_DIR, candidate)):
            template_name = candidate

    template = _env.get_template(template_name)
    return template.render(**kwargs).strip()
