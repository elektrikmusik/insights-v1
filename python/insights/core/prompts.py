"""
Prompt Manager for loading and rendering Jinja2 templates.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages loading and rendering of prompt templates.
    Assuming templates are located in `configs/prompts/` at the project root.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            # Resolve project root from this file's location:
            # insights/core/prompts.py -> core -> insights -> python -> root
            project_root = Path(__file__).resolve().parents[3]
            templates_dir = project_root / "configs" / "prompts"
        
        self.templates_dir = templates_dir
        
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
        
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml', 'jinja2']),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a template with context.
        
        Args:
            template_name: Relative path to template (e.g., "experts/risk_analyst.jinja2")
            context: Variables to pass to the template
            
        Returns:
            Rendered string
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**(context or {}))
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name} in {self.templates_dir}")
            # Fallback or re-raise
            raise ValueError(f"Prompt template '{template_name}' not found.")
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise


# Global instance
prompt_manager = PromptManager()
