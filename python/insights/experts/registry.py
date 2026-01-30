import yaml
import importlib
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from insights.experts.base import BaseExpert

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """
    Central registry for all available experts.
    Support dynamic loading from YAML configuration.
    """
    
    def __init__(self):
        self._experts: Dict[str, BaseExpert] = {}
        self._config: Dict[str, Any] = {}

    def load_from_yaml(self, path: str | Path) -> None:
        """Load expert configurations from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.error(f"Experts config not found at {path}")
            return

        try:
            with open(path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            
            experts_cfg = self._config.get("experts", {})
            for expert_id, cfg in experts_cfg.items():
                if not cfg.get("enabled", True) and "enabled" in cfg:
                    logger.info(f"Expert {expert_id} is disabled. Skipping.")
                    continue
                
                class_path = cfg.get("class")
                if not class_path:
                    logger.warning(f"No class specified for expert {expert_id}")
                    continue
                
                try:
                    # Dynamically load the class
                    module_path, class_name = class_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    expert_class = getattr(module, class_name)
                    
                    # Instantiate and register
                    # NOTE: We currently assume experts have a default constructor
                    # but we could pass config to them if we evolve the BaseExpert.
                    expert_instance = expert_class()
                    self.register(expert_instance)
                    
                except (ImportError, AttributeError, Exception) as e:
                    logger.error(f"Failed to load expert {expert_id} ({class_path}): {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse experts config: {e}")

    def register(self, expert: BaseExpert) -> None:
        """Register a new expert instance."""
        if expert.expert_id in self._experts:
            logger.warning(f"Expert {expert.expert_id} already registered. Overwriting.")
        
        self._experts[expert.expert_id] = expert
        logger.info(f"Registered expert: {expert.expert_id}")

    def get_expert(self, expert_id: str) -> Optional[BaseExpert]:
        """Get an expert by ID."""
        return self._experts.get(expert_id)

    def list_experts(self) -> List[BaseExpert]:
        """List all registered experts."""
        return list(self._experts.values())

    def clear(self) -> None:
        """Clear all experts from the registry."""
        self._experts.clear()


# Global registry instance
registry = ExpertRegistry()
