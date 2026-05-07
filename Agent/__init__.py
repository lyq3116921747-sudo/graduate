from .config import AgentConfig

__all__ = ["AgentConfig", "SepsisMedGemmaAgent"]


def __getattr__(name: str):
    if name == "SepsisMedGemmaAgent":
        from .agent import SepsisMedGemmaAgent

        return SepsisMedGemmaAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
