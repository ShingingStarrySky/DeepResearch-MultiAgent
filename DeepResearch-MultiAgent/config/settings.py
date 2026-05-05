import os
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


class LLMProviderSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore", env_prefix="")

    api_key: str = ""
    base_url: str = ""
    model: str = ""


class DeepSeekSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="DEEPSEEK_", extra="ignore")

    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    pro_model: str = "deepseek-chat"
    flash_model: str = "deepseek-reasoner"


class GLMSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="GLM_", extra="ignore")

    api_key: str = os.getenv("GLM_API_KEY", "")
    base_url: str = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    model: str = "glm-4-flash"


class OpenAISettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="OPENAI_", extra="ignore")

    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = "gpt-4o"


class SemanticScholarSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="SEMANTIC_SCHOLAR_", extra="ignore")

    api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    base_url: str = "https://api.semanticscholar.org/graph/v1"


class PostgresSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="POSTGRES_", extra="ignore")

    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    db: str = os.getenv("POSTGRES_DB", "deepresearch")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class Neo4jSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_prefix="NEO4J_", extra="ignore")

    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")


class Settings:
    def __init__(self):
        self.deepseek = DeepSeekSettings()
        self.glm = GLMSettings()
        self.openai = OpenAISettings()
        self.semantic_scholar = SemanticScholarSettings()
        self.postgres = PostgresSettings()
        self.neo4j = Neo4jSettings()

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.screening_threshold: float = float(os.getenv("SCREENING_THRESHOLD", "0.72"))
        self.max_papers_per_day: int = int(os.getenv("MAX_PAPERS_PER_DAY", "30"))
        self.debate_max_rounds: int = int(os.getenv("DEBATE_MAX_ROUNDS", "3"))
        self.trend_analysis_interval_days: int = int(
            os.getenv("TREND_ANALYSIS_INTERVAL_DAYS", "7")
        )
        self.num_researchers: int = int(os.getenv("NUM_RESEARCHERS", "6"))

        self.project_root: Path = PROJECT_ROOT
        self.data_dir: Path = PROJECT_ROOT / "data"
        self.log_dir: Path = PROJECT_ROOT / "logs"
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
