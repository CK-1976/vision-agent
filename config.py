from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    DASHSCOPE_API_KEY: str      
    DASHSCOPE_BASE_URL: str

    class Config:
        env_file = ".env"
        extra = "ignore"  

settings = Settings()