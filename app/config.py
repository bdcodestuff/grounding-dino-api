from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    dino_model: str = "IDEA-Research/grounding-dino-tiny"

settings = Settings()

# In config.py
print(dir())  # See what symbols are available at the end of the file
