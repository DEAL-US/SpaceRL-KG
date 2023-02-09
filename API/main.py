import os
from subprocess import run

from fastapi import FastAPI
from pathlib import Path

app = FastAPI()

current_dir = Path(__file__).parent.resolve()

@app.get("/")
def hello(name: str):
    return {'Hello ' + name + '!'} 


if __name__ == "__main__":
    command = f"uvicorn main:app --reload".split()
    run(command, cwd = current_dir)