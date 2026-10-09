from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from run import run_multi


app = FastAPI(
    title="FT Kitchen"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():

    return {
        "system":
        "FT Kitchen Intelligent Cooking Scheduler",

        "status":
        "running"
    }



@app.post("/schedule")
def schedule(data:dict):


    recipe_ids=data["recipe_ids"]


    result=run_multi(
        recipe_ids=recipe_ids
    )


    return {

        "overview":
        result.summary,


        "timeline":
        result.timeline,


        "gantt":
        result.gantt_path

    }