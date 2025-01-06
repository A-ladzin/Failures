from contextlib import asynccontextmanager
import functools
from typing import AsyncGenerator, Any, Union

import json
import pickle
import uvicorn
import uvloop
from models.nli import NLIModel
from models.model_coupled import CoupledBert
from models.utils import load_model
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from transformers import BertModel, AutoTokenizer

from core.definitions import DATA_DIR, MODEL_DIR
from handlers.routes import routes
from core.settings import Settings
from core.logger import JSONLogger

logger = JSONLogger(__name__)
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    
    logger.info("Starting up...")
    await register_app_dependencies(app)
    logger.info("Start is complete!")
    
    yield
    
    logger.info("Shutdown is complete!")


async def validation_error_handler(request: Request, exception: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": exception.errors(),
            "body": exception.body
        }
    )


async def exception_handler(_: Request, __: Exception) -> Response:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def create_app()-> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        lifespan = lifespan,
        debug=settings.DEBUG,
        exception_handlers={
            RequestValidationError: validation_error_handler,
            Exception: exception_handler,
        },
    )
    app.include_router(router=APIRouter(routes=routes))
    return app


async def run_server(entrypoint: Union[str, FastAPI], port: int) -> None:
    config = uvicorn.Config(
        entrypoint,
        host=str(settings.APP_HOST),
        port=port,
        log_level=settings.LOGLEVEL.lower(),
    )
    srv = uvicorn.Server(config=config)

    logger.info("Server is running on %s:%s", settings.APP_HOST, port)

    try:
        await srv.serve()
    except Exception as exc: 
        logger.exception(exc)


def event_shutdown()-> None:
    raise Exception("Error")


async def register_app_dependencies(app: FastAPI) -> None:
    
    app.state.server_logger = logger


    bert_tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    bert_tokenizer.add_tokens(['[QUERY]','[DESCRIPTION]','[NUM]'])
    bert = BertModel.from_pretrained(MODEL_DIR / "10-8nli" )
    bert.eval()
        
    nli_model = NLIModel(bert,bert_tokenizer,embedding_dim=512,k=50,mask_prob=0)
    load_model(nli_model, MODEL_DIR / "10-8mean_sbert8___/checkpoint-404")
    bert.requires_grad_(False)

    model = CoupledBert(nli_model,bert_tokenizer,50,50)
    model.prepare_description_tokens()

    app.state.model = model

    with open(DATA_DIR / 'mapping_backend.json', 'r') as f:
        app.state.mapping = json.load(f)

    app.add_event_handler(event_type="shutdown", func=functools.partial(event_shutdown))


async def main() -> None:
    uvloop.install()

    app = create_app()

    await run_server(app, settings.APP_PORT)