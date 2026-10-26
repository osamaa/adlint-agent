"""AWS Lambda entry point: the same FastAPI app, behind a Lambda Function URL."""

from mangum import Mangum

from webapp.app import app

handler = Mangum(app, lifespan="off")
