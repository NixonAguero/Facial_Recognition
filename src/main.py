from src.utils import args
from src.api import verify, register
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.engine.YOLOv8face import YOLOv8Face
from src.utils.constants import MODEL_PATH, DEVICE

@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = YOLOv8Face(model_path=MODEL_PATH, device=DEVICE)

    print("Calentando YOLOv8...")
    detector.warmup()
    print("Modelo listo.")

    verify.set_detector(detector)
    register.set_detector(detector)

    yield

    print("Aplicación cerrada.")


app = FastAPI(title="Facial Recognition API", lifespan=lifespan)

app.include_router(verify.router, prefix="/api")
app.include_router(register.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == '__main__':
    args = args.parse_args()

    if args.action == 'sign-up':
        register.register_user(args)
    elif args.action == 'sign-in':
        verify.verify_user(args)
