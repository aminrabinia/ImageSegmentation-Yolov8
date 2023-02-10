from ultralytics import YOLO
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



model = YOLO('last.pt') 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.glissai.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get('/')
def root():
    return {"message": "hello from image segmentation!"}


async def gradio_infer(myimg, conf=0.50):
    if myimg is None:
        return
    results = model.predict(myimg, device='cpu', conf=conf, save=True) 
    print(results)
    if len(results[0].numpy())==0:
        return myimg
    else:
        return model.predictor.annotator.result()


io = gr.Interface(fn=gradio_infer, 
	inputs=[gr.Image(), 
			gr.Slider(0.20, 0.80, step=0.10, value=0.50, label="Set the confidence level:")],
	outputs=gr.Image(),
	examples=[["testimg.jpg", 0.30], ["testimg2.jpg", 0.60]],
	allow_flagging = 'never',
	css="footer {visibility: hidden}",
	live=True
	)

gr.mount_gradio_app(app, io, path="/gradio")


if __name__ == "__main__":

	uvicorn.run(app, host='0.0.0.0', port=8080)

