from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize 
app = FastAPI(title="test summarizer app", description="get summary of you content by t5-small transformer")

# load model and tokennizer 
model= T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer= T5Tokenizer.from_pretrained("./saved_summary_model")

#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

#templating
templates = Jinja2Templates(directory=".") #tellig fast api that the html file is in same directory

#define input format ==> string 
class DialogueInput(BaseModel):
    dialogue: str

#cleaning data
def clean_data(text):
    text = re.sub(r"\r\n", " ",text) # replace the \r\n\ charcter with " "
    text = re.sub(r"\s+", " ",text) # replace spaces with " "
    text = re.sub(r"<.*?>", " ",text) # replace html tags with " "
    text = text.strip().lower() #remove extra edge spaces and keep all the characters into lower case 
    return text

def summarize_dialogue(dialogue : str)->str:
    dialogue = clean_data(dialogue) #clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 128,
        truncation = True,
        return_tensors = "pt"
    ).to(device)


    # generate the summary --> whihc will be formend in token ids
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 64,
        num_beams = 4, #the transformer will give 4 diff outputs , and the best one will be selected 
        early_stopping = True 
    )

    #these token ids have to decoded into eng words 
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # special tokens= spaces , tabs etc
    return summary

# API endpoints

#Browser opens page

#       ↓
#GET /
#        ↓
#FastAPI sends index.html
#       ↓
#User types dialogue
#        ↓
#User clicks summarize
#        ↓
#POST /summarize/
#        ↓
#FastAPI calls T5 model
#        ↓
#Summary returned
#        ↓
#Browser displays result
@app.post("/summarize/") 
async def summarize(dialogue_input: DialogueInput):
    summary=summarize_dialogue(dialogue_input.dialogue)
    return {"summary: " : summary}

@app.get("/", response_class=HTMLResponse) #
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

