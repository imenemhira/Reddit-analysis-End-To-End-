from  fastapi import FastAPI ,HTTPException
from pydantic import BaseModel,Field
import logging
import joblib
from pathlib import Path
from src.features.text_cleaning import clean_comment
logger=logging.getLogger(__name__)
model_path= Path('artifacts/best_model.joblib')
model=joblib.load(model_path)
logger.info('model loaded successfully')

app=FastAPI(
    title= 'reddit sentiment analysis API',
    description='API for predicting Reddit comment sentiment',
    version='1.0.0'
)
class CommentRequest(BaseModel):
    comment : str = Field(
        ...,
        min_length=1,
        description='Reddit comment to analyze'
    )
@app.get('/')
def home():
    return {
        'message': 'Reddit sentiment Analysis API is running'

    }

@app.post('/predict')

def predict_sentiment(request :CommentRequest):
    original_comment = request.comment

    cleaned_comment = clean_comment(original_comment)
    try:
        comment=request.comment.strip()
        if not comment :
            raise HTTPException(
                status_code=400,
                detail='comment cannot be empty mr !'
            )
        prediction=model.predict([cleaned_comment])
        return {
            'comment':cleaned_comment,
            'prediction':sentiment_labels[int(prediction[0])]
        }
    except HTTPException :
        raise
sentiment_labels={
    1: 'Positive',
    0 : 'neutral',
    -1: 'Negative'
}