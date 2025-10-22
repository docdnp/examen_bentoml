"""
BentoML Service for Admission Prediction API with JWT Authentication.
Provides secure endpoints for login and admission chance prediction.
"""

import sys
from datetime import datetime, timedelta

import bentoml
import jwt
import numpy as np
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/predict":
            print("Authenticating request for predict endpoint")
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})
            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
            request.state.user = payload.get("sub")
        response = await call_next(request)
        return response

# Pydantic model to validate admission input data
class AdmissionInput(BaseModel):
    gre_score: float = Field(..., description="GRE Score (260-340)")
    toefl_score: float = Field(..., description="TOEFL Score (0-120)")
    cgpa: float = Field(..., description="CGPA (0-10)")

    @validator('gre_score')
    def validate_gre_score(cls, v):
        if not 260 <= v <= 340:
            raise ValueError('GRE Score must be between 260 and 340')
        return v

    @validator('toefl_score')
    def validate_toefl_score(cls, v):
        if not 0 <= v <= 120:
            raise ValueError('TOEFL Score must be between 0 and 120')
        return v

    @validator('cgpa')
    def validate_cgpa(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('CGPA must be between 0 and 10')
        return v

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

# Create a BentoML Service using the new-style API (v1.4+)
@bentoml.service
class AdmissionPredictorService:
    def __init__(self) -> None:
        # Load the model using BentoML's sklearn API
        self.model = bentoml.sklearn.load_model("admission_predictor:latest")

    # Login endpoint
    @bentoml.api
    def login(self, credentials: dict) -> dict:
        try:
            username = credentials.get("username")
            password = credentials.get("password")
            if username in USERS and USERS[username] == password:
                token = create_jwt_token(username)
                return {"token": token}
        except Exception as e:
            print(f"Error during login: {e}")
            return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})
        else:
            print("Invalid login attempt")
            return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    # Prediction endpoint
    @bentoml.api
    def predict(self, input_data: AdmissionInput) -> dict:
        # Load scaler for input normalization
        import joblib
        scaler = joblib.load('data/processed/scaler.pkl')
        
        # Convert the input data to a numpy array (same order as training)
        input_array = np.array([
            input_data.gre_score,
            input_data.toefl_score,
            input_data.cgpa
        ]).reshape(1, -1)
        
        # Apply same scaling as during training
        input_scaled = scaler.transform(input_array)
        
        # Run prediction
        prediction = self.model.predict(input_scaled)
        
        # Return prediction with additional info
        return {
            "prediction": float(prediction[0]),
            "chance_of_admission": f"{prediction[0]:.1%}",
            "input_features": {
                "gre_score": input_data.gre_score,
                "toefl_score": input_data.toefl_score,
                "cgpa": input_data.cgpa
            }
        }

svc = AdmissionPredictorService
svc.add_asgi_middleware(JWTAuthMiddleware)
