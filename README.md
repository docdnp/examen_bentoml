# BentoML Exam

This repository contains the basic architecture to deliver the assessment for the BentoML exam.

You are free to add other folders or files if you find it useful to do so.

Here is how the exam submission folder is structured:

```bash       
├── examen_bentoml          
│   ├── data       
│   │   ├── processed      
│   │   └── raw           
│   ├── models      
│   ├── src       
│   └── README.md
```

In order to start the project you must follow these steps:

- Fork the project on your GitHub account

- Clone the project on your machine

- Download the dataset from the following link: [Download link](https://datascientest.s3-eu-west-1.amazonaws.com/examen_bentoml/admissions.csv)


## Run the DVC pipeline

### Running it the very first time and reproduce everything:

```bash
curl https://datascientest.s3-eu-west-1.amazonaws.com/examen_bentoml/admissions.csv -o data/raw/admission.csv
uv run dvc repro
```

### Running it with the S3 data from dagshub

Ensure to have logged in with DVC to dagshub's S3:

```bash
uv run dvc remote modify origin --local access_key_id <your-token>
uv run dvc remote modify origin --local secret_access_key <your-token>
```

Then pull latest data and run DVC pipeline (actually nothing should happen):

```bash
uv run dvc pull
uv run dvc repro
```

## Testing bentoml

### Build and containerize bento

Build bento:

```bash
uv run bentoml build
```

Build container:
```bash
BENTO=$(uv run bentoml list  | grep 'admission_predictor_service'  | awk '{print $1}')
uv run bentoml containerize $BENTO
```


### Start API:

Either directly:

```bash
uv run bentoml serve src.service:AdmissionPredictorService --reload
```

or via docker

```bash
BENTO=$(uv run bentoml list  | grep 'admission_predictor_service'  | awk '{print $1}')
docker run --rm -p 3000:3000 $BENTO
```

### Test API manually

```bash
TOKEN=$(curl -s -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"credentials": {"username": "user123", "password": "password123"}}' | jq -r '.token')

curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "gre_score": 330.0,
    "toefl_score": 115.0,
    "cgpa": 9.5
  }'
```

### Test API by unit tests

```bash
uv run pytest -v test_api.py
```
