FROM python:3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# docker build --no-cache -t maru-rec-sys-image .
# docker run -d --name maru-rec-sys-container -p 8000:8000 maru-rec-sys-image

# 만약, 직접 들어가보고 싶으면.
# docker exec -it maru-rec-sys-container /bin/bash 

# 아래 처럼 가상환경 만들고 해당 환경으로 requirements.txt 생성함
# python3 -m venv maru-rec-sys-env
# source maru-rec-sys-env/bin/activate
