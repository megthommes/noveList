FROM python:3.7
MAINTAINER meghan.thommes@gmail.com

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

# Expose port:
EXPOSE $PORT

# Run application:
ENTRYPOINT "streamlit run noveList.py --server.port $PORT"