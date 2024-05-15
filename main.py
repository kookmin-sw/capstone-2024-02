import ast
import json
from contextlib import asynccontextmanager
import time
from tracemalloc import start
from typing import List
from urllib import request
from databases import Database
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from google.cloud import firestore
import os

import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from asyncio import Lock
from collections import defaultdict
from sklearn.impute import SimpleImputer

from request import Recommend

state_lock = Lock()


# uvicorn main:app --reload

load_dotenv()  # .env 파일에서 환경 변수 로드

class SimilarityItem(BaseModel):
    id: str
    similarity: float


class UserCategory(BaseModel):
    my: List[SimilarityItem]
    mate: List[SimilarityItem]


class PostCategory(BaseModel):
    my: List[SimilarityItem]
    mate: List[SimilarityItem]


class DataModel(BaseModel):
    user: UserCategory
    post: PostCategory

# DATABASE_URL = "postgresql://cheesecrust:0810jack@mydatabase.c3kmc4wcyz81.ap-northeast-2.rds.amazonaws.com/maru"
DATABASE_URL = "postgresql://localhost:5432/maru"
database = Database(DATABASE_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://capstone-maru.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(data):

    if data is None:
        return '{"options": "[]", "smoking": "상관없어요", "room_sharing_option": "상관없어요"}'

    if data == "null":
        return '{"options": "[]", "smoking": "상관없어요", "room_sharing_option": "상관없어요"}'

    parsed_data = json.loads(data)
    # "options" 키의 값 추출
    options_value = parsed_data['options']

    # 문자열에서 배열로 변환

    options_array = []

    if options_value is not None:
        options_array = ast.literal_eval(options_value)
    
    result_array = []
    for value in options_array:
        if isinstance(value, list):
            result_array.append(str(value))
        # elif isinstance(value, int):
        #     print("int value : ", value)
        #     result_array.extend(str(value))
        else:
            result_array.append(value)
    # 추가할 키의 값들 추출
    smoking_value = parsed_data['smoking']
    mate_age_value = None
    if mate_age_value in parsed_data:
        mate_age_value = parsed_data['mate_age']
    room_sharing_option_value = parsed_data['room_sharing_option']

    # 추가할 값들을 배열에 추가
    result_array.extend([smoking_value, room_sharing_option_value])
    if mate_age_value is not None:
        result_array.append(mate_age_value)

    return result_array

def generate_df_data(data):
    df = pd.DataFrame(data)

    if "features" in df.columns:

        df["features"] = df["features"].apply(extract_features)
        
        features = (
            df["features"]
            .apply(pd.Series)
            .stack()
            .reset_index(level=1, drop=True)
            .to_frame("features")
        )
        dummies = (
            pd.get_dummies(features, prefix="", prefix_sep="").groupby(level=0).sum()
        )

        if "[]" in dummies:
            dummies.drop("[]", axis=1, inplace=True)
        if "null" in dummies:
            dummies.drop("null", axis=1, inplace=True)
        
        for column in dummies.columns:
            if column.startswith("[") and column.endswith("]"):
                dummies.drop(column, axis=1, inplace=True)
        
        df = pd.concat([df, dummies], axis=1).drop("features", axis=1)

    return df


def convert_fit_data(df, columns=["id", "gender", "card_type", 'birth_year', 'location']):
    result = df.drop(columns=columns, axis=1)
    return result


def feature_card_cosine_similarity(card1, card2):
    user_df = generate_df_data([card1, card2])
    user_fit_df = convert_fit_data(user_df)
    result = cosine_similarity(user_fit_df)
    return result[0][1].item()


async def fetch_data(user_id, card_type, want_to_find):

    location_cluster = [{'강서구', '양천구'}, {'구로구', '영등포구', '금천구'}, {'동작구', '관악구'}, {'서초구', '강남구'},
               {'송파구', '강동구'}, {'은평구', '서대문구', '마포구'}, {'종로구', '중구', '용산구'}, {'중랑구', '동대문구', '성동구', '광진구'},
               {'성북구', '강북구', '도봉구', '노원구'}]

    query = f"""
            SELECT member_id AS id, location, member_features::jsonb AS features, gender, '{card_type}' AS card_type, birth_year
            FROM member_account
            JOIN feature_card
            ON member_account.{card_type}_card_id = feature_card.feature_card_id
            WHERE '{user_id}' = member_id
            """
    
    user = await database.fetch_one(query)
    user_location = user['location'].split()[1]
    user_gender = user['gender']

    cluster_index = -1
    for i, cluster in enumerate(location_cluster):
        if user_location in cluster:
            cluster_index = i
            break

    cards = []

    if want_to_find == 'member':
        for location in location_cluster[cluster_index]:
            if card_type == 'my':
                query = f"""
                        SELECT member_id as id, location, member_features::jsonb AS features, gender, 'mate' AS card_type, birth_year
                        FROM member_account
                        JOIN feature_card 
                        ON member_account.mate_card_id = feature_card.feature_card_id
                        WHERE location like '%{location}%' and gender IN ('{user_gender.lower()}', '{user_gender.upper()}')
                        """ 
                cards.extend([dict(record) for record in await database.fetch_all(query)])

            elif card_type == 'mate':
                query = f"""
                        SELECT member_id AS id, location, member_features::jsonb AS features, gender, 'my' AS card_type, birth_year
                        FROM member_account
                        JOIN feature_card 
                        ON member_account.my_card_id = feature_card.feature_card_id
                        WHERE location like '%{location}%' and gender IN ('{user_gender.lower()}', '{user_gender.upper()}')
                        """
                cards.extend([dict(record) for record in await database.fetch_all(query)])

    if want_to_find == 'post':
        query = f"""
                SELECT id, location, member_features::jsonb AS features, gender, 'room' AS card_type, member_account.birth_year
                FROM shared_room_post
                JOIN feature_card ON shared_room_post.room_mate_card_id = feature_card.feature_card_id
                JOIN member_account ON member_account.member_id = shared_room_post.publisher_id
                WHERE location like '%{location}%' and gender IN ('{user_gender.lower()}', '{user_gender.upper()}')
                """
        cards.extend([dict(record) for record in await database.fetch_all(query)])

    return cards, dict(user)

def fill_missing_values(df):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(df)

async def clustering(cards, user_card):

    if cards == []:
        cards = [{'id': 'male_default', 'features': None, 'gender': 'MALE', 'card_type': 'my', 'birth_year': '1999'}, 
                      {'id': 'male_default', 'features': None, 'gender': 'MALE', 'card_type': 'mate', 'birth_year': '1999'}]

    total_recommendation_result = {"user": {"my": [], "mate": []}, "post": {"my": [], "mate": []}}

    for card in cards:
        if (
            card['card_type'] == user_card['card_type']
            or (card['card_type'] != 'room' and card['id'] == user_card['id'])
        ):
            continue

        similarity = feature_card_cosine_similarity(user_card, card)

        if card["card_type"] != "room":
            total_recommendation_result["user"][user_card["card_type"]].append(
                {
                    "id": card["id"],
                    "score": similarity,
                    "cardType":  card["card_type"],
                }
            )
        else:
            total_recommendation_result["post"][user_card["card_type"]].append(
                {
                    "id": card["id"],
                    "score": similarity,
                    "cardType":  card["card_type"],
                }
            )

    print("complete recommendation")
    # print("male key : ", male_recommendation_result.keys())
    # print("male result : ", male_recommendation_result.values())
    """
    user_id <-> id, user_card_type, score, id_type
    """

    await database.execute("truncate table recommend")
    # print(male_recommendation_result)
    # print(female_recommendation_result)

    query = """
        insert into recommend (user_id, card_type, recommendation_id, recommendation_card_type, score)
        values (:user_id, :card_type, :recommendation_id, :recommendation_card_type, :score)
        """

    for card_data in total_recommendation_result["user"]["my"]:
        recommendation_id = card_data["id"]
        recommendation_card_type = card_data["cardType"]
        score = card_data["score"] * 100

        await database.execute(query, 
                               {"user_id": user_card["id"], "card_type": user_card["card_type"], 
                                "recommendation_id": recommendation_id, "recommendation_card_type": recommendation_card_type,
                                "score": score
                                })

    for card_data in total_recommendation_result["user"]["mate"]:
        recommendation_id = card_data["id"]
        recommendation_card_type = card_data["cardType"]
        score = card_data["score"] * 100

        await database.execute(query, 
                               {"user_id": user_card["id"], "card_type": user_card["card_type"], 
                                "recommendation_id": recommendation_id, "recommendation_card_type": recommendation_card_type,
                                "score": score
                                })

    for card_data in total_recommendation_result["post"]["my"]:
        recommendation_id = str(card_data["id"])
        recommendation_card_type = card_data["cardType"]
        score = card_data["score"] * 100

        await database.execute(query, 
                               {"user_id": user_card["id"], "card_type": user_card["card_type"], 
                                "recommendation_id": recommendation_id, "recommendation_card_type": recommendation_card_type,
                                "score": score
                                })

    for card_data in total_recommendation_result["post"]["mate"]:
        recommendation_id = str(card_data["id"])
        recommendation_card_type = card_data["cardType"]
        score = card_data["score"] * 100

        await database.execute(query, 
                               {"user_id": user_card["id"], "card_type": user_card["card_type"], 
                                "recommendation_id": recommendation_id, "recommendation_card_type": recommendation_card_type,
                                "score": score
                                })


@app.get("/")
async def root():
    return {"detail": "ok"}


@app.post("/recommendation/update")
async def update(requset: Recommend):
    start = time.time()
    cards, user_card = (
        await fetch_data(requset.user_id, requset.card_type, requset.want_to_find)
    )
    
    print('fetch complete')

    await clustering(cards, user_card)

    print("clustering complete")
    print("time : ", time.time() - start)
    return {"detail": "ok"}

@app.get("/fetch")
async def fetch():
    cards, user_card = (
        await fetch_data("kakao_0", "my", "member")
    )

    print(cards)
    print(generate_df_data(cards))

@app.get("/test")
async def test():
    await fetch_data("kakao_0", "my")