import ast
import json
from contextlib import asynccontextmanager
from typing import List
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

from asyncio import Lock
from collections import defaultdict
from sklearn.impute import SimpleImputer

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

DATABASE_URL = "postgresql://cheesecrust:0810jack@mydatabase.c3kmc4wcyz81.ap-northeast-2.rds.amazonaws.com/maru"
# DATABASE_URL = "postgresql://localhost:5432/maru"
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
        return []

    parsed_data = json.loads(data)
    # "options" 키의 값 추출
    options_value = parsed_data['options']

    # 문자열에서 배열로 변환
    options_array = json.loads(options_value)

    # 추가할 키의 값들 추출
    smoking_value = parsed_data['smoking']
    mate_age_value = parsed_data['mate_age']
    room_sharing_option_value = parsed_data['room_sharing_option']

    # 추가할 값들을 배열에 추가
    options_array.extend([smoking_value, mate_age_value, room_sharing_option_value])
    return options_array

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

        df = pd.concat([df, dummies], axis=1).drop("features", axis=1)

    return df


def convert_fit_data(df, columns=["id", "gender", "card_type"]):
    result = df.drop(columns=columns, axis=1)
    return result


def feature_card_cosine_similarity(card1, card2):
    user_df = generate_df_data([card1, card2])
    user_fit_df = convert_fit_data(user_df)
    result = cosine_similarity(user_fit_df)
    return result[0][1].item()


async def fetch_data():
    
    query = """
            SELECT member_id AS id, member_features::jsonb AS features, gender, 'my' AS card_type, birth_year
            FROM member_account
            JOIN feature_card ON member_account.my_card_id = feature_card.feature_card_id
            UNION ALL
            SELECT member_id as id, member_features::jsonb AS features, gender, 'mate' AS card_type, birth_year
            FROM member_account
            JOIN feature_card ON member_account.mate_card_id = feature_card.feature_card_id
            """
    user_cards = [dict(record) for record in await database.fetch_all(query)]

    query = """
            SELECT id, member_features::jsonb AS features, gender, 'room' AS card_type, member_account.birth_year
            FROM shared_room_post
            JOIN feature_card ON shared_room_post.room_mate_card_id = feature_card.feature_card_id
            JOIN member_account ON member_account.member_id = shared_room_post.publisher_id
            """
    post_cards = [dict(record) for record in await database.fetch_all(query)]

    user_male_cards = []
    user_female_cards = []
    for record in user_cards:
        user_gender = record["gender"]

        cards = user_male_cards
        if user_gender.lower() == "female":
            cards = user_female_cards
        cards.append(record)

    post_male_cards = []
    post_female_cards = []
    for record in post_cards:
        user_gender = record["gender"]

        cards = post_male_cards
        if user_gender.lower() == "female":
            cards = post_female_cards
        cards.append(record)

    return user_male_cards, user_female_cards, post_male_cards, post_female_cards

def fill_missing_values(df):
    print("fill_missing_values")
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(df)

def clustering(user_male_cards, user_female_cards, post_male_cards, post_female_cards):

    male_cards = [*user_male_cards, *post_male_cards]
    male_df = generate_df_data(male_cards)
    print(male_df)

    female_cards = [*user_female_cards, *post_female_cards]
    female_df = generate_df_data(female_cards)
    

    # 여기의 fit 이 뭔데 값으로??
    # 결측값 우선 처리

    male_cluster_model = DBSCAN(eps=0.2, min_samples=2)
    male_cluster_model.fit(
        convert_fit_data(male_df)
    )
    
    female_cluster_model = DBSCAN(eps=0.2, min_samples=2)
    female_cluster_model.fit(
        convert_fit_data(female_df)
    )
    
    male_cluster = defaultdict(lambda: [])

    find_male_user_cluster = defaultdict(lambda: {"my": None, "mate": None})

    for index, cluster in enumerate(
        male_cluster_model.fit_predict(convert_fit_data(male_df))
    ):
        print(cluster)
        card = male_cards[index]

        male_cluster[cluster].append(male_cards[index])
        if card["card_type"] != "room":
            user_id = card["id"]
            find_male_user_cluster[user_id][card["card_type"]] = cluster

    female_cluster = defaultdict(lambda: [])
    find_female_user_cluster = defaultdict(lambda: {"my": None, "mate": None})
    for index, cluster in enumerate(
        female_cluster_model.fit_predict(convert_fit_data(female_df))
    ):
        card = female_cards[index]

        female_cluster[cluster].append(female_cards[index])
        if card["card_type"] != "room":
            user_id = card["id"]
            find_female_user_cluster[user_id][card["card_type"]] = cluster

    male_recommendation_result = defaultdict(
        lambda: {"user": {"my": [], "mate": []}, "post": {"my": [], "mate": []}}
    )

    for cluster, cluster_item in male_cluster.items():
        for i, card in enumerate(cluster_item):
            if card["card_type"] == "room":
                continue

            user_id = card["id"]
            card_type = card["card_type"]
            for j, other_card in enumerate(cluster_item):
                if (
                    i == j
                    # or card_type == other_card["card_type"]
                    # or user_id == other_card["id"]
                ):
                    continue

                similarity = feature_card_cosine_similarity(card, other_card)
                if other_card["card_type"] != "room":
                    male_recommendation_result[user_id]["user"][card_type].append(
                        {
                            "id": other_card["id"],
                            "score": similarity,
                            "cardType": other_card["card_type"],
                        }
                    )
                else:
                    male_recommendation_result[user_id]["post"][card_type].append(
                        {
                            "id": other_card["id"],
                            "score": similarity,
                            "cardType": other_card["card_type"],
                        }
                    )

    female_recommendation_result = defaultdict(
        lambda: {"user": {"my": [], "mate": []}, "post": {"my": [], "mate": []}}
    )
    for cluster, cluster_item in female_cluster.items():
        for i, card in enumerate(cluster_item):
            if card["card_type"] == "room":
                continue

            user_id = card["id"]
            card_type = card["card_type"]
            for j, other_card in enumerate(cluster_item):
                if (
                    i == j
                    # or card_type == other_card["card_type"]
                    # or user_id == other_card["id"]
                ):
                    continue

                similarity = feature_card_cosine_similarity(card, other_card)
                if other_card["card_type"] != "room":
                    female_recommendation_result[user_id]["user"][card_type].append(
                        {
                            "id": other_card["id"],
                            "score": similarity,
                            "cardType": other_card["card_type"],
                        }
                    )
                else:
                    female_recommendation_result[user_id]["post"][card_type].append(
                        {
                            "id": other_card["id"],
                            "score": similarity,
                            "cardType": other_card["card_type"],
                        }
                    )
    print(male_recommendation_result)
    print(female_recommendation_result)
    
    # 여기에 insert


@app.get("/")
async def root():
    return {"detail": "ok"}


@app.get("/recommendation/update")
async def update():
    user_male_cards, user_female_cards, post_male_cards, post_female_cards = (
        await fetch_data()
    )

    clustering(user_male_cards, user_female_cards, post_male_cards, post_female_cards)
    print("clustering complete")

    return {"detail": "ok"}

@app.get("/fetch")
async def fetch():
    user_male_cards, user_female_cards, post_male_cards, post_female_cards = (
        await fetch_data()
    )
    # print("user male cards : ", user_male_cards)
    # print()
    # print("user female cards : ", user_female_cards)
    # print()
    # print(generate_df_data(user_male_cards))
    # print()
    print(user_female_cards)
    print(generate_df_data(user_female_cards))

@app.get("/insert")
async def insert():
    query = """
            insert into recommend (id, user_id, card_type, recommendation_id, recommendation_card_type, score)
            values (2,'test', 'test', 'test', 'tset', 100)
            """
    
    await database.execute(query)