import ast
from contextlib import asynccontextmanager
from typing import List
from xml.etree.ElementInclude import default_loader
from databases import Database
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import firestore
import os

import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from asyncio import Lock
from collections import defaultdict

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


DATABASE_URL = f"postgresql://{os.getenv('USER_NAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/{os.getenv('DATABASE')}"
database = Database(DATABASE_URL)
nosql_database = firestore.Client(database="maru")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()
    nosql_database.close()


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


def generate_df_data(data):
    df = pd.DataFrame(data)

    if "features" in df.columns:
        df["features"] = df["features"].apply(ast.literal_eval)

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


def convert_fit_data(df_data, columns=["nickname", "id", "gender", "card_type"]):
    result = df_data.drop(columns=columns, axis=1)
    return result


def user_cosine_similarity(user1, user2):
    user_df_data = generate_df_data([user1, user2])
    user_fit_data = convert_fit_data(user_df_data)
    result = cosine_similarity(user_fit_data)
    return result[0][1].item()


async def fetch_data():
    async with state_lock:
        query = """
                SELECT member_id as id, features, birth_year, gender, nickname, 'my' AS card_type
                FROM member_account
                JOIN feature_card ON member_account.my_card_id = feature_card.feature_card_id
                UNION ALL
                SELECT member_id as id, features, birth_year, gender, nickname, 'mate' AS card_type
                FROM member_account
                JOIN feature_card ON member_account.mate_card_id = feature_card.feature_card_id
                """
        user_card_data = [dict(record) for record in await database.fetch_all(query)]

        query = """
                SELECT id, features, birth_year, gender, nickname, 'room' AS card_type
                FROM shared_room_post
                JOIN feature_card ON shared_room_post.room_mate_card_id = feature_card.feature_card_id
                JOIN member_account ON member_account.member_id = shared_room_post.publisher_id
                """
        post_card_data = [dict(record) for record in await database.fetch_all(query)]

        user_male_cards = []
        user_female_cards = []
        for record in user_card_data:
            user_gender = record["gender"]

            cards = user_male_cards
            if user_gender.lower() == "female":
                cards = user_female_cards
            cards.append(record)

        post_male_cards = []
        post_female_cards = []
        for record in post_card_data:
            user_gender = record["gender"]

            cards = post_male_cards
            if user_gender.lower() == "female":
                cards = post_female_cards
            cards.append(record)


async def clustering(
    user_male_cards, user_female_cards, post_male_cards, post_female_cards
):
    async with state_lock:
        male_data = [*user_male_cards, *post_male_cards]
        male_df_data = generate_df_data(male_data)

        female_data = [*user_female_cards, *post_female_cards]
        female_df_data = generate_df_data(female_data)

        male_cluster_model = DBSCAN(eps=0.2, min_samples=2)
        male_cluster_model.fit(convert_fit_data(male_df_data))

        female_cluster_model = DBSCAN(eps=0.2, min_samples=2)
        female_cluster_model.fit(convert_fit_data(female_df_data))

        male_dict = defaultdict(lambda: {})
        male_cluster = defaultdict(lambda: [])
        male_user_cluster_id = defaultdict(lambda: {"my": -1, "mate": -1})
        for index, cluster in enumerate(
            male_cluster_model.fit_predict(convert_fit_data(male_df_data))
        ):
            cluster = cluster.item()

            user_id = male_data[index]["id"]
            card_type = male_data[index]["card_type"]

            if card_type != "room":
                male_dict[user_id][card_type] = male_data[index]

            if card_type == "my":
                male_user_cluster_id[user_id]["my"] = cluster
            elif card_type == "mate":
                male_user_cluster_id[user_id]["mate"] = cluster
            male_cluster[cluster].append(male_data[index])

        female_dict = defaultdict(lambda: {})
        female_cluster = defaultdict(lambda: [])
        female_user_cluster_id = defaultdict(lambda: {"my": -1, "mate": -1})
        for index, cluster in enumerate(
            female_cluster_model.fit_predict(convert_fit_data(female_df_data))
        ):
            cluster = cluster.item()

            user_id = female_data[index]["id"]
            card_type = female_data[index]["card_type"]

            if card_type != "room":
                female_dict[user_id][card_type] = female_data[index]

            if card_type == "my":
                female_user_cluster_id[user_id]["my"] = cluster
            elif card_type == "mate":
                female_user_cluster_id[user_id]["mate"] = cluster
            female_cluster[cluster].append(female_data[index])

        male_similarity = defaultdict(
            lambda: {"user": {"my": [], "mate": []}, "post": {"my": [], "mate": []}}
        )
        for user_id in male_user_cluster_id:
            my_card_id = male_dict[user_id]["my"]
            mate_card_id = male_dict[user_id]["mate"]

            for other_id in male_cluster[male_user_cluster_id[user_id]["my"]]:
                pass

            for other_id in male_cluster[male_user_cluster_id[user_id]["mate"]]:
                pass


@app.get("/")
async def root():
    await fetch_data()
    return {"detail": "ok"}


@app.get("/recommendation/update")
async def update():
    await fetch_data()
    await clustering()
    return {"detail": "ok"}


@app.get("/recommendation/{member_id}/{card_type}")
async def recommendation(member_id, card_type, page: int, size: int = 10):
    if male_cluster_model == None or female_cluster_model == None:
        await fetch_data()
        await clustering()

    query = f"SELECT member_id, gender, nickname FROM member_account WHERE member_id = '{member_id}'"

    record = await database.fetch_one(query)
    if record == None:
        raise HTTPException(status_code=400, detail="User not found")

    user = dict(record)
    user_id = user["member_id"]
    user_gender = user["gender"]
    find_card_type = "mate" if card_type == "my" else "my"

    if user_gender in ("male", "MALE"):
        if user_id not in male_cluster_target:
            await fetch_data()
            await clustering()

        recommendation = []
        if (user_id, find_card_type) in male_similarity:
            for other_user_id, similarity, other_card_type in male_similarity[
                (user_id, find_card_type)
            ][page * size : (page + 1) * size]:
                recommendation.append(
                    {
                        "userId": other_user_id,
                        "similarity": similarity,
                        "cardType": other_card_type,
                    }
                )

        return {
            "user": {"id": user_id, "gender": user_gender},
            "recommendation": recommendation,
        }
    else:
        if user_id not in female_cluster_target:
            await fetch_data()
            await clustering()

        recommendation = []
        if (user_id, find_card_type) in female_similarity:
            for other_user_id, similarity, other_card_type in female_similarity[
                (user_id, find_card_type)
            ][page * size : (page + 1) * size]:
                recommendation.append(
                    {
                        "userId": other_user_id,
                        "similarity": similarity,
                        "cardType": other_card_type,
                    }
                )

        return {
            "user": {"id": user_id, "gender": user_gender},
            "recommendation": recommendation,
        }
