from contextlib import asynccontextmanager
from databases import Database
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os


import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from asyncio import Lock

state_lock = Lock()


# uvicorn main:app --reload

load_dotenv()  # .env 파일에서 환경 변수 로드

DATABASE_URL = f"postgresql://{os.getenv('USER_NAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/{os.getenv('DATABASE')}"
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
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

male_data = []
male_data_dict = {}
male_cluster = {}
male_cluster_target = {}
male_cluster_model = None

female_data = []
female_data_dict = {}
female_cluster = {}
female_cluster_target = {}
female_cluster_model = None


def generate_df_data(data):
    df = pd.DataFrame(data)

    if "member_features" in df.columns:
        features = (
            df["member_features"]
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
        df = pd.concat([df, dummies], axis=1).drop("member_features", axis=1)

    return df


def convert_fit_data(df_data):
    result = df_data.drop(
        columns=["nickname", "member_id", "gender", "card_type"], axis=1
    )
    return result


def member_cosine_similarity(member_id1, member_id2, gender):
    data = male_data_dict
    if gender in ("FEMALE", "female"):
        data = female_data_dict

    if member_id1 not in data or member_id2 not in data:
        return None

    user1 = data[member_id1]
    user2 = data[member_id2]

    user_df_data = generate_df_data([user1, user2])

    user_fit_data = convert_fit_data(user_df_data)

    result = cosine_similarity(user_fit_data)
    return result[0][1].item()


async def fetch_data():
    async with state_lock:
        query = """
                SELECT member_id, member_features, birth_year, gender, nickname, 'my' AS card_type
                FROM member_account
                JOIN member_card ON member_account.my_card_id = member_card.member_card_id
                UNION ALL
                SELECT member_id, member_features, birth_year, gender, nickname, 'mate' AS card_type
                FROM member_account
                JOIN member_card ON member_account.mate_card_id = member_card.member_card_id
                """
        all_data = [dict(record) for record in await database.fetch_all(query)]

        global male_data, female_data
        male_data = [user for user in all_data if user["gender"].lower() == "male"]
        female_data = [user for user in all_data if user["gender"].lower() == "female"]

        global male_data_dict, female_data_dict
        male_data_dict = {user["member_id"]: user for user in male_data}
        female_data_dict = {user["member_id"]: user for user in female_data}


async def clustering():
    global male_cluster, male_cluster_model, male_cluster_target, female_cluster, female_cluster_model, female_cluster_target
    async with state_lock:
        male_df_data = generate_df_data(male_data)
        female_df_data = generate_df_data(female_data)

        male_cluster_model = DBSCAN(eps=0.2, min_samples=2)
        male_cluster_model.fit(convert_fit_data(male_df_data))

        female_cluster_model = DBSCAN(eps=0.2, min_samples=2)
        female_cluster_model.fit(convert_fit_data(female_df_data))

        male_cluster = {}
        male_cluster_target = {}
        for index, cluster in enumerate(
            male_cluster_model.fit_predict(convert_fit_data(male_df_data))
        ):
            cluster = cluster.item()
            user = male_data[index]["member_id"]
            card_type = male_data[index]["card_type"]

            male_cluster_target[user] = cluster
            if cluster in male_cluster:
                male_cluster[cluster].append((user, card_type))
            else:
                male_cluster[cluster] = [(user, card_type)]

        female_cluster = {}
        female_cluster_target = {}
        for index, cluster in enumerate(
            female_cluster_model.fit_predict(convert_fit_data(female_df_data))
        ):
            cluster = cluster.item()
            user = female_data[index]["member_id"]
            card_type = female_data[index]["card_type"]

            female_cluster_target[user] = cluster
            if cluster in female_cluster:
                female_cluster[cluster].append((user, card_type))
            else:
                female_cluster[cluster] = [(user, card_type)]


@app.get("/update")
async def update():
    await fetch_data()
    await clustering()
    return {"detail": "ok"}


@app.get("/recommendation/{member_id}/{card_type}")
async def recommendation(member_id, card_type):
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

    if user_gender in ("male", "MALE"):
        if user_id not in male_cluster_target:
            await fetch_data()
            await clustering()

        recommendation = []
        for other, other_card_type in male_cluster[male_cluster_target[user_id]]:
            if card_type == other_card_type or user_id == other:
                continue
            similarity = member_cosine_similarity(user_id, other, user_gender)
            recommendation.append(
                {
                    "userId": other,
                    "name": male_data_dict[other]["nickname"],
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
        for other, other_card_type in female_cluster[female_cluster_target[user_id]]:
            if card_type == other_card_type or user_id == other:
                continue
            similarity = member_cosine_similarity(user_id, other, user_gender)
            recommendation.append(
                {
                    "userId": other,
                    "name": female_data_dict[other]["nickname"],
                    "similarity": similarity,
                    "cardType": other_card_type,
                }
            )

        return {
            "user": {"id": user_id, "gender": user_gender},
            "recommendation": recommendation,
        }
