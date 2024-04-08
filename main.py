from contextlib import asynccontextmanager
from databases import Database
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os


import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# uvicorn main:app --reload

load_dotenv()  # .env 파일에서 환경 변수 로드

DATABASE_URL = os.getenv("DATABASE_URL")


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
    result = df_data.drop(columns=["member_id", "gender"], axis=1)
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
    global male_data, male_data_dict, female_data, female_data_dict

    query = """
            SELECT member_id, member_features, birth_year, gender
            FROM member_account
            JOIN member_card ON member_account.my_card_id = member_card.member_card_id
            WHERE gender ILIKE 'male'
            """
    male_data = [dict(record) for record in await database.fetch_all(query)]
    male_data_dict = {user["member_id"]: user for user in male_data}

    query = """
            SELECT member_id, member_features, birth_year, gender
            FROM member_account
            JOIN member_card ON member_account.my_card_id = member_card.member_card_id
            WHERE gender ILIKE 'female'
            """
    female_data = [dict(record) for record in await database.fetch_all(query)]
    female_data_dict = {user["member_id"]: user for user in female_data}


async def clustering():
    global male_cluster, male_cluster_model, male_cluster_target, female_cluster, female_cluster_model, female_cluster_target

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

        male_cluster_target[user] = cluster
        if cluster in male_cluster:
            male_cluster[cluster].append(user)
        else:
            male_cluster[cluster] = [user]

    female_cluster = {}
    female_cluster_target = {}
    for index, cluster in enumerate(
        female_cluster_model.fit_predict(convert_fit_data(female_df_data))
    ):
        cluster = cluster.item()
        user = female_data[index]["member_id"]

        female_cluster_target[user] = cluster
        if cluster in female_cluster:
            female_cluster[cluster].append(user)
        else:
            female_cluster[cluster] = [user]


@app.get("/update")
async def update():
    await fetch_data()
    await clustering()
    return {"detail": "ok"}


@app.get("/recommendation/{member_id}")
async def recommendation(member_id):
    if male_cluster_model == None or female_cluster_model == None:
        await fetch_data()
        await clustering()

    query = (
        f"SELECT member_id, gender FROM member_account WHERE member_id = '{member_id}'"
    )

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
        for other in male_cluster[male_cluster_target[user_id]]:
            similarity = member_cosine_similarity(user_id, other, user_gender)
            recommendation.append({"userId": other, "similarity": similarity})

        return {
            "user": {"id": user_id, "gender": user_gender},
            "recommendation": recommendation,
        }
    else:
        if user_id not in female_cluster_target:
            await fetch_data()
            await clustering()

        recommendation = []
        for other in female_cluster[female_cluster_target[user_id]]:
            similarity = member_cosine_similarity(user_id, other, user_gender)
            recommendation.append({"user_id": other, "similarity": similarity})

        return {
            "user": {"id": user_id, "gender": user_gender},
            "recommendation": recommendation,
        }
