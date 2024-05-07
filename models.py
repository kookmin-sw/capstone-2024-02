from sqlalchemy import Column, Integer, String, Text, DateTime

from database import Base, engine

# recommendation_collection = recommendation_database.collection("recommendation")
# for male_user_id, recommendation_result in male_recommendation_result.items():
#     doc_ref = recommendation_collection.document(f"{male_user_id}")
#     doc_ref.set(recommendation_result)
# for female_user_id, recommendation_result in female_recommendation_result.items():
#     doc_ref = recommendation_collection.document(f"{female_user_id}")
#     doc_ref.set(recommendation_result)

class recommendation(Base):
    __tablename__ = "recommendation"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    recommendation = Column(Text)
    # created_at = Column(DateTime)
    # updated_at = Column(DateTime)