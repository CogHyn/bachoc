import os
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
DATABASE_URL = os.getenv("DB_URI", "")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 896))
engine = sa.create_engine(DATABASE_URL)

class RelationType(Base):
    __tablename__ = "relation_types"
    
    id = sa.Column(sa.BigInteger, primary_key=True)
    type = sa.Column(sa.String, nullable=False)
    definition = sa.Column(sa.String, nullable=False)
    embedding = sa.Column(Vector(VECTOR_DIM))
    
class Entity(Base):
    __tablename__ = "entity"
    
    id = sa.Column(sa.BigInteger, primary_key=True)
    name = sa.Column(sa.String, nullable=False)
    embedding = sa.Column(Vector(VECTOR_DIM))

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)