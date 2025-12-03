from KG_builder.models.db_engine import SessionLocal, Entity, RelationType
from numpy.typing import NDArray
from typing import List, Tuple
import numpy as np
from sqlalchemy import select 

class EntityService:
    
    @staticmethod
    def add(*,
            name: str,
            embedding: NDArray[np.float64],
    ): 
        with SessionLocal() as session:
            entity_query = select(Entity).where(Entity.name == name)
            result_entity = session.execute(entity_query).scalar_one_or_none()
            if result_entity:
                return result_entity
            
            entity = Entity(
                name = name,
                embedding = embedding
            )
            
            try:
                session.add(entity)
                session.commit()
                return entity
            except Exception as e:
                print(f"Exception in add Entity record: {e}")
    
    @staticmethod
    def query(*,
              embed: NDArray[np.float64], 
              top_k: int) -> List[Tuple[Entity, float]]:
        with SessionLocal() as session:
            distance_query = select(
                Entity,
                Entity.embedding.cosine_distance(embed).label("distance")
            ).order_by("distance").limit(top_k)
            
            result = session.execute(distance_query).all()
            
            return result


            
class RelationTypeService:
    
    @staticmethod
    def add(*,
            type: str,
            definition: str,
            embedding: NDArray[np.float64],
    ): 
        with SessionLocal() as session:
            relation_type_query = select(RelationType).where(RelationType.type == type)
            result_relation_type = session.execute(relation_type_query).scalar_one_or_none()
            if result_relation_type:
                return result_relation_type
            
            relation_type = RelationType(
                type = type,
                definition = definition,
                embedding=embedding
            )
            
            try:
                session.add(relation_type)
                session.commit()
                return relation_type
            except Exception as e:
                print(f"Exception in add Entity record: {e}")
    
    @staticmethod
    def query(*,
              embed: NDArray[np.float64], 
              top_k: int) -> List[Tuple[RelationType, float]]:
        with SessionLocal() as session:
            distance_query = select(
                RelationType,
                RelationType.embedding.cosine_distance(embed).label("distance")
            ).order_by("distance").limit(top_k)
            
            result = session.execute(distance_query).all()
            
            return result