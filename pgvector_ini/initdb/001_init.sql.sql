CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS relation_types;
DROP TABLE IF EXISTS entity;

-- Table for standardized relation types (unique by "type")
CREATE TABLE IF NOT EXISTS relation_types (
  id           BIGSERIAL PRIMARY KEY,
  type         TEXT UNIQUE NOT NULL,
  definition   TEXT NOT NULL,
  embedding    VECTOR(896) NOT NULL
);

CREATE TABLE IF NOT EXISTS entity (
  id        BIGSERIAL PRIMARY KEY,
  name      TEXT UNIQUE NOT NULL,
  embedding VECTOR(896) NOT NULL
);

CREATE INDEX ON relation_types
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX ON entity
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
