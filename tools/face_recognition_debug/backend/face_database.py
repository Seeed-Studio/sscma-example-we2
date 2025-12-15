"""
Face Database Module
SQLite storage for face embeddings with cosine similarity matching
"""

import sqlite3
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import os


@dataclass
class FaceRecord:
    id: int
    name: str
    embedding: np.ndarray
    sample_count: int
    created_at: str


class FaceDatabase:
    def __init__(self, db_path: str = "faces.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    embedding BLOB NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()

    def add_face(self, name: str, embedding: np.ndarray, sample_count: int = 1) -> bool:
        """Add or update a face in the database"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                embedding_blob = embedding.astype(np.float32).tobytes()

                # Try to insert, update if exists
                cursor.execute('''
                    INSERT OR REPLACE INTO faces (name, embedding, sample_count, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (name, embedding_blob, sample_count, datetime.now().isoformat()))

                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Error adding face: {e}")
                return False

    def delete_face(self, name: str) -> bool:
        """Delete a face from the database"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
                affected = cursor.rowcount
                conn.commit()
                conn.close()
                return affected > 0
            except Exception as e:
                print(f"Error deleting face: {e}")
                return False

    def get_all_faces(self) -> List[FaceRecord]:
        """Get all faces from the database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, embedding, sample_count, created_at FROM faces')
            rows = cursor.fetchall()
            conn.close()

            faces = []
            for row in rows:
                embedding = np.frombuffer(row[2], dtype=np.float32)
                faces.append(FaceRecord(
                    id=row[0],
                    name=row[1],
                    embedding=embedding,
                    sample_count=row[3],
                    created_at=row[4]
                ))
            return faces

    def recognize(self, embedding: np.ndarray, threshold: float = 0.8) -> Tuple[Optional[str], float]:
        """
        Recognize a face by comparing embedding with database

        Args:
            embedding: Face embedding (128D or 512D)
            threshold: Minimum cosine similarity for positive match

        Returns:
            Tuple of (name, similarity) or (None, 0) if no match
        """
        # Normalize input
        embedding = embedding / np.linalg.norm(embedding)
        input_dim = len(embedding)

        faces = self.get_all_faces()
        if not faces:
            return None, 0.0

        best_match = None
        best_similarity = -1.0

        for face in faces:
            # Skip faces with different embedding dimensions
            if len(face.embedding) != input_dim:
                continue

            # Cosine similarity (both vectors are normalized)
            similarity = float(np.dot(embedding, face.embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face.name

        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity


class EnrollmentSession:
    """Handles face enrollment process with multiple samples"""

    def __init__(self, name: str, duration: float = 5.0, min_samples: int = 3):
        self.name = name
        self.duration = duration
        self.min_samples = min_samples
        self.embeddings: List[np.ndarray] = []
        self.start_time = datetime.now()
        self.active = True

    def add_sample(self, embedding: np.ndarray, confidence: float) -> bool:
        """Add a sample if confidence is high enough"""
        if not self.active:
            return False

        # Only accept high-confidence detections
        if confidence < 0.5:
            return False

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        self.embeddings.append(embedding)
        return True

    def is_expired(self) -> bool:
        """Check if enrollment session has expired"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return elapsed >= self.duration

    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Calculate average embedding from all samples"""
        if len(self.embeddings) < self.min_samples:
            return None

        # Stack and average
        stacked = np.stack(self.embeddings, axis=0)
        avg = np.mean(stacked, axis=0)

        # Re-normalize
        avg = avg / np.linalg.norm(avg)

        # Check consistency
        similarities = [float(np.dot(e, avg)) for e in self.embeddings]
        sim_mean = np.mean(similarities)
        sim_std = np.std(similarities)
        sim_min = np.min(similarities)

        print(f"\n[Enrollment Stats] Name: {self.name}")
        print(f"  Samples: {len(self.embeddings)}")
        print(f"  Consistency (Sim to Mean): Avg={sim_mean:.4f}, Std={sim_std:.4f}, Min={sim_min:.4f}")
        
        if sim_min < 0.8:
            print("  [WARNING] High variance detected! Outlier samples present.")

        return avg

    def get_progress(self) -> dict:
        """Get enrollment progress info"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        remaining = max(0, self.duration - elapsed)

        return {
            "name": self.name,
            "samples": len(self.embeddings),
            "min_samples": self.min_samples,
            "elapsed": elapsed,
            "remaining": remaining,
            "active": self.active and not self.is_expired()
        }

    def finalize(self) -> Optional[np.ndarray]:
        """Finalize enrollment and return average embedding"""
        self.active = False
        return self.get_average_embedding()
