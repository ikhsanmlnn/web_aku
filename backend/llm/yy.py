"""
Course Recommender menggunakan Sentence Transformers
Model: all-MiniLM-L6-v2 (lebih ringan & cepat)
Disesuaikan dengan notebook ML
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None
    torch = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None


# Model: all-MiniLM-L6-v2 (384 dimensions, fast)
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = Path(__file__).parent / "models" / "course_embeddings.pt"


def clean_text(text) -> str:
    """Clean text untuk preprocessing - sesuai notebook"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("-", " ")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class CourseRecommender:
    """
    Course Recommender dengan Sentence Transformer
    Model: all-MiniLM-L6-v2 (sesuai notebook)
    """
    
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.lp_combined = None
        
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(MODEL_NAME)
                print(f"✅ Loaded model: {MODEL_NAME}")
            except Exception as e:
                print(f"Failed to load Sentence Transformer: {e}")
    
    def load_data(self, lp_answer_df: pd.DataFrame, course_df: pd.DataFrame):
        """
        Load dan prepare data sesuai notebook
        
        Args:
            lp_answer_df: DataFrame Learning Path Answer
            course_df: DataFrame Course
        """
        # Mapping learning path - PERSIS NOTEBOOK
        learning_path_mapping = {
            1: "AI Engineer",
            2: "Android Developer",
            3: "Back-End Developer JavaScript",
            4: "Back-End Developer Python",
            5: "Data Scientist",
            6: "DevOps Engineer",
            7: "Front-End Web Developer",
            8: "Gen AI Engineer",
            9: "Google Cloud Professional",
            10: "iOS Developer",
            11: "MLOps Engineer",
            12: "Multi-Platform App Developer",
            13: "React Developer"
        }
        
        # Mapping level - PERSIS NOTEBOOK
        course_level_mapping = {
            1: "Beginner",
            2: "Beginner",
            3: "Intermediate",
            4: "Advanced",
            5: "Advanced"
        }
        
        course_df = course_df.copy()
        course_df['learning_path'] = course_df['learning_path_id'].map(learning_path_mapping)
        course_df['course_level'] = course_df['course_level_str'].map(course_level_mapping)
        
        # Merge data - PERSIS NOTEBOOK
        self.lp_combined = lp_answer_df.merge(
            course_df[['course_name', 'course_level', 'learning_path']],
            left_on='name',
            right_on='course_name',
            how='left'
        )
        
        # Combined text untuk embedding - PERSIS NOTEBOOK
        self.lp_combined['combined_text'] = (
            self.lp_combined['learning_path'].fillna('') + " " +
            self.lp_combined['course_level'].fillna('') + " " +
            self.lp_combined['course_name'].fillna('') + " " +
            self.lp_combined['description'].fillna('') + " " +
            self.lp_combined['technologies'].fillna('') + " "
        ).apply(clean_text)
        
        print(f"✅ Loaded {len(self.lp_combined)} courses")
        
        # Build embeddings
        if self.model:
            self._build_embeddings()
    
    def _build_embeddings(self):
        """
        Build embeddings untuk semua course
        """
        if self.lp_combined is None:
            return
        
        texts = self.lp_combined['combined_text'].tolist()
        print(f"Building embeddings for {len(texts)} courses...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print("✅ Embeddings built")
    
    def save_embeddings(self, path: Optional[Path] = None):
        """
        Save embeddings ke file .pt
        """
        if self.embeddings is None or torch is None:
            print("No embeddings to save or torch not available")
            return
        
        save_path = path or EMBEDDINGS_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.embeddings, save_path)
        print(f"✅ Embeddings saved to {save_path}")
    
    def load_embeddings(self, path: Optional[Path] = None):
        """
        Load embeddings dari file .pt
        
        Returns:
            True jika berhasil, False jika gagal
        """
        load_path = path or EMBEDDINGS_PATH
        
        if not load_path.exists():
            print(f"Embeddings file not found: {load_path}")
            return False
        
        if torch is None:
            print("torch not available")
            return False
        
        try:
            self.embeddings = torch.load(load_path)
            print(f"✅ Embeddings loaded from {load_path}")
            return True
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            return False
    
    def recommend(
        self, 
        user_input: str, 
        user_level: Optional[str] = None, 
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Rekomendasi course berdasarkan user input - PERSIS NOTEBOOK
        
        Args:
            user_input: Deskripsi user (job role, skills, dll)
            user_level: Level user (Beginner/Intermediate/Advanced)
            top_k: Jumlah rekomendasi
        
        Returns:
            DataFrame dengan top-K course
        """
        if not SENTENCE_TRANSFORMER_AVAILABLE or self.model is None or self.embeddings is None:
            return pd.DataFrame()
        
        # Combine user_level jika ada - PERSIS NOTEBOOK
        if user_level:
            final_input = f"{user_input} {user_level}"
        else:
            final_input = user_input
        
        user_input_clean = clean_text(final_input)
        user_embedding = self.model.encode([user_input_clean])
        
        # Cosine similarity - PERSIS NOTEBOOK
        if SKLEARN_AVAILABLE:
            sim_scores = cosine_similarity(user_embedding, self.embeddings).flatten()
        else:
            # Fallback: dot product
            sim_scores = np.dot(user_embedding, self.embeddings.T).flatten()
        
        # Ambil top-K - PERSIS NOTEBOOK
        top_indices = sim_scores.argsort()[::-1][:top_k]
        
        return self.lp_combined.iloc[top_indices][[
            'course_name',
            'learning_path',
            'course_level',
            'summary',
            'course_price'
        ]]