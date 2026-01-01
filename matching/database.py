"""
Embedding Database Module
=========================
Efficient storage and retrieval of palm vein embeddings using FAISS.
Supports enrollment, search, and verification operations.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class EmbeddingDatabase:
    """
    Database for storing and searching palm vein embeddings.
    Uses FAISS for efficient similarity search with fallback to numpy.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "Flat",
        distance_metric: str = "cosine",
        db_path: Optional[Union[str, Path]] = None,
        use_faiss: bool = True
    ):
        """
        Initialize the embedding database.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type (Flat, IVFFlat, IVFPQ)
            distance_metric: Distance metric (cosine, euclidean)
            db_path: Path to save/load database
            use_faiss: Whether to use FAISS (falls back to numpy if unavailable)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.db_path = Path(db_path) if db_path else None
        
        # Try to import FAISS
        self.faiss = None
        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
                logger.info("FAISS loaded successfully")
            except ImportError:
                logger.warning("FAISS not available, using numpy fallback")
        
        # Initialize storage
        self.index = None
        self.embeddings = []  # numpy fallback
        self.metadata = {}  # id -> {name, enrollment_date, etc.}
        self.name_to_ids = defaultdict(list)  # name -> [ids]
        self.id_counter = 0
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
        
        # Initialize index
        self._init_index()
        
        # Load existing database if path provided
        if self.db_path and self.db_path.exists():
            self.load()
    
    def _init_index(self):
        """Initialize the FAISS index or numpy storage."""
        if self.faiss is not None:
            if self.distance_metric == "cosine":
                # Use inner product with normalized vectors for cosine similarity
                if self.index_type == "Flat":
                    self.index = self.faiss.IndexFlatIP(self.embedding_dim)
                elif self.index_type == "IVFFlat":
                    quantizer = self.faiss.IndexFlatIP(self.embedding_dim)
                    self.index = self.faiss.IndexIVFFlat(
                        quantizer, self.embedding_dim, 100, self.faiss.METRIC_INNER_PRODUCT
                    )
                elif self.index_type == "IVFPQ":
                    quantizer = self.faiss.IndexFlatIP(self.embedding_dim)
                    self.index = self.faiss.IndexIVFPQ(
                        quantizer, self.embedding_dim, 100, 16, 8, self.faiss.METRIC_INNER_PRODUCT
                    )
            else:  # euclidean
                if self.index_type == "Flat":
                    self.index = self.faiss.IndexFlatL2(self.embedding_dim)
                elif self.index_type == "IVFFlat":
                    quantizer = self.faiss.IndexFlatL2(self.embedding_dim)
                    self.index = self.faiss.IndexIVFFlat(
                        quantizer, self.embedding_dim, 100
                    )
        else:
            self.embeddings = []
    
    def enroll(
        self,
        embedding: np.ndarray,
        name: str,
        additional_info: Optional[Dict] = None
    ) -> int:
        """
        Enroll a new identity with their embedding.
        
        Args:
            embedding: Normalized embedding vector
            name: Person's name/ID
            additional_info: Additional metadata
            
        Returns:
            Assigned ID
        """
        with self.lock:
            # Ensure embedding is normalized
            embedding = self._normalize(embedding)
            
            # Assign ID
            person_id = self.id_counter
            self.id_counter += 1
            
            # Add to index
            if self.faiss is not None and self.index is not None:
                self.index.add(embedding.reshape(1, -1).astype(np.float32))
            else:
                self.embeddings.append(embedding)
            
            # Store metadata
            self.metadata[person_id] = {
                'name': name,
                'enrollment_date': datetime.now().isoformat(),
                'additional_info': additional_info or {}
            }
            
            # Update name mapping
            self.name_to_ids[name].append(person_id)
            
            logger.info(f"Enrolled {name} with ID {person_id}")
            
            return person_id
    
    def enroll_batch(
        self,
        embeddings: np.ndarray,
        names: List[str],
        additional_infos: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Enroll multiple identities at once.
        
        Args:
            embeddings: Array of embeddings (N, D)
            names: List of names
            additional_infos: List of additional metadata dicts
            
        Returns:
            List of assigned IDs
        """
        if additional_infos is None:
            additional_infos = [None] * len(names)
        
        ids = []
        for emb, name, info in zip(embeddings, names, additional_infos):
            person_id = self.enroll(emb, name, info)
            ids.append(person_id)
        
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            threshold: Similarity threshold (optional)
            
        Returns:
            List of results with id, name, similarity, and metadata
        """
        with self.lock:
            query_embedding = self._normalize(query_embedding)
            
            if self.faiss is not None and self.index is not None:
                if self.index.ntotal == 0:
                    return []
                
                # FAISS search
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), k
                )
                distances = distances[0]
                indices = indices[0]
                
                # Convert distances to similarities
                if self.distance_metric == "cosine":
                    similarities = distances  # Already similarity for IP
                else:
                    # Convert L2 distance to similarity
                    similarities = 1 / (1 + distances)
            else:
                if not self.embeddings:
                    return []
                
                # Numpy fallback
                embeddings_array = np.array(self.embeddings)
                
                if self.distance_metric == "cosine":
                    similarities = np.dot(embeddings_array, query_embedding)
                else:
                    distances = np.linalg.norm(embeddings_array - query_embedding, axis=1)
                    similarities = 1 / (1 + distances)
                
                # Get top k
                top_k_idx = np.argsort(similarities)[::-1][:k]
                indices = top_k_idx
                similarities = similarities[top_k_idx]
            
            # Build results
            results = []
            for idx, sim in zip(indices, similarities):
                if idx < 0:  # FAISS returns -1 for empty slots
                    continue
                
                idx = int(idx)
                if threshold is not None and sim < threshold:
                    continue
                
                if idx in self.metadata:
                    results.append({
                        'id': idx,
                        'name': self.metadata[idx]['name'],
                        'similarity': float(sim),
                        'metadata': self.metadata[idx]
                    })
            
            return results
    
    def verify(
        self,
        query_embedding: np.ndarray,
        claimed_name: str,
        threshold: float = 0.75
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Verify if query embedding matches claimed identity.
        
        Args:
            query_embedding: Query embedding
            claimed_name: Claimed identity name
            threshold: Verification threshold
            
        Returns:
            (is_verified, best_similarity, best_match_info)
        """
        with self.lock:
            # Get all embeddings for claimed identity
            claimed_ids = self.name_to_ids.get(claimed_name, [])
            
            if not claimed_ids:
                return False, 0.0, None
            
            query_embedding = self._normalize(query_embedding)
            
            best_similarity = -1
            best_match = None
            
            for person_id in claimed_ids:
                # Get embedding for this ID
                if self.faiss is not None and self.index is not None:
                    # Reconstruct from FAISS
                    stored_embedding = self.index.reconstruct(person_id)
                else:
                    stored_embedding = self.embeddings[person_id]
                
                # Compute similarity
                if self.distance_metric == "cosine":
                    similarity = float(np.dot(query_embedding, stored_embedding))
                else:
                    distance = np.linalg.norm(query_embedding - stored_embedding)
                    similarity = 1 / (1 + distance)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = self.metadata[person_id]
            
            is_verified = best_similarity >= threshold
            
            return is_verified, best_similarity, best_match
    
    def identify(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.75
    ) -> Tuple[Optional[str], float, Optional[Dict]]:
        """
        Identify the person from query embedding.
        
        Args:
            query_embedding: Query embedding
            threshold: Identification threshold
            
        Returns:
            (identified_name, similarity, metadata) or (None, 0, None) if not identified
        """
        results = self.search(query_embedding, k=1, threshold=threshold)
        
        if results:
            top_result = results[0]
            return top_result['name'], top_result['similarity'], top_result['metadata']
        
        return None, 0.0, None
    
    def delete(self, person_id: int) -> bool:
        """
        Delete an enrollment by ID.
        Note: FAISS doesn't support deletion, so we mark as deleted in metadata.
        """
        with self.lock:
            if person_id not in self.metadata:
                return False
            
            name = self.metadata[person_id]['name']
            
            # Mark as deleted
            self.metadata[person_id]['deleted'] = True
            self.metadata[person_id]['deletion_date'] = datetime.now().isoformat()
            
            # Remove from name mapping
            if name in self.name_to_ids:
                self.name_to_ids[name] = [
                    pid for pid in self.name_to_ids[name] if pid != person_id
                ]
            
            logger.info(f"Deleted enrollment {person_id} ({name})")
            return True
    
    def delete_by_name(self, name: str) -> int:
        """Delete all enrollments for a name. Returns count deleted."""
        ids = self.name_to_ids.get(name, []).copy()
        count = 0
        for person_id in ids:
            if self.delete(person_id):
                count += 1
        return count
    
    def get_all_names(self) -> List[str]:
        """Get list of all enrolled names."""
        return list(self.name_to_ids.keys())
    
    def get_enrollment_count(self) -> int:
        """Get total number of active enrollments."""
        count = 0
        for meta in self.metadata.values():
            if not meta.get('deleted', False):
                count += 1
        return count
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save database to disk."""
        path = Path(path) if path else self.db_path
        if path is None:
            raise ValueError("No save path specified")
        
        path.mkdir(parents=True, exist_ok=True)
        
        with self.lock:
            # Save FAISS index
            if self.faiss is not None and self.index is not None:
                self.faiss.write_index(self.index, str(path / "index.faiss"))
            else:
                # Save numpy embeddings
                np.save(path / "embeddings.npy", np.array(self.embeddings))
            
            # Save metadata
            db_data = {
                'metadata': self.metadata,
                'name_to_ids': dict(self.name_to_ids),
                'id_counter': self.id_counter,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'distance_metric': self.distance_metric
            }
            
            with open(path / "metadata.json", 'w') as f:
                json.dump(db_data, f, indent=2, default=str)
            
            logger.info(f"Database saved to {path}")
    
    def load(self, path: Optional[Union[str, Path]] = None):
        """Load database from disk."""
        path = Path(path) if path else self.db_path
        if path is None:
            raise ValueError("No load path specified")
        
        if not path.exists():
            raise FileNotFoundError(f"Database not found at {path}")
        
        with self.lock:
            # Load metadata
            with open(path / "metadata.json", 'r') as f:
                db_data = json.load(f)
            
            self.metadata = {int(k): v for k, v in db_data['metadata'].items()}
            self.name_to_ids = defaultdict(list, db_data['name_to_ids'])
            self.id_counter = db_data['id_counter']
            self.embedding_dim = db_data['embedding_dim']
            self.index_type = db_data['index_type']
            self.distance_metric = db_data['distance_metric']
            
            # Load FAISS index or numpy embeddings
            faiss_path = path / "index.faiss"
            numpy_path = path / "embeddings.npy"
            
            if self.faiss is not None and faiss_path.exists():
                self.index = self.faiss.read_index(str(faiss_path))
            elif numpy_path.exists():
                self.embeddings = np.load(numpy_path).tolist()
            
            logger.info(f"Database loaded from {path} ({self.get_enrollment_count()} enrollments)")
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def rebuild_index(self):
        """Rebuild the FAISS index (useful after many deletions)."""
        if self.faiss is None:
            return
        
        with self.lock:
            # Collect all active embeddings
            active_embeddings = []
            active_ids = []
            
            for person_id, meta in self.metadata.items():
                if not meta.get('deleted', False):
                    if self.index is not None:
                        emb = self.index.reconstruct(person_id)
                    else:
                        emb = self.embeddings[person_id]
                    active_embeddings.append(emb)
                    active_ids.append(person_id)
            
            if not active_embeddings:
                self._init_index()
                return
            
            # Create new index
            self._init_index()
            
            # Add embeddings back
            embeddings_array = np.array(active_embeddings).astype(np.float32)
            
            if self.index_type in ["IVFFlat", "IVFPQ"]:
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # Update ID mapping
            # Note: This changes IDs, so metadata needs updating
            new_metadata = {}
            new_name_to_ids = defaultdict(list)
            
            for new_id, old_id in enumerate(active_ids):
                new_metadata[new_id] = self.metadata[old_id]
                name = self.metadata[old_id]['name']
                new_name_to_ids[name].append(new_id)
            
            self.metadata = new_metadata
            self.name_to_ids = new_name_to_ids
            self.id_counter = len(active_ids)
            
            logger.info(f"Index rebuilt with {len(active_ids)} embeddings")


class MultiTemplateDatabase(EmbeddingDatabase):
    """
    Extended database that stores multiple templates per person
    for more robust matching.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        max_templates_per_person: int = 5,
        **kwargs
    ):
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.max_templates_per_person = max_templates_per_person
    
    def enroll_with_augmentation(
        self,
        embeddings: List[np.ndarray],
        name: str,
        additional_info: Optional[Dict] = None
    ) -> List[int]:
        """
        Enroll multiple templates for the same person.
        
        Args:
            embeddings: List of embeddings (different augmentations)
            name: Person's name
            additional_info: Additional metadata
            
        Returns:
            List of assigned IDs
        """
        # Limit number of templates
        embeddings = embeddings[:self.max_templates_per_person]
        
        ids = []
        for i, emb in enumerate(embeddings):
            info = (additional_info or {}).copy()
            info['template_index'] = i
            person_id = self.enroll(emb, name, info)
            ids.append(person_id)
        
        return ids
    
    def verify_multi_template(
        self,
        query_embedding: np.ndarray,
        claimed_name: str,
        threshold: float = 0.75,
        aggregation: str = "max"
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Verify using multiple templates with aggregation.
        
        Args:
            query_embedding: Query embedding
            claimed_name: Claimed identity
            threshold: Verification threshold
            aggregation: How to aggregate template scores (max, mean, voting)
        """
        claimed_ids = self.name_to_ids.get(claimed_name, [])
        
        if not claimed_ids:
            return False, 0.0, None
        
        query_embedding = self._normalize(query_embedding)
        
        similarities = []
        for person_id in claimed_ids:
            meta = self.metadata.get(person_id, {})
            if meta.get('deleted', False):
                continue
            
            if self.faiss is not None and self.index is not None:
                stored_embedding = self.index.reconstruct(person_id)
            else:
                stored_embedding = self.embeddings[person_id]
            
            if self.distance_metric == "cosine":
                sim = float(np.dot(query_embedding, stored_embedding))
            else:
                dist = np.linalg.norm(query_embedding - stored_embedding)
                sim = 1 / (1 + dist)
            
            similarities.append(sim)
        
        if not similarities:
            return False, 0.0, None
        
        # Aggregate
        if aggregation == "max":
            final_similarity = max(similarities)
        elif aggregation == "mean":
            final_similarity = np.mean(similarities)
        elif aggregation == "voting":
            votes = sum(1 for s in similarities if s >= threshold)
            final_similarity = votes / len(similarities)
            is_verified = votes >= len(similarities) / 2
            return is_verified, final_similarity, self.metadata.get(claimed_ids[0])
        else:
            final_similarity = max(similarities)
        
        is_verified = final_similarity >= threshold
        return is_verified, final_similarity, self.metadata.get(claimed_ids[0])
