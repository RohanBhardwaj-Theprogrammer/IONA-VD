use std::collections::BinaryHeap;

use crate::similarity::function::cosine;

type T = f32; // for simplicity, using f32. Can be changed to f64 if needed or anyother numeric type

const MAX_CLUSTER_SIZE: usize = 1000; // maximum number of vectors in a cluster

struct SimilarityScore(T, usize); // (similarity score, index in the cluster vectors)

impl std::cmp::PartialEq for SimilarityScore {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl std::cmp::Eq for SimilarityScore {}
impl std::cmp::PartialOrd for SimilarityScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl std::cmp::Ord for SimilarityScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub struct Cluster {
    id: u32,
    name: String,
    cluster_size: usize,
    vector_size: usize,
    // cluster_metric: SimilarityMetric, // to be used in future
    centroid: Vec<T>,
    vectors: Vec<Vec<T>>,
}

impl Cluster {
    /// Creates a new `Cluster` with the specified parameters.
    ///
    /// The cluster's size is capped at `MAX_CLUSTER_SIZE` to prevent excessive memory usage.
    /// The centroid is initialized to a zero vector of the given dimension.
    ///
    /// # Arguments
    /// * `id` - A unique identifier for the cluster (e.g., `1`).
    /// * `name` - A descriptive name for the cluster (e.g., `"MyCluster"`).
    /// * `size` - The maximum number of vectors the cluster can hold. Must be > 0; otherwise, defaults to `MAX_CLUSTER_SIZE`.
    /// * `vector_size` - The dimensionality of each vector (e.g., `3` for 3D vectors).
    /// # Returns
    /// A new `Cluster` instance with an empty vector list and zero-initialized centroid.
    pub fn new(id: u32, name: String, size: usize, vector_size: usize) -> Self {
        Cluster {
            id,
            name,
            cluster_size: if size > MAX_CLUSTER_SIZE || size <= 0 {
                MAX_CLUSTER_SIZE
            } else {
                size
            },
            vector_size,
            centroid: vec![0 as T; vector_size],
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: Vec<T>) -> bool {
        // normalize the vector if needed but not yet to be used

        if self.vectors.len() < self.cluster_size {
            self.update_centroid(&vector);
            self.vectors.push(vector);
            return true;
        } else {
            return false; // cluster is full
        }
    }

    fn update_centroid(&mut self, query: &Vec<T>) {
        let num_vectors = self.vectors.len() as T;

        if num_vectors == 0 as T {
            self.centroid = query.clone();
            return;
        }
        self.centroid
            .iter_mut()
            .zip(query.iter())
            .for_each(|(c, q)| {
                *c = ((*c * num_vectors) + *q) / (num_vectors + 1 as T);
            });
    }

    pub fn search_topK(&self, query: &Vec<T>, result_k: usize) -> Vec<Vec<T>> {
        let mut results: Vec<Vec<T>> = Vec::new();
        if result_k <= 0 {
            return results;
        }
        use std::collections::BinaryHeap;
        let mut top_k_res_heap: BinaryHeap<SimilarityScore> = BinaryHeap::with_capacity(result_k);

        for (idx, vector) in self.vectors.iter().enumerate() {
            let sim_score = cosine(&vector, &query);

            if top_k_res_heap.len() < result_k {
                top_k_res_heap.push(SimilarityScore(sim_score, idx));
            } else if let Some(mut lowest) = top_k_res_heap.peek_mut() {
                if sim_score > lowest.0 {
                    *lowest = SimilarityScore(sim_score, idx);
                }
            }
        }

        // convert the heap back to a sorted vector , in a descending order of similarity
        for SimilarityScore(_, idx) in top_k_res_heap.into_sorted_vec().into_iter().rev() {
            results.push(self.vectors[idx].clone());
        }

        return results;
    }
}

/** *************************************************************** */
//** TESTS */
#[cfg(test)]
mod cluster_tests {
    use std::vec;

    use super::*;
    #[test]
    fn test_add_vector() {
        let mut cluster = Cluster::new(1, "TestCluster".to_string(), 3, 3);
        assert!(cluster.add_vector(vec![1.0, 2.0, 3.0]));
        assert!(cluster.add_vector(vec![4.0, 5.0, 6.0]));
        assert!(cluster.add_vector(vec![7.0, 8.0, 9.0]));
        assert!(!cluster.add_vector(vec![10.0, 11.0, 12.0])); // Should fail as cluster is full
        assert_eq!(cluster.vectors.len(), 3);
    }

    #[test]
    fn test_search_topK() {
        let mut cluster = Cluster::new(1, "TestCluster".to_string(), 10, 3);
        cluster.add_vector(vec![1.0, 0.0, 0.0]);
        cluster.add_vector(vec![0.0, 1.0, 0.0]);
        cluster.add_vector(vec![0.0, 0.0, 1.0]);
        cluster.add_vector(vec![1.0, 1234561.0, 0.0]);
        cluster.add_vector(vec![1.0, 1.0, 1.0]);
        cluster.add_vector(vec![1.0, 0.5, 12222.75]);
        cluster.add_vector(vec![1.0, 0.5, 0.5]);

        println!("The Centroid is : {:?}", cluster.centroid);

        for vec in &cluster.vectors {
            println!("Cluster Vector: {:?}", vec);
        }

        for vec in &cluster.vectors {
            let sim = cosine(&vec, &vec![1.0, 0.5, 0.5]);
            println!("Similarity with [1.0, 0.5, 0.5]: {:.4}", sim);
        }

        let results = cluster.search_topK(&vec![1.0, 0.5, 0.5], 4);
        println!(
            "Top {} similar vectors: to  {}",
            results.len(),
            "[1.0, 0.5, 0.5]"
        );
        for res in &results {
            println!("{:?}", res);
        }

        assert_eq!(results.len(), 4);
        // The top results should be the vectors closest to [1.0, 0.5, 0.5]
        // Since [1.0, 0.5, 0.5] wasn't added (cluster full), the closest are others
    }
}
