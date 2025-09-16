type T = f32; // for simplicity, using f32. Can be changed to f64 if needed or anyother numeric type
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Manhattan,
    DotProduct,
}

pub enum MetricRangeComparator {
    GreaterThan,
    LessThan,
    EqualTo,
}

pub enum similarityDirection {
    Ascending,  // for metrics where lower value means more similar (e.g., Euclidean, Manhattan)
    Descending, // for metrics where higher value means more similar (e.g., Cosine)
}

pub enum Normalization {
    VecNormalization,
    None,
}

// to be used by the others module so that we can use them in the arguments for decoupling
pub trait Similarity {
    fn compute(&self, base_vec: &Vec<T>, query_vec: &Vec<T>) -> T;
}

pub struct SimilarityFunction {
    metric: SimilarityMetric,
    direction: similarityDirection, // will be used when comparing two similarity scores of different vectors
    normalization: Normalization,
}

impl SimilarityFunction {
    pub fn new(metric: SimilarityMetric) -> Self {
        let direction = match metric {
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct => {
                similarityDirection::Descending
            }
            SimilarityMetric::Euclidean | SimilarityMetric::Manhattan => {
                similarityDirection::Ascending
            }
        };
        let normalization = match metric {
            SimilarityMetric::Cosine => Normalization::VecNormalization,
            _ => Normalization::None,
        };
        SimilarityFunction {
            metric,
            direction,
            normalization,
        }
    }

    pub fn compute(&self, base_vec: &Vec<T>, query_vec: &Vec<T>) -> T {
        match self.metric {
            SimilarityMetric::Cosine => function::cosine(base_vec, query_vec),
            SimilarityMetric::Euclidean => function::euclidean(base_vec, query_vec),
            SimilarityMetric::Manhattan => function::manhattan_distance(base_vec, query_vec),
            SimilarityMetric::DotProduct => function::dot_product(base_vec, query_vec),
        }
    }
}

///*
/// functions for similarity calculations
/// # Arguments
/// * `comperator` - First vector
/// * `compared` - Second vector    
/// # Returns
/// * `f32` - Similarity value
/// # Range of value depends on the metric used
/// */
pub mod function {

    type T = f32;

    /// Cosine Similarity between two vectors
    /// # Arguments
    /// * `comperator` - First vector
    /// * `compared` - Second vector
    /// # Returns
    /// * `f32` - Cosine similarity value between -1 and 1
    /// # Range of value [-1, 1]
    pub fn cosine(comperator: &Vec<T>, compared: &Vec<T>) -> T {
        // will only run upto the length of the smaller vector

        let dot_product: T = comperator
            .iter()
            .zip(compared.iter())
            .map(|(x, y)| x * y)
            .sum();
        let magnitude_a: T = comperator.iter().map(|x| x.powi(2)).sum::<T>().sqrt();
        let magnitude_b: T = compared.iter().map(|x| x.powi(2)).sum::<T>().sqrt();

        // to avoid division by zero
        if magnitude_a == 0.0 as T || magnitude_b == 0.0 as T {
            return 0.0 as T;
        }

        // value will be between -1 and 1
        // 1 means exactly same direction
        // -1 means exactly opposite direction
        // 0 means orthogonal (no similarity)
        return (dot_product / (magnitude_a * magnitude_b)) as T;
    }

    ///*
    /// Euclidean Distance between two vectors
    /// # Arguments
    /// * `comperator` - First vector
    /// * `compared` - Second vector
    /// # Returns
    /// * `f32` - Euclidean distance value (non-negative)
    /// # output Range of value [0, ∞)
    pub fn euclidean(comperator: &Vec<T>, compared: &Vec<T>) -> T {
        // work as : sqrt( comparatoer[i] - compared[i])^2 )
        let sum_squared_diff: T = comperator
            .iter()
            .zip(compared.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        return sum_squared_diff.sqrt();
    }

    /// Manhattan Distance between two vectors
    /// # Arguments
    /// * `comperator` - First vector
    /// * `compared` - Second vector
    /// # Returns
    /// * `f32` - Manhattan distance value (non-negative)
    /// # output Range of value [0, ∞)
    ///
    pub fn manhattan_distance(comperator: &Vec<T>, compared: &Vec<T>) -> T {
        // work as : |comparatoer[i] - compared[i]|
        let sum_abs_diff: T = comperator
            .iter()
            .zip(compared.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        return sum_abs_diff;
    }

    pub fn dot_product(comperator: &Vec<T>, compared: &Vec<T>) -> T {
        return comperator
            .iter()
            .zip(compared.iter())
            .map(|(x, y)| x * y)
            .sum();
    }

    pub fn normalize_vec(vector: &mut Vec<T>) {
        let magnitude: T = vector.iter().map(|x| x.powi(2)).sum::<T>().sqrt();
        if magnitude == 0.0 as T {
            return;
        }
        for i in 0..vector.len() {
            vector[i] = vector[i] / magnitude;
        }
    }
}

//** TESTS */
#[cfg(test)]
mod tests {
    use super::function::*;
    use super::*;
    #[test]
    fn test_cosine() {
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0];
        let vec_c = vec![1.0, 1.0, 0.0];
        let vec_d = vec![2.0, 0.0, 0.0];

        assert!((cosine(&vec_a, &vec_a) - 1.0).abs() < 1e-6); // Same vector
        assert!((cosine(&vec_a, &vec_b) - 0.0).abs() < 1e-6); // Orthogonal vectors
        assert!((cosine(&vec_a, &vec_c) - 0.70710678).abs() < 1e-6); // 45 degrees
        assert!((cosine(&vec_a, &vec_d) - 1.0).abs() < 1e-6); // Same direction, different magnitude
    }
    #[test]
    fn test_euclidean() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![4.0, 5.0, 6.0];
        let vec_c = vec![1.0, 2.0, 3.0];

        assert!((euclidean(&vec_a, &vec_b) - 5.19615242).abs() < 1e-6); // Distance between a and b
        assert!((euclidean(&vec_a, &vec_c) - 0.0).abs() < 1e-6); // Same vector
    }
    #[test]
    fn test_manhattan() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![4.0, 5.0, 6.0];
        let vec_c = vec![1.0, 2.0, 3.0];

        assert!((manhattan_distance(&vec_a, &vec_b) - 9.0).abs() < 1e-6); // Distance between a and b
        assert!((manhattan_distance(&vec_a, &vec_c) - 0.0).abs() < 1e-6); // Same vector
    }
    #[test]
    fn test_dot_product() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![4.0, 5.0, 6.0];
        let vec_c = vec![1.0, 2.0, 3.0];

        assert!((dot_product(&vec_a, &vec_b) - 32.0).abs() < 1e-6); // Dot product of a and b
        assert!((dot_product(&vec_a, &vec_c) - 14.0).abs() < 1e-6); // Dot product of a and c
    }
    #[test]
    fn test_normalize_vec() {
        let mut vec_a = vec![3.0, 4.0];
        normalize_vec(&mut vec_a);
        assert!((vec_a[0] - 0.6f32).abs() < 1e-6);
        assert!((vec_a[1] - 0.8f32).abs() < 1e-6);

        let mut vec_b = vec![0.0, 0.0, 0.0];
        normalize_vec(&mut vec_b);
        assert_eq!(vec_b, vec![0.0, 0.0, 0.0]); // Zero vector remains unchanged
    }

    #[test]
    fn test_similarity_function() {
        let sim_func_cosine = SimilarityFunction::new(SimilarityMetric::Cosine);
        let sim_func_euclidean = SimilarityFunction::new(SimilarityMetric::Euclidean);
        let sim_func_manhattan = SimilarityFunction::new(SimilarityMetric::Manhattan);
        let sim_func_dot = SimilarityFunction::new(SimilarityMetric::DotProduct);

        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![0.0, 1.0, 0.0];

        assert!((sim_func_cosine.compute(&vec_a, &vec_a) - 1.0).abs() < 1e-6);
        assert!((sim_func_cosine.compute(&vec_a, &vec_b) - 0.0).abs() < 1e-6);

        assert!((sim_func_euclidean.compute(&vec_a, &vec_b) - 1.41421356).abs() < 1e-6);
        assert!((sim_func_euclidean.compute(&vec_a, &vec_a) - 0.0).abs() < 1e-6);

        assert!((sim_func_manhattan.compute(&vec_a, &vec_b) - 2.0).abs() < 1e-6);
        assert!((sim_func_manhattan.compute(&vec_a, &vec_a) - 0.0).abs() < 1e-6);

        assert!((sim_func_dot.compute(&vec_a, &vec_b) - 0.0).abs() < 1e-6);
        assert!((sim_func_dot.compute(&vec_a, &vec_a) - 1.0).abs() < 1e-6);
    }
}
