use std::{collections::HashMap, vec};
const DEFAULT_VECTOR_SIZE: usize = 100;

/// A simple vectorizer that maps words to integer values and creates fixed-size vectors.
/// It supports adding words, transforming documents into vectors, and padding vectors to a fixed size.
/// # Functionality
/// - `new(size: usize)`: Creates a new Vectorizer with a specified vector size.
/// - `add_word(word: &str, vector_value: isize)`: Adds a word and its corresponding integer value to the mappings.
/// - `word_to_int_transform(word: &str)`: Transforms a word into its corresponding integer value using a simple hash function.
/// - `vectorize(document: &str)`: Vectorizes a document (string) into a vector of integers based on word mappings.
/// - `devectorize(vector: &Vec<f32>)`: Converts a vector of integers back into a document (string) using the integer-to-word mapping.
/// - `transform(document: &str)`: Transforms a document (string) into a vector of integers.
pub struct Vectorizer {
    str_to_int: HashMap<String, isize>,
    int_to_str: HashMap<isize, String>,
    vector_size: usize,
}

impl Vectorizer {
    pub fn new(size: usize) -> Self {
        Vectorizer {
            str_to_int: HashMap::new(),
            int_to_str: HashMap::new(),
            vector_size: if size > 0 { size } else { DEFAULT_VECTOR_SIZE },
        }
    }

    /// Adds a word and its corresponding integer vector value to the mappings.
    /// # Arguments
    /// * `word` - The word to be added.
    /// * `vector_value` - The integer value representing the word in the vector space.
    /// # Example
    ///   let mut vectorizer = Vectorizer::new(50);
    ///  vectorizer.add_word("example", 12345);
    fn add_word(&mut self, word: &str, vector_value: isize) {
        if !self.str_to_int.contains_key(word) {
            self.str_to_int.insert(word.to_string(), vector_value);
            self.int_to_str.insert(vector_value, word.to_string());
        }
    }

    /// Transforms a word into its corresponding integer vector value using a simple hash function.
    /// # Arguments
    /// * `word` - The word to be transformed.
    /// # Returns
    /// * `isize` - The integer value representing the word in the vector space.
    /// # Example
    ///  let mut vectorizer = Vectorizer::new(50);
    /// let hash_value = vectorizer.word_to_int_transform("example");
    fn word_to_int_transform(&mut self, word: &str) -> isize {
        let mut hash = 0isize;
       
      let mut  BASE: isize  = 17 ;

        let modulo = if cfg!(target_pointer_width = "64") {
        9_223_372_036_854_775_783isize
    } else if cfg!(target_pointer_width = "32") {
        2_147_483_647isize 
    } else {
        panic!("Unsupported pointer width") ;
    };
        for ch in word.chars() {
            hash = (hash * BASE + (ch as u8 as isize)) % modulo;
            BASE = (BASE * 31) % modulo; // Update BASE for next character
        }

        self.add_word(word, hash);
        return hash;
    }

    /// Vectorizes a document (string) into a vector of integers based on word mappings.
    /// # Arguments
    /// * `document` - The input document as a string.
    /// # Returns
    /// * `Vec<isize>` - The resulting vector representation of the document.
    /// ```
    /// use custom_vector_db::vectorizer::Vectorizer;
    /// let mut vectorizer = Vectorizer::new(50);
    /// let document = "hello world this is a test document";
    /// let vector = vectorizer.vectorize(document);
    /// println!("{:?}", vector);
    /// // Output might look like: [123456, 789012, 345678, ...]
    /// ```
    /// If the document length exceeds the vector size, circular mapping is applied.
    pub fn vectorize(&mut self, document: &str) -> Vec<f32> {
        let mut vector = Vec::new();
        let mut index = 0;

        document.split_whitespace().for_each(|word| {
            let vector_value = if self.str_to_int.contains_key(word) {
                *self.str_to_int.get(word).unwrap()
            } else {
                self.word_to_int_transform(word)
            } as f32;

            if vector.len() >= self.vector_size {
                vector[index % self.vector_size] = vector_value;
                index += 1;
            } else {
                vector.push(vector_value);
            }
        });

        return vector;
    }
    /// Converts a vector of integers back into a document (string) using the integer-to-word mapping.
    /// #Note 
    /// nan or 0 values will be mapped to whitespace
    /// # Arguments
    /// * `vector` - The input vector of integers.
    /// # Returns
    /// * `String` - The resulting document as a string.
    /// # Example
    /// ```
    /// use custom_vector_db::vectorizer::Vectorizer;
    /// let mut vectorizer = Vectorizer::new(50);
    /// let document = "hello world this is a test document";
    /// let vector = vectorizer.vectorize(document);
    /// let reconstructed = vectorizer.devectorize(&vector);
    /// println!("{}", reconstructed);
    /// ```
    // TODO: add choice of handling unknown integers (e.g., skip, placeholder, etc.)
    pub fn devectorize (&self, vector: &Vec<f32>) -> String {
        let mut words = Vec::new();
        for &val in vector.iter() {
            let int_val = val as isize;
            if let Some(word) = self.int_to_str.get(&int_val) {
                words.push(word.clone());
            } else {
                words.push("_".to_string());
            }
        }
        let sentence = words.join(" ");
        return sentence;
    }

    /// Transforms a document into multiple vectors by splitting it into chunks based on the vector size.
    /// # Arguments
    /// * `document` - The input document as a string.
    /// # Returns
    /// * `Vec<Vec<isize>>` - A vector of vectors, each representing a chunk of the document.
    /// # Example
    /// ```
    /// use custom_vector_db::vectorizer::Vectorizer;
    /// let mut vectorizer = Vectorizer::new(3);
    /// let document = "hello world this is a test document";
    /// let transformed = vectorizer.transform(document);
    /// ```
    ///
    pub fn transform(&mut self, document: &str) -> Vec<Vec<f32>> {
        let words: Vec<&str> = document.split_whitespace().collect();
        let mut vectors = Vec::new();

        for chunk in words.chunks(self.vector_size) {
            let doc_chunk = chunk.join(" ");
            vectors.push(self.vectorize(&doc_chunk));
        }

        return vectors;
    }

    /// Pads a vector with zeros to center it within the specified vector size.
    /// to make the vectors of fixed size , with the content in the center
    /// # Arguments
    /// * `vector` - The input vector to be padded.
    /// # Example
    ///  let mut vectorizer = Vectorizer::new(5);
    /// let mut vector = vec![1.0, 2.0, 3.0];
    /// vectorizer.pad_center(&mut vector);
    /// println!("{:?}", vector); // Output: [0.0, 1.0, 2.0, 3.0, 0.0]
    /// # Note
    fn pad_center(&self, vectors: &mut Vec<Vec<f32>>) {
        for vector in vectors.iter_mut() {
            let current_len = vector.len();
            if current_len >= self.vector_size {
                continue; // No padding needed
            }

            let total_padding = self.vector_size - current_len;
            let padding_start = total_padding / 2;
            let padding_end = total_padding - padding_start;

            let mut padded_vector = Vec::with_capacity(self.vector_size);
            padded_vector.extend(vec![0.0; padding_start]);
            padded_vector.extend(vector.iter());
            padded_vector.extend(vec![0.0; padding_end]);

            *vector = padded_vector;
        }
    }

    fn pad_left(&self, vectors: &mut Vec<Vec<f32>>) {
        for vector in vectors.iter_mut() {
            let current_len = vector.len();
            if current_len >= self.vector_size {
                continue; // No padding needed
            }

            let total_padding = self.vector_size - current_len;

            let mut padded_vector = Vec::with_capacity(self.vector_size);
            padded_vector.extend(vec![0.0; total_padding]);
            padded_vector.extend(vector.iter());

            *vector = padded_vector;
        }
    }

    fn pad_right(&self, vectors: &mut Vec<Vec<f32>>) {
        for vector in vectors.iter_mut() {
            while vector.len() < self.vector_size {
                vector.push(0.0);
            }
        }
    }

    pub fn padding(&self, vectors: &mut Vec<Vec<f32>>, mode: &str) {
        match mode {
            "center" => self.pad_center(vectors),
            "left" => self.pad_left(vectors),
            "right" | _ => self.pad_right(vectors), // default is right padding
        }
    }

}

//** TESTS */
#[cfg(test)]
mod tests {
    use super::Vectorizer;

    #[test]
    fn test_vectorize() {
        let mut vectorizer = Vectorizer::new(10);
        let document = "hello world this is a test document";
        let v1 = vectorizer.vectorize(document);
        let v2 = vectorizer.vectorize(document);
        // Length equals number of words (<= vector_size)
        assert_eq!(v1.len(), 7);
        // Deterministic mapping for the same input
        assert_eq!(v1, v2);
        // Values are finite
        assert!(v1.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_transform() {
        let mut vectorizer = Vectorizer::new(3);
        let document = "hello world this is a test document";
        let transformed = vectorizer.transform(document);
        // With vector_size=3 and 7 words, we expect 3 chunks: [3,3,1]
        assert_eq!(transformed.len(), 3);
        assert_eq!(transformed[0].len(), 3);
        assert_eq!(transformed[1].len(), 3);
        assert_eq!(transformed[2].len(), 1);
        // Deterministic per chunk
        let again = vectorizer.transform(document);
        assert_eq!(transformed, again);
    }

    #[test]
    fn test_padding() {
        let mut vectorizer = Vectorizer::new(5);
        let mut vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0]];
        vectorizer.padding(&mut vectors, "center");
        // pad_center places roughly half zeros before and after
        assert_eq!(vectors[0], vec![0.0, 1.0, 2.0, 3.0, 0.0]);
        assert_eq!(vectors[1], vec![0.0, 4.0, 5.0, 0.0, 0.0]);
        assert_eq!(vectors[2], vec![0.0, 0.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_padding_left() {
        let mut vectorizer = Vectorizer::new(5);
        let mut vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0]];
        vectorizer.padding(&mut vectors, "left");
        assert_eq!(vectors[0], vec![0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(vectors[1], vec![0.0, 0.0, 0.0, 4.0, 5.0]);
        assert_eq!(vectors[2], vec![0.0, 0.0, 0.0, 0.0, 6.0]);
    }

    #[test]
    fn test_devectorize() {
        let mut vectorizer = Vectorizer::new(50);
        let document = "hello world this is a test document";
        let vector = vectorizer.vectorize(document) ;
        let reconstructed = vectorizer.devectorize(&vector);
        assert_eq!(vectorizer.vector_size, 50);
        println!("Original: {}", document);
        println!("Reconstructed: {}", reconstructed);
        
        // The reconstructed string should contain the original words (order may vary due to padding)
        for word in document.split_whitespace(){
            println!("Checking for word: {}", word);
            assert!(reconstructed.contains(word));
        }
    }
}
