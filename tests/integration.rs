use custom_vector_db::{cluster, vectorizer};

#[test]
fn vectorizer_end_to_end_padding_and_transform() {
    let mut v = vectorizer::Vectorizer::new(4);
    let doc = "one two three four five six"; // 6 words
    let chunks = v.transform(doc);
    assert_eq!(chunks.len(), 2); // [4,2]
    assert_eq!(chunks[0].len(), 4);
    assert_eq!(chunks[1].len(), 2);

    let mut padded = chunks.clone();
    v.padding(&mut padded, "right");
    assert!(padded.iter().all(|c| c.len() == 4));
}

#[test]
fn similarity_and_cluster_topk() {
    // Build a simple cluster and check top-k ordering
    let mut c = cluster::Cluster::new(1, "c".to_string(), 10, 3);
    c.add_vector(vec![1.0, 0.0, 0.0]);
    c.add_vector(vec![0.0, 1.0, 0.0]);
    c.add_vector(vec![0.0, 0.0, 1.0]);

    let results = c.search_topK(&vec![1.0, 0.0, 0.0], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], vec![1.0, 0.0, 0.0]);
}
