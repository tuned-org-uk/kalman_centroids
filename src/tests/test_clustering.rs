use crate::KalmanClusterer;
use smartcore::linalg::basic::arrays::Array;

fn make_two_blob_dataset(n_per: usize) -> Vec<Vec<f64>> {
    // Deterministic “two blobs” in 2D, no RNG required.
    // Blob A around (0,0), Blob B around (10,10).
    let mut rows = Vec::with_capacity(2 * n_per);
    for i in 0..n_per {
        let t = i as f64 / (n_per as f64);
        rows.push(vec![0.2 * t, -0.2 * t]); // A: small spread
        rows.push(vec![10.0 + 0.2 * t, 10.0 - 0.2 * t]); // B: small spread
    }
    rows
}

fn make_three_blob_dataset(n_per: usize) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(3 * n_per);
    for i in 0..n_per {
        let t = i as f64 / (n_per as f64);
        rows.push(vec![0.0 + 0.1 * t, 0.0 - 0.1 * t]);
        rows.push(vec![10.0 + 0.1 * t, 10.0 - 0.1 * t]);
        rows.push(vec![-10.0 + 0.1 * t, 10.0 + 0.1 * t]);
    }
    rows
}

#[test]
fn kalman_clusterer_two_blobs_creates_multiple_centroids_and_assigns_all() {
    let rows = make_two_blob_dataset(200);
    let n = rows.len();

    let mut kc = KalmanClusterer::new(64, n);

    // Make splitting easier than default (default may still work, this just hardens the test).
    kc.split_threshold_mahal = 1.0;
    kc.merge_threshold_bhatt = 0.05;

    kc.fit(&rows);

    assert!(
        kc.centroids.len() >= 2,
        "Expected at least 2 centroids for two blobs, got {}",
        kc.centroids.len()
    );

    assert_eq!(
        kc.assignments.len(),
        n,
        "Assignments length mismatch: {} vs {}",
        kc.assignments.len(),
        n
    );

    for (i, a) in kc.assignments.iter().enumerate() {
        let idx = a.expect("Every row should be assigned in this dataset");
        assert!(
            idx < kc.centroids.len(),
            "Assignment out of bounds at row {}: {} >= {}",
            i,
            idx,
            kc.centroids.len()
        );
    }

    for (cidx, c) in kc.centroids.iter().enumerate() {
        assert_eq!(c.mean.len(), 2, "Centroid {} mean dimension mismatch", cidx);
        assert_eq!(
            c.variance.len(),
            2,
            "Centroid {} variance dimension mismatch",
            cidx
        );
        assert!(
            c.mean.iter().all(|x| x.is_finite()),
            "Centroid {} mean has NaN/inf",
            cidx
        );
        assert!(
            c.variance.iter().all(|x| x.is_finite() && *x > 0.0),
            "Centroid {} variance has invalid values",
            cidx
        );
        assert!(c.count >= 1, "Centroid {} count must be >=1", cidx);
    }
}

#[test]
fn kalman_clusterer_export_centroids_shape_is_correct() {
    let rows = make_three_blob_dataset(120);
    let n = rows.len();

    let mut kc = KalmanClusterer::new(128, n);
    kc.split_threshold_mahal = 1.0;
    kc.merge_threshold_bhatt = 0.05;

    kc.fit(&rows);

    let dm = kc.export_centroids();
    let (k, f) = dm.shape();

    assert_eq!(f, 2, "Expected 2 features in exported centroids, got {}", f);
    assert_eq!(
        k,
        kc.centroids.len(),
        "DenseMatrix rows should match number of centroids"
    );
    assert!(
        k >= 3,
        "Expected at least 3 centroids for three blobs, got {}",
        k
    );

    // Basic finiteness check on exported matrix.
    for i in 0..k {
        for j in 0..f {
            let v = dm.get((i, j));
            assert!(
                v.is_finite(),
                "export_centroids contains NaN/inf at ({},{})",
                i,
                j
            );
        }
    }
}

#[test]
fn kalman_clusterer_respects_max_k_cap() {
    // Many far-apart points in 2D; with a low split threshold, we try to create lots of clusters.
    let mut rows = Vec::new();
    for i in 0..200usize {
        rows.push(vec![i as f64 * 10.0, 0.0]);
    }

    let n = rows.len();
    let max_k = 8;
    let mut kc = KalmanClusterer::new(max_k, n);
    kc.split_threshold_mahal = 1e-6; // force new cluster creation pressure

    kc.fit(&rows);

    assert!(
        kc.centroids.len() <= max_k,
        "Expected centroids <= max_k, got {} > {}",
        kc.centroids.len(),
        max_k
    );

    // Even when max_k hit, we still assign all points.
    assert!(kc.assignments.iter().all(|a| a.is_some()));
}
