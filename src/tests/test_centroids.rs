use crate::centroids::KalmanCentroid;

fn assert_all_finite(xs: &[f64]) {
    assert!(
        xs.iter().all(|v| v.is_finite()),
        "expected all finite, got {xs:?}"
    );
}

#[test]
fn new_initializes_expected_state() {
    let x0 = vec![1.0, -2.0, 3.5];
    let c = KalmanCentroid::new(&x0, 1e-4, 1e-2);

    assert_eq!(c.mean, x0);
    assert_eq!(c.variance.len(), x0.len());
    assert!(c.variance.iter().all(|&v| (v - 10.0).abs() < 1e-12));
    assert_eq!(c.count, 1);
    assert!((c.process_noise - 1e-4).abs() < 1e-18);
    assert!((c.measurement_noise - 1e-2).abs() < 1e-18);
    assert!((c.confidence - 0.1).abs() < 1e-12);

    assert_all_finite(&c.mean);
    assert_all_finite(&c.variance);
    assert!(c.confidence.is_finite());
}

#[test]
fn update_moves_mean_toward_measurement_and_shrinks_variance() {
    let x0 = vec![0.0, 0.0];
    let mut c = KalmanCentroid::new(&x0, 1e-6, 1e-2);

    let p_before = c.variance.clone();
    let mean_before = c.mean.clone();
    let count_before = c.count;

    let z = vec![1.0, -1.0];
    c.update(&z);

    // Mean should move toward measurement (not necessarily equal, but closer than before).
    for i in 0..2 {
        let dist_before = (z[i] - mean_before[i]).abs();
        let dist_after = (z[i] - c.mean[i]).abs();
        assert!(
            dist_after < dist_before,
            "mean did not move toward measurement on dim {i}: before={dist_before} after={dist_after}"
        );
    }

    // Variance should shrink (posterior variance < predicted variance).
    // Given p_pred = p + Q, and posterior = (1-K)*p_pred with 0<K<1 => posterior < p_pred.
    // With small Q, posterior should also be <= prior in this configuration.
    for i in 0..2 {
        assert!(
            c.variance[i] < (p_before[i] + c.process_noise),
            "variance did not shrink vs predicted on dim {i}"
        );
        assert!(
            c.variance[i] <= p_before[i],
            "variance unexpectedly increased vs prior on dim {i}"
        );
    }

    assert_eq!(c.count, count_before + 1);
    assert!(c.confidence > 0.0);
    assert!(c.confidence.is_finite());

    assert_all_finite(&c.mean);
    assert_all_finite(&c.variance);
}

#[test]
fn repeated_updates_converge_mean_and_increase_confidence() {
    let mut c = KalmanCentroid::new(&[0.0, 0.0, 0.0], 1e-8, 1e-3);
    let z = vec![2.0, -1.0, 0.5];

    let mut prev_err = f64::INFINITY;
    let mut prev_conf = c.confidence;

    for _ in 0..50 {
        c.update(&z);

        let err = c
            .mean
            .iter()
            .zip(&z)
            .map(|(m, zi)| (zi - m).abs())
            .sum::<f64>();

        assert!(err.is_finite());
        assert!(
            err <= prev_err + 1e-12,
            "error did not monotonically decrease"
        );
        prev_err = err;

        assert!(c.confidence.is_finite());
        assert!(
            c.confidence >= prev_conf - 1e-12,
            "confidence did not increase"
        );
        prev_conf = c.confidence;
    }

    // Should end close-ish to measurement (tolerance is loose to avoid test brittleness).
    for i in 0..3 {
        assert!((c.mean[i] - z[i]).abs() < 1e-2);
    }
}

#[test]
fn mahalanobis_distance_sq_is_zero_at_mean_and_non_negative() {
    let c = KalmanCentroid::new(&[1.0, 2.0], 1e-4, 1e-2);

    let d0 = c.mahalanobis_distance_sq(&[1.0, 2.0]);
    assert!((d0 - 0.0).abs() < 1e-12);

    let d1 = c.mahalanobis_distance_sq(&[2.0, 0.0]);
    assert!(d1 >= 0.0);
    assert!(d1.is_finite());
}

#[test]
fn mahalanobis_uses_variance_scaling() {
    let mut c = KalmanCentroid::new(&[0.0, 0.0], 0.0, 1.0);
    // Make dim0 very certain (small variance), dim1 uncertain (large variance)
    c.variance = vec![1e-3, 1000.0];

    // Same absolute deviation in both dims: x = [1, 1]
    let d = c.mahalanobis_distance_sq(&[1.0, 1.0]);
    // Contribution from dim0 should dominate due to tiny variance.
    let d0 = (1.0f64 * 1.0) / 1e-3;
    let d1 = (1.0f64 * 1.0) / 1000.0;
    let expected = d0 + d1;
    assert!((d - expected).abs() / expected < 1e-12);
}

#[test]
fn bhattacharyya_is_symmetric_and_zero_for_identical_centroids() {
    let mut a = KalmanCentroid::new(&[0.5, -0.5], 1e-4, 1e-2);
    let mut b = KalmanCentroid::new(&[0.5, -0.5], 1e-4, 1e-2);

    // Force identical variances as well
    a.variance = vec![0.2, 0.3];
    b.variance = vec![0.2, 0.3];

    let d_ab = a.bhattacharyya_distance(&b);
    let d_ba = b.bhattacharyya_distance(&a);

    assert!(d_ab.is_finite());
    assert!(d_ba.is_finite());
    assert!((d_ab - d_ba).abs() < 1e-12, "expected symmetry");
    assert!(
        d_ab.abs() < 1e-12,
        "expected ~0 for identical distributions"
    );
}

#[test]
fn bhattacharyya_increases_with_mean_separation() {
    let mut a = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    let mut b = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    a.variance = vec![1.0, 1.0];
    b.variance = vec![1.0, 1.0];

    let d0 = a.bhattacharyya_distance(&b);

    b.mean = vec![0.1, 0.1];
    let d_small = a.bhattacharyya_distance(&b);

    b.mean = vec![2.0, 2.0];
    let d_big = a.bhattacharyya_distance(&b);

    assert!(d0 >= 0.0);
    assert!(d_small > d0);
    assert!(d_big > d_small);
}

#[test]
fn numerical_stability_tiny_variance_does_not_nan() {
    let mut a = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    let mut b = KalmanCentroid::new(&[1.0, -1.0], 1e-4, 1e-2);

    // Pathological near-zero variances
    a.variance = vec![1e-20, 1e-30];
    b.variance = vec![1e-25, 1e-22];

    let d_m = a.mahalanobis_distance_sq(&[1.0, 1.0]);
    let d_b = a.bhattacharyya_distance(&b);

    assert!(d_m.is_finite());
    assert!(d_b.is_finite());
    assert!(d_m >= 0.0);
    assert!(d_b >= 0.0);
}

#[test]
#[should_panic]
fn update_panics_on_dimension_mismatch() {
    let mut c = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    c.update(&[1.0]); // wrong dim
}
