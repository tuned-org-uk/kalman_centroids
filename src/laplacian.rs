use crate::centroids::KalmanCentroid;
use log::{debug, info, trace, warn};

/// Generic trait for distance-based graph construction
/// Allows plugging different statistical distance metrics
pub trait StatisticalDistance: Send + Sync {
    /// Compute distance between two centroids
    fn distance(&self, a: &KalmanCentroid, b: &KalmanCentroid) -> f64;

    /// Convert distance to edge weight
    fn distance_to_weight(&self, distance: f64) -> f64;

    /// Name of the distance metric (for logging)
    fn name(&self) -> &'static str;
}

/// Bhattacharyya distance for Gaussian centroids
pub struct BhattacharyyaDistance;

impl StatisticalDistance for BhattacharyyaDistance {
    fn distance(&self, a: &KalmanCentroid, b: &KalmanCentroid) -> f64 {
        trace!(
            "Computing Bhattacharyya distance: centroid_a.count={}, centroid_b.count={}, dim={}",
            a.count,
            b.count,
            a.mean.len()
        );

        let mut sum = 0.0;
        for i in 0..a.mean.len() {
            let sigma_avg = (a.variance[i] + b.variance[i]) / 2.0;
            let mean_diff = a.mean[i] - b.mean[i];

            // Simplified Bhattacharyya for diagonal Gaussians
            let term1 = 0.25 * (mean_diff * mean_diff) / sigma_avg.max(1e-9);
            let term2 = 0.5 * ((a.variance[i] * b.variance[i]).sqrt() / sigma_avg).ln();

            sum += term1 + term2;

            // Warn on numerical instabilities
            if !sum.is_finite() {
                warn!(
                    "Non-finite Bhattacharyya term at dim {}: term1={:.6}, term2={:.6}, \
                     var_a={:.6}, var_b={:.6}, mean_diff={:.6}",
                    i, term1, term2, a.variance[i], b.variance[i], mean_diff
                );
            }
        }

        trace!("Bhattacharyya distance computed: {:.6}", sum);
        sum
    }

    fn distance_to_weight(&self, distance: f64) -> f64 {
        let weight = (-distance).exp(); // Exponential kernel
        trace!(
            "Bhattacharyya distance {:.6} → weight {:.6}",
            distance, weight
        );
        weight
    }

    fn name(&self) -> &'static str {
        "Bhattacharyya"
    }
}

/// Kullback-Leibler divergence (asymmetric)
pub struct KLDivergence;

impl StatisticalDistance for KLDivergence {
    fn distance(&self, a: &KalmanCentroid, b: &KalmanCentroid) -> f64 {
        trace!(
            "Computing KL divergence: centroid_a.count={}, centroid_b.count={}, dim={}",
            a.count,
            b.count,
            a.mean.len()
        );

        let mut kl = 0.0;
        for i in 0..a.mean.len() {
            let var_ratio = a.variance[i] / b.variance[i].max(1e-9);
            let mean_diff = b.mean[i] - a.mean[i];

            // KL(a || b) for diagonal Gaussians
            let term = 0.5
                * (var_ratio + (mean_diff * mean_diff) / b.variance[i].max(1e-9) - 1.0
                    + var_ratio.ln());
            kl += term;

            // Warn on numerical instabilities
            if !kl.is_finite() {
                warn!(
                    "Non-finite KL term at dim {}: term={:.6}, var_ratio={:.6}, \
                     var_a={:.6}, var_b={:.6}, mean_diff={:.6}",
                    i, term, var_ratio, a.variance[i], b.variance[i], mean_diff
                );
            }
        }

        trace!("KL divergence computed: {:.6}", kl);
        kl
    }

    fn distance_to_weight(&self, distance: f64) -> f64 {
        let weight = 1.0 / (1.0 + distance); // Rational kernel
        trace!("KL divergence {:.6} → weight {:.6}", distance, weight);
        weight
    }

    fn name(&self) -> &'static str {
        "KL-Divergence"
    }
}
