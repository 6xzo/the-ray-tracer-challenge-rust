/// Constant for use in comparing floats within the ray tracer
const EPSILON: f64 = 0.00001;

/// Compares the two floats to see if they are epsilon away from each other.
/// See the definition of epsilon to see the resolution of the compare.
/// Units of measure are ignored.
pub fn compare_float(x: f64, y: f64) -> bool {
    (x - y).abs() < EPSILON
}

/// Compares the float to see if it is within epsilon of 0.0.
/// See the definition of epsilon to see the resolution of the compare.
/// Units of measure are ignored.
pub fn compare_to_zero(x: f64) -> bool {
    compare_float(x, 0.0)
}
