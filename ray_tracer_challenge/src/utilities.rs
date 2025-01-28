const EPSILON: f64 = 0.00001;

pub fn compare_float(x: f64, y: f64) -> bool {
    (x - y).abs() < EPSILON
}

pub fn compare_to_zero(x: f64) -> bool {
    compare_float(x, 0.0)
}
