// Chapter 0 — Structs and methods (Rust's version of a class).
//
// Python:
//   class Point:
//       def __init__(self, x, y):
//           self.x = x
//           self.y = y
//       def norm(self):
//           return (self.x**2 + self.y**2) ** 0.5
//
// Rust splits this in two: a `struct` holds the DATA, and an `impl` block holds
// the BEHAVIOR (methods). There's no inheritance — you compose and use traits.
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    // An "associated function" (no self) — the idiomatic constructor, like __new__.
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    // A method: &self borrows the instance, like Python's `self`.
    fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    // A method that mutates needs &mut self.
    fn scale(&mut self, k: f64) {
        self.x *= k;
        self.y *= k;
    }
}

fn main() {
    let mut p = Point::new(3.0, 4.0);
    println!("norm = {}", p.norm()); // 5.0

    p.scale(2.0);
    println!("after scale: ({}, {})", p.x, p.y); // (6, 8)
    println!("norm = {}", p.norm()); // 10.0
}
