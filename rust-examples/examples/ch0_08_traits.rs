// Chapter 0 — Traits: shared behavior without inheritance.
//
// Python uses duck typing (or ABCs): anything with a .area() method works.
//   class Circle:
//       def area(self): ...
//   class Square:
//       def area(self): ...
//   def total_area(shapes): return sum(s.area() for s in shapes)
//
// Rust names that contract explicitly with a `trait`, and each type provides an
// `impl`. Functions can then accept "any type that implements Area". This is
// exactly how Burn works: a Backend is a trait, and NdArray/Wgpu/etc. implement it.
trait Area {
    fn area(&self) -> f64;
}

struct Circle {
    r: f64,
}
struct Square {
    side: f64,
}

impl Area for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.r * self.r
    }
}

impl Area for Square {
    fn area(&self) -> f64 {
        self.side * self.side
    }
}

// Generic over "any T that implements Area" — the compiler picks the right
// area() for each concrete type (static dispatch, zero runtime cost).
fn describe<T: Area>(shape: &T) {
    println!("area = {:.4}", shape.area());
}

fn main() {
    let c = Circle { r: 1.0 };
    let s = Square { side: 2.0 };

    describe(&c); // area = 3.1416
    describe(&s); // area = 4.0000

    // A trait object (&dyn Area) lets a single collection hold mixed types,
    // closer to the Python list-of-shapes feel (dynamic dispatch).
    let shapes: Vec<&dyn Area> = vec![&c, &s];
    let total: f64 = shapes.iter().map(|s| s.area()).sum();
    println!("total area = {total:.4}"); // 7.1416
}
