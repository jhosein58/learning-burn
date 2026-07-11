// Chapter 0 — Enums, Option, and pattern matching.
//
// Python leans on None and isinstance checks:
//   def first(xs):
//       return xs[0] if xs else None
//   x = first([])
//   if x is None: ...
//
// Rust has no null. "Maybe a value" is the Option<T> enum: Some(v) or None,
// and the compiler FORCES you to handle both cases via `match`.
enum Shape {
    Circle(f64),       // radius
    Rectangle(f64, f64), // width, height
}

fn area(shape: &Shape) -> f64 {
    // match is exhaustive: forget a variant and it won't compile.
    match shape {
        Shape::Circle(r) => std::f64::consts::PI * r * r,
        Shape::Rectangle(w, h) => w * h,
    }
}

fn first(xs: &[i32]) -> Option<i32> {
    if xs.is_empty() {
        None
    } else {
        Some(xs[0])
    }
}

fn main() {
    let shapes = [Shape::Circle(1.0), Shape::Rectangle(2.0, 3.0)];
    for s in &shapes {
        println!("area = {:.4}", area(s)); // 3.1416, 6.0000
    }

    // Handling Option: match, or the convenient helpers.
    match first(&[10, 20, 30]) {
        Some(v) => println!("first = {v}"), // first = 10
        None => println!("empty"),
    }

    // unwrap_or supplies a default instead of crashing on None.
    let x = first(&[]).unwrap_or(-1);
    println!("x = {x}"); // -1
}
