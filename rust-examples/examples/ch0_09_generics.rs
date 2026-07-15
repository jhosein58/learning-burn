// Chapter 0 — Generics: one function, many types, checked at compile time.
//
// Python is generic by default (duck typing); type hints are optional and not
// enforced at runtime:
//   def largest(xs): return max(xs)   # works for any comparable sequence
//
// Rust generics are monomorphized: the compiler stamps out a specialized copy
// for each concrete type you actually use, so there's no runtime cost and full
// type checking. Trait bounds (T: PartialOrd) say what the type must support.
fn largest<T: PartialOrd + Copy>(items: &[T]) -> T {
    let mut biggest = items[0];
    for &item in &items[1..] {
        if item > biggest {
            biggest = item;
        }
    }
    biggest
}

// A generic struct — like Point<T> that works for f64 or i32.
struct Pair<T> {
    first: T,
    second: T,
}

impl<T: std::fmt::Display> Pair<T> {
    fn show(&self) {
        println!("({}, {})", self.first, self.second);
    }
}

fn main() {
    let ints = [3, 7, 2, 9, 4];
    let floats = [1.5, 0.5, 2.25];

    // Same function, two different element types.
    println!("largest int   = {}", largest(&ints)); // 9
    println!("largest float = {}", largest(&floats)); // 2.25

    let words = ["burn", "rust", "tensor"];
    println!("largest word  = {}", largest(&words)); // "tensor" (lexicographic)

    let p = Pair {
        first: 1,
        second: 2,
    };
    p.show(); // (1, 2)
}
