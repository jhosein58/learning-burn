// Chapter 0 — Rust for Python people (no Burn yet, just the language).
//
// Variables, mutability, and types.
//
// Python:
//   x = 5          # a name bound to a value; rebindable to anything
//   x = "hello"    # totally fine in Python
//
// Rust: every value has a fixed type known at compile time, and bindings are
// immutable BY DEFAULT. You opt into mutation with `mut`.
fn main() {
    // Immutable by default. `x` is inferred to be i32.
    let x = 5;
    // x = 6; // <- would NOT compile: cannot assign twice to immutable `let`

    // Opt in to mutation with `mut`.
    let mut count = 0;
    count += 1;
    count += 1;

    // Type annotations are optional when the compiler can infer them.
    let pi: f64 = 3.14159;
    let name: &str = "burn";
    let ok: bool = true;

    // "Shadowing": re-use a name with a fresh `let`. Unlike Python rebinding,
    // this creates a NEW variable and can even change the type.
    let value = "42"; // &str
    let value = value.parse::<i32>().unwrap(); // now i32
    let value = value * 2; // still i32, = 84

    println!("x = {x}, count = {count}");
    println!("pi = {pi}, name = {name}, ok = {ok}");
    println!("value = {value}"); // 84
}
