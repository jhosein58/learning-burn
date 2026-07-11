// Chapter 0 — Ownership and move semantics.
//
// This is the ONE idea Python doesn't have. In Python every variable is a
// reference to a garbage-collected object, so two names can point at the same
// list and both keep it alive:
//
//   a = [1, 2, 3]
//   b = a          # b and a are the SAME list; nobody "owns" it
//   a.append(4)    # b sees the change too
//
// In Rust, each value has exactly ONE owner. Assigning a heap value (like a
// String or Vec) MOVES ownership; the old name can no longer be used.
fn main() {
    let a = String::from("hello");
    let b = a; // ownership MOVES from a to b
    // println!("{a}"); // <- would NOT compile: `a` was moved out of

    println!("b owns: {b}");

    // Want two independent copies? Ask explicitly with .clone().
    let c = b.clone();
    let d = b; // b moved into d; c is a separate copy
    println!("c = {c}, d = {d}");

    // Small "Copy" types (integers, floats, bools) are copied, not moved,
    // because they live entirely on the stack — cheap to duplicate.
    let n = 10;
    let m = n; // copy, not move
    println!("n = {n}, m = {m}"); // both still usable

    // When an owner goes out of scope, Rust frees the memory automatically —
    // no garbage collector, and no manual free().
}
