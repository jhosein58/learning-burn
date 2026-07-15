// Chapter 0 — Vectors and iterators (the Rust answer to lists + comprehensions).
//
// Python:
//   xs = [1, 2, 3, 4, 5]
//   squares = [x*x for x in xs if x % 2 == 1]   # [1, 9, 25]
//   total = sum(xs)                              # 15
//
// Rust uses Vec<T> for growable arrays and a lazy iterator chain
// (.iter().filter().map()...) that you finish with .collect() or a reducer.
fn main() {
    let xs = vec![1, 2, 3, 4, 5];

    // The comprehension above, as an iterator chain.
    let squares: Vec<i32> = xs
        .iter() // borrow each element
        .filter(|&&x| x % 2 == 1) // keep odds
        .map(|&x| x * x) // square them
        .collect(); // materialize into a Vec
    println!("squares = {squares:?}"); // [1, 9, 25]

    // Reducers consume the iterator to a single value.
    let total: i32 = xs.iter().sum();
    let max = xs.iter().max().unwrap();
    println!("sum = {total}, max = {max}"); // 15, 5

    // enumerate() mirrors Python's enumerate().
    for (i, x) in xs.iter().enumerate() {
        print!("xs[{i}]={x} ");
    }
    println!();

    // Building a Vec incrementally (like list.append in a loop).
    let mut doubled = Vec::new();
    for x in &xs {
        doubled.push(x * 2);
    }
    println!("doubled = {doubled:?}"); // [2, 4, 6, 8, 10]
}
