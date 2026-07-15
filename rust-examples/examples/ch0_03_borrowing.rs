// Chapter 0 — Borrowing: references instead of giving ownership away.
//
// In Python, passing an object to a function passes a reference, and the
// function can mutate it in place:
//
//   def add_one(xs):      # xs is the same list
//       xs.append(1)
//
// In Rust you "borrow" with & (shared, read-only) or &mut (exclusive, writable),
// so a function can use a value WITHOUT taking ownership of it.
fn main() {
    let s = String::from("burn");

    // Shared borrow: len() only needs to read, so it takes &String.
    let n = char_count(&s);
    println!("'{s}' has {n} chars"); // s is still ours to use

    // Mutable borrow: the function may modify what we lend it.
    let mut greeting = String::from("hello");
    shout(&mut greeting);
    println!("{greeting}"); // HELLO!

    // The rule: you can have many shared (&) borrows OR one mutable (&mut)
    // borrow at a time — never both. This is what prevents data races.
}

fn char_count(text: &String) -> usize {
    text.chars().count()
}

fn shout(text: &mut String) {
    *text = text.to_uppercase();
    text.push('!');
}
