// Chapter 0 — Result and error handling (Rust has no exceptions).
//
// Python raises and catches:
//   def parse_sum(a, b):
//       return int(a) + int(b)     # raises ValueError on bad input
//   try:
//       print(parse_sum("2", "x"))
//   except ValueError as e:
//       print("bad:", e)
//
// Rust makes failure part of the RETURN TYPE: Result<T, E> is Ok(value) or
// Err(error). The `?` operator is the ergonomic "propagate the error upward",
// roughly like a re-raise.
fn parse_sum(a: &str, b: &str) -> Result<i32, std::num::ParseIntError> {
    let x = a.parse::<i32>()?; // on Err, return early with that Err
    let y = b.parse::<i32>()?;
    Ok(x + y)
}

fn main() {
    // The happy path.
    match parse_sum("2", "40") {
        Ok(v) => println!("sum = {v}"), // sum = 42
        Err(e) => println!("bad input: {e}"),
    }

    // The failing path — no crash, just an Err we choose to handle.
    match parse_sum("2", "x") {
        Ok(v) => println!("sum = {v}"),
        Err(e) => println!("bad input: {e}"), // bad input: invalid digit found in string
    }

    // Like Option, Result has helpers: unwrap_or gives a fallback.
    let total = parse_sum("oops", "5").unwrap_or(0);
    println!("total = {total}"); // 0
}
