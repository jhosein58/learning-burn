# Example 17.2 — A Dataset over a Folder of Images

A hand-written `Dataset` over `data/shapes/<class>/*.png`: `new()` lists the files, `get()` decodes one image. Labels come from the sorted folder names.

## Files

| File | What it is |
|---|---|
| `rust/src/main.rs` | Rust version + `#[test]`. |
| `python/main.py` | Python reference + `pytest` test. |

## Run & test

```bash
cargo run -- -c 17 -e 2 rust      # from the repo root
cargo run -- -c 17 -e 2 python
cargo test -p c17e2 --release     # the Rust parity test
pytest examples/chapter17/e02_image_folder/python/main.py      # the Python parity test
```

## Parity

Same classes, same label and pixel sum (4017) for item 7, same first batch `[8, 1, 12, 12]` with labels `[0,0,0,0,0,0,1,1]` and mean pixel.
