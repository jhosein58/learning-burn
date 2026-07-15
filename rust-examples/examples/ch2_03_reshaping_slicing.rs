use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};

type B = NdArray;

fn main() {
    let dev = Default::default();

    let t = Tensor::<B, 1, Int>::arange(0..12, &dev);
    let m = t.clone().reshape([3, 4]);
    let flat = m.clone().flatten::<1>(0, 1);
    let piece = m.clone().slice([0..2, 1..3]); // rows 0-1, cols 1-2
    let col = m.clone().narrow(1, 0, 1); // first column

    println!("reshaped =\n{}", m);

    let rows = m.select(0, Tensor::<B, 1, Int>::from_data([0, 2], &dev));

    println!("flat     =\n{}", flat);
    println!("piece    =\n{}", piece);
    println!("first col=\n{}", col);
    println!("rows 0,2 =\n{}", rows);
}
