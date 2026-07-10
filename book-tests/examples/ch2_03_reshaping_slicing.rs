use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};

type Backend = NdArray;

fn main() {
    let device = Default::default();

    // 0..12 as a 1-D Int tensor, then viewed as 3x4.
    let t = Tensor::<Backend, 1, Int>::arange(0..12, &device);
    let m = t.clone().reshape([3, 4]);

    // Slicing uses ranges where Python uses colon syntax.
    let piece = m.clone().slice([0..2, 1..3]); // rows 0-1, cols 1-2
    let col = m.clone().slice([0..3, 0..1]); // first column

    // Gather specific rows (rows 0 and 2).
    let idx = Tensor::<Backend, 1, Int>::from_data(TensorData::from([0i64, 2]), &device);
    let rows = m.clone().select(0, idx);

    println!("reshaped =\n{}", m);
    println!("piece    =\n{}", piece);
    println!("first col=\n{}", col);
    println!("rows 0,2 =\n{}", rows);
}
