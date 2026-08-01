use burn::backend::NdArray;
use burn::tensor::backend::Backend as _;
use burn::tensor::{Distribution, Tensor};
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::register_font;

type Backend = NdArray;

const N: usize = 20_000;
const BINS: usize = 40;

/// Everything below is in "design pixels", multiplied by this on the way out.
/// Bump it for a sharper PNG; the layout stays identical.
const SCALE: u32 = 4;

/// ab_glyph ships a rasteriser but no font, so point it at one on disk.
fn load_font() {
    const CANDIDATES: [&str; 3] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ];

    let bytes = CANDIDATES
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .expect("no usable font found");

    if register_font("sans-serif", FontStyle::Normal, bytes.leak()).is_err() {
        panic!("font file was not valid OpenType");
    }
}

fn main() {
    load_font();

    let device = Default::default();

    // fix the random state: same draws, same picture, every run
    Backend::seed(&device, 42);

    // same two distributions as e05, but 20k draws each instead of 2x3
    let u = Tensor::<Backend, 1>::random([N], Distribution::Default, &device);
    let n = Tensor::<Backend, 1>::random([N], Distribution::Normal(0.0, 1.0), &device);

    // pull the numbers back to the host
    let u: Vec<f32> = u.into_data().to_vec().unwrap();
    let n: Vec<f32> = n.into_data().to_vec().unwrap();

    // next to the crate, not wherever cargo happened to be invoked from
    let path = format!("{}/distributions.png", env!("CARGO_MANIFEST_DIR"));
    let panel_width = 1500 * SCALE;
    let panel_height = 400 * SCALE;
    let root = BitMapBackend::new(&path, (panel_width, panel_height)).into_drawing_area();
    // fill before titling: the buffer starts black, and titled() only hands back the area below
    root.fill(&WHITE).unwrap();
    let root = root
        .titled(
            "Same function, two shapes of randomness",
            ("sans-serif", 26 * SCALE),
        )
        .unwrap();
    // uniform on the left, normal on the right
    let (left, right) = root.split_horizontally(panel_width / 2 );

    panel(&left, "uniform [0, 1)", &u, 0.0, 1.0, BLUE);
    panel(&right, "normal (0, 1)", &n, -4.0, 4.0, RGBColor(85, 107, 47)
);

    root.present().unwrap();
    println!("wrote {path}");
}

fn panel(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    xs: &[f32],
    lo: f32,
    hi: f32,
    colour: RGBColor,
) {
    let width = (hi - lo) / BINS as f32;

    // only needed to pick the y range; plotters does the actual binning
    let mut counts = vec![0u32; BINS];
    for &x in xs {
        let bin = (((x - lo) / width) as usize).min(BINS - 1);
        counts[bin] += 1;
    }
    let peak = *counts.iter().max().unwrap();

    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;

    let mut chart = ChartBuilder::on(area)
        .caption(
            format!("{title}   n={}  mean={mean:.3}  std={:.3}", xs.len(), var.sqrt()),
            ("sans-serif", 15 * SCALE),
        )
        .margin(10 * SCALE)
        .x_label_area_size(30 * SCALE)
        .y_label_area_size(55 * SCALE)
        // use_floor, not use_round: rounding would split the edge bins in half
        .build_cartesian_2d((lo..hi).step(width).use_floor(), 0..peak + peak / 10)
        .unwrap();

    chart
        .configure_mesh()
        .label_style(("sans-serif", 12 * SCALE))
        .draw()
        .unwrap();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(colour.filled())
                .margin(1)
                .data(xs.iter().map(|&x| (x, 1))),
        )
        .unwrap();
}
