use burn::backend::NdArray;
use burn::tensor::Tensor;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::register_font;
use plotters::style::text_anchor::{HPos, Pos, VPos};

type Backend = NdArray;

/// Everything below is in "design pixels", multiplied by this on the way out.
/// Bump it for a sharper PNG; the layout stays identical.
const SCALE: u32 = 4;

const INK: RGBColor = RGBColor(60, 60, 60);
const GREY: RGBColor = RGBColor(120, 120, 120);
const GRID: RGBColor = RGBColor(238, 238, 238);
const BLUE: RGBColor = RGBColor(62, 101, 153);
const TEAL: RGBColor = RGBColor(37, 110, 122);

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

    // the same four numbers as the e08 example
    let input = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);

    let min = input.clone().min();
    let max = input.clone().max();
    let scaled = (input.clone() - min.clone()).div(max - min);

    let before: Vec<f32> = input.into_data().to_vec().unwrap();
    let after: Vec<f32> = scaled.into_data().to_vec().unwrap();

    // next to the crate, not wherever cargo happened to be invoked from
    let path = format!("{}/normalisation.png", env!("CARGO_MANIFEST_DIR"));
    let root = BitMapBackend::new(&path, (1280 * SCALE, 420 * SCALE)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let root = root
        .titled(
            "Min-max normalisation squeezes any range into [0, 1]",
            ("sans-serif", 21 * SCALE).into_font().color(&INK),
        )
        .unwrap();

    let (left, right) = root.split_horizontally(640 * SCALE);

    panel(&left, "before", &before, 0, BLUE);
    panel(&right, "after  (x - min) / (max - min)", &after, 3, TEAL);

    root.present().unwrap();
    println!("wrote {path}");
}

/// One bar chart, with each bar's value written above it.
fn panel(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    values: &[f32],
    decimals: usize,
    colour: RGBColor,
) {
    let top = values.iter().copied().fold(f32::MIN, f32::max) * 1.18;

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 17 * SCALE).into_font().color(&INK))
        .margin(25 * SCALE)
        .x_label_area_size(30 * SCALE)
        .y_label_area_size(45 * SCALE)
        // one slot per element, so the bars sit between tick marks rather than on them
        // (the range is inclusive once segmented, hence len - 1)
        .build_cartesian_2d((0..values.len() as u32 - 1).into_segmented(), 0f32..top)
        .unwrap();

    // whole numbers on the left, one decimal on the right
    let y_decimals = usize::from(decimals > 0);

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(GRID)
        .light_line_style(TRANSPARENT)
        .axis_style(RGBColor(210, 210, 210))
        .label_style(("sans-serif", 12 * SCALE).into_font().color(&GREY))
        .x_label_formatter(&|v| match v {
            SegmentValue::CenterOf(i) => format!("[{i}]"),
            _ => String::new(),
        })
        .y_label_formatter(&|v| format!("{v:.y_decimals$}"))
        .y_labels(6)
        .draw()
        .unwrap();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(colour.filled())
                .margin(30 * SCALE)
                .data(values.iter().enumerate().map(|(i, &v)| (i as u32, v))),
        )
        .unwrap();

    // value labels, sitting just above each bar
    let label = ("sans-serif", 15 * SCALE)
        .into_font()
        .color(&INK)
        .pos(Pos::new(HPos::Center, VPos::Bottom));

    chart
        .draw_series(values.iter().enumerate().map(|(i, &v)| {
            Text::new(
                format!("{v:.decimals$}"),
                (SegmentValue::CenterOf(i as u32), v + top * 0.03),
                label.clone(),
            )
        }))
        .unwrap();
}
