use image::{GenericImageView, buffer::ConvertBuffer};
use imageproc::point::Point;
use imageproc::rect::Rect;
use num::{Num, ToPrimitive};
use std::path::PathBuf;
use structopt::StructOpt;
use structopt::clap::AppSettings;
use float_ord::FloatOrd;

mod corner_filter;
mod peak_finder;

#[derive(StructOpt)]
#[structopt(setting = AppSettings::ColoredHelp)]
#[structopt(setting = AppSettings::UnifiedHelpMessage)]
#[structopt(setting = AppSettings::DeriveDisplayOrder)]
struct Options {
	/// The image to process.
	image: PathBuf,

	/// Minumum squareness of checkerboard tiles (computed as max side / min side).
	#[structopt(long, short = "s")]
	#[structopt(value_name = "0..1")]
	#[structopt(default_value = "0.6")]
	min_squareness: f32,

	/// Minumum area of checkerboard tiles relative to the total image size.
	#[structopt(long, short = "a")]
	#[structopt(value_name = "0..1")]
	#[structopt(default_value = "0.03")]
	min_area: f32,

	/// Maximum residual of simplifying a contour to a quadrangle relative to image size.
	#[structopt(long, short = "r")]
	#[structopt(value_name = "0..1")]
	#[structopt(default_value = "0.005")]
	max_residual: f32,

	/// Dilation to perform before finding contours.
	#[structopt(long, short = "d")]
	#[structopt(value_name = "INTEGER")]
	#[structopt(default_value = "1")]
	dilation: u8,
}

#[show_image::main]
fn main() {
	if let Err(e) = do_main(Options::from_args()) {
		eprintln!("{}", e);
		std::process::exit(1);
	}
}

fn wait_space(events: &mut std::sync::mpsc::Receiver<show_image::event::WindowEvent>) {
	for event in events.iter() {
		if let show_image::event::WindowEvent::KeyboardInput(event) = event {
			if event.input.state.is_released() && event.input.key_code == Some(show_image::event::VirtualKeyCode::Space) {
				return;
			}
		}
	}
}

fn do_main(options: Options) -> Result<(), String> {
	let image = image::open(&options.image)
		.map_err(|e| format!("failed to open image: {}: {}", options.image.display(), e))?;
	let image = image.resize(image.width() * 4, image.height() * 4, image::imageops::FilterType::Lanczos3);
	let image = image.to_rgb8();

	let min_area = image.width() as f32 * image.height() as f32 * options.min_area.powi(2);
	let max_residual = image.width().min(image.height()) as f32 * options.max_residual;
	println!("minimum tile area in pixels: {}", min_area);
	println!("maximum average quadranglification residual in pixels: {}", max_residual);

	let grayscale: image::GrayImage = image.convert();
	let equalized = imageproc::contrast::equalize_histogram(&grayscale);

	let window = show_image::create_window("visualization", Default::default()).unwrap();
	let mut events = window.event_channel().unwrap();

	let gradients = corner_filter::cornerness_map(&grayscale, 4);
	let min = gradients.pixels().min_by_key(|p| FloatOrd(p[0])).unwrap()[0];
	let max = gradients.pixels().max_by_key(|p| FloatOrd(p[0])).unwrap()[0];
	let mut gradients_visual: image::GrayImage = image::ImageBuffer::new(gradients.width(), gradients.height());
	eprintln!("min, max cornerness: {}, {}", min, max);
	for (source, visual) in gradients.pixels().zip(gradients_visual.pixels_mut()) {
		visual[0] = ((source[0] - min) / (max - min) * 255.0) as u8;
	}

	println!("showing gradients");
	//let gradients: image::GrayImage = gradients.convert();
	window.set_image("gradients", gradients_visual).unwrap();
	wait_space(&mut events);

	println!("showing binary image");
	let binarized = imageproc::contrast::threshold(&equalized, 40);
	window.set_image("binarized", binarized.clone()).unwrap();
	wait_space(&mut events);

	println!("showing dilated image");
	let dilated = if options.dilation > 0 {
		imageproc::morphology::dilate(&binarized, imageproc::distance_transform::Norm::L1, options.dilation)
	} else {
		binarized.clone()
	};
	window.set_image("dilated", dilated.clone()).unwrap();
	wait_space(&mut events);

	let mut contours = imageproc::contours::find_contours::<i32>(&dilated);
	contours.retain(|c| c.points.len() >= 4 && c.parent == None);

	let magenta = image::Rgb([200, 0, 200]);
	let cyan = image::Rgb([0, 255, 255]);
	let green = image::Rgb([0, 255, 0]);
	let quads: Vec<_> = contours.iter().map(|c| fit_quadrangle(&c.points)).collect();

	let mut visual = image.clone();
	for (contour, quad) in contours.iter().zip(quads.iter()) {
		//draw_polygon(&mut visual, &contour.points, magenta);

		let bbox = match bounding_box(quad) {
			Some(x) => x,
			None => continue,
		};
		let area = bbox.width() as f32 * bbox.height() as f32;
		if area < min_area {
			println!("rejecting tile based on minimum area: {} < {}", area, min_area);
			continue;
		}

		let squareness = squareness(quad);
		if squareness < options.min_squareness {
			println!("rejecting tile based on squareness: {} < {}", squareness, options.min_squareness);
			continue;
		}

		let residual = residual(quad, &contour.points);
		if residual > max_residual {
			println!("rejecting tile based on quadranglification residual: {} > {}", residual, max_residual);
			continue;
		}

		//draw_polygon(&mut visual, quad, cyan);
		let search_radius = area.sqrt() * 0.10;
		println!("fine-tuning quads with search radius {}", search_radius);

		let tweaked_quad = [
			peak_finder::find_subpixel_peak(&gradients, quad[0].x, quad[0].y, search_radius.ceil() as i32, 4),
			peak_finder::find_subpixel_peak(&gradients, quad[1].x, quad[1].y, search_radius.ceil() as i32, 4),
			peak_finder::find_subpixel_peak(&gradients, quad[2].x, quad[2].y, search_radius.ceil() as i32, 4),
			peak_finder::find_subpixel_peak(&gradients, quad[3].x, quad[3].y, search_radius.ceil() as i32, 4),
		];

		draw_subpixel_points(&mut visual, &tweaked_quad[..], magenta);
	}

	println!("showing {} contours", contours.len());
	window.set_image("contours", visual).unwrap();
	wait_space(&mut events);

	Ok(())
}

fn draw_polygon<T, C>(image: &mut image::ImageBuffer<image::Rgb<u8>, C>, points: &[Point<i32>], color: image::Rgb<u8>)
where
	C: std::ops::Deref<Target = [u8]> + std::ops::DerefMut,
	T: num::NumCast,
{
	for (a, b) in points.iter().zip(points.iter().cycle().skip(1)) {
		let a = (a.x.to_i32().unwrap(), a.y.to_i32().unwrap());
		let b = (b.x.to_i32().unwrap(), b.y.to_i32().unwrap());
		imageproc::drawing::draw_antialiased_line_segment_mut(image, a, b, color, imageproc::pixelops::interpolate);
	}
}

fn draw_points<C>(image: &mut image::ImageBuffer<image::Rgb<u8>, C>, points: &[Point<i32>], color: image::Rgb<u8>)
where
	C: std::ops::Deref<Target = [u8]> + std::ops::DerefMut,
{
	for point in points {
		imageproc::drawing::draw_cross_mut(image, color, point.x.to_i32().unwrap(), point.y.to_i32().unwrap())
	}
}

fn draw_subpixel_points<C>(image: &mut image::ImageBuffer<image::Rgb<u8>, C>, points: &[Point<f32>], color: image::Rgb<u8>)
where
	C: std::ops::Deref<Target = [u8]> + std::ops::DerefMut,
{
	for point in points {
		imageproc::drawing::draw_line_segment_mut(image, (point.x - 8.0, point.y - 1.0), (point.x + 8.0, point.y - 1.0), color);
		imageproc::drawing::draw_line_segment_mut(image, (point.x - 8.0, point.y + 0.0), (point.x + 8.0, point.y + 0.0), color);
		imageproc::drawing::draw_line_segment_mut(image, (point.x - 8.0, point.y + 1.0), (point.x + 8.0, point.y + 1.0), color);
		imageproc::drawing::draw_line_segment_mut(image, (point.x - 1.0, point.y - 8.0), (point.x - 1.0, point.y + 8.0), color);
		imageproc::drawing::draw_line_segment_mut(image, (point.x + 0.0, point.y - 8.0), (point.x + 0.0, point.y + 8.0), color);
		imageproc::drawing::draw_line_segment_mut(image, (point.x + 1.0, point.y - 8.0), (point.x + 1.0, point.y + 8.0), color);
	}
}

fn bounding_box(points: &[Point<i32>]) -> Option<Rect> {
	if points.is_empty() {
		return None;
	}
	let mut min_x = points[0].x;
	let mut max_x = points[0].x;
	let mut min_y = points[0].y;
	let mut max_y = points[0].y;
	for point in points {
		min_x = min_x.min(point.x);
		max_x = max_x.max(point.x);
		min_y = min_y.min(point.y);
		max_y = max_y.max(point.y);
	}
	if min_x == max_x || min_y == max_y {
		None
	} else {
		Some(imageproc::rect::Rect::at(min_x, min_y).of_size((max_x - min_x) as u32, (max_y - min_y) as u32))
	}
}

fn fit_quadrangle<T>(points: &[Point<T>]) -> [Point<T>; 4]
where
	T: Copy + Num + ToPrimitive,
{
	assert!(points.len() >= 4);
	let (a, b) = get_max_distance_pair(points);

	// TODO: We now assume that one point must be in the section from a to b,
	// and one in the section from b to a.
	// For non-square quads, they could both be in the same section.
	let c = (a + 1..b)
		.max_by_key(|&i| float_ord::FloatOrd(norm(points[a] - points[i]) + norm(points[b] - points[i])))
		.unwrap();
	let d = (b + 1..points.len())
		.chain(0..a)
		.max_by_key(|&i| float_ord::FloatOrd(norm(points[a] - points[i]) + norm(points[b] - points[i])))
		.unwrap();

	[
		points[a],
		points[c],
		points[b],
		points[d],
	]
}

fn get_max_distance_pair<T>(points: &[Point<T>]) -> (usize, usize)
where
	T: Copy + Num + ToPrimitive,
{
	let mut longest_distance = 0.0;
	let mut best_a = 0;
	let mut best_b = 0;
	for a in 0..points.len() {
		for b in a..points.len() {
			let pair_distance = norm_squared(points[a] - points[b]);
			if pair_distance > longest_distance {
				longest_distance = pair_distance;
				best_a = a;
				best_b = b;
			}
		}
	}

	if best_a < best_b {
		(best_a, best_b)
	} else {
		(best_b, best_a)
	}
}

fn residual<T: Copy + Num + ToPrimitive>(simplified: &[Point<T>], original: &[Point<T>]) -> f32 {
	let mut residual = 0.0;
	for point in original {
		let mut min = f32::INFINITY;
		for (a, b) in simplified.iter().zip(simplified.iter().cycle().skip(1)) {
			let line = to_f32(*b - *a);
			min = min.min(rejection(to_f32(*point - *a), line));
		}
		residual += min.abs() / original.len() as f32;
	}
	residual
}

fn squareness<T: std::fmt::Debug + Copy + Num + ToPrimitive>(quad: &[Point<T>; 4]) -> f32 {
	let mut min = f32::INFINITY;
	let mut max = f32::NEG_INFINITY;
	for (&a, &b) in quad.iter().zip(quad.iter().cycle().skip(1)) {
		let side = norm(b - a);
		min = min.min(side);
		max = max.max(side);
	}
	min / max
}

fn norm<T: Copy + Num + ToPrimitive>(p: Point<T>) -> f32 {
	norm_squared(p).sqrt()
}

fn norm_squared<T: Copy + Num + ToPrimitive>(p: Point<T>) -> f32 {
	let x = p.x.to_f32().unwrap();
	let y = p.y.to_f32().unwrap();
	x * x + y * y
}

fn dot(a: Point<f32>, b: Point<f32>) -> f32 {
	a.x * b.x + a.y * b.y
}

fn projection(point: Point<f32>, target: Point<f32>) -> Point<f32> {
	let dot_target = dot(target, target);
	if dot_target == 0.0 {
		Point::new(0.0, 0.0)
	} else {
		let scale = dot(point, target) / dot_target;
		Point::new(scale * target.x, scale * target.y)
	}
}

fn rejection(point: Point<f32>, target: Point<f32>) -> f32 {
	let difference = point - projection(point, target);
	let x = difference.x;
	let y = difference.y;
	(x * x + y * y).sqrt()
}

fn to_f32<T: ToPrimitive>(point: Point<T>) -> Point<f32> {
	let x = point.x.to_f32().unwrap();
	let y = point.y.to_f32().unwrap();
	Point::new(x, y)
}
