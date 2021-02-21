use image::{ImageBuffer, Luma};
use imageproc::point::Point;

pub fn find_subpixel_peak<C>(image: &ImageBuffer<Luma<f32>, C>, x: i32, y: i32, search_radius: i32, tune_radius: i32) -> Point<f32>
where
	C: std::ops::Deref<Target = [f32]>,
{
	let peak = find_peak(image, x, y, search_radius);
	finetune_peak(image, peak.x, peak.y, tune_radius)
	// Point::new(peak.x as f32, peak.y as f32)
}


fn find_peak<C>(image: &ImageBuffer<Luma<f32>, C>, x: i32, y: i32, radius: i32) -> Point<i32>
where
	C: std::ops::Deref<Target = [f32]>,
{
	let mut best_x = x;
	let mut best_y = y;
	let mut best_score = f32::NEG_INFINITY;

	let base_x = x;
	let base_y = y;

	for y in -radius ..= radius {
		for x in -radius ..= radius {
			if x * x + y * y > radius * radius {
				continue
			}
			let x = base_x + x;
			let y = base_y + y;
			let score = image.get_pixel(x as u32, y as u32)[0];
			if score > best_score {
				best_x = x;
				best_y = y;
				best_score = score;
			}
		}
	}

	Point::new(best_x, best_y)
}

fn finetune_peak<C>(image: &ImageBuffer<Luma<f32>, C>, x: i32, y: i32, radius: i32) -> Point<f32>
where
	C: std::ops::Deref<Target = [f32]>,
{
	let base_x = x;
	let base_y = y;
	let mut avg_x = 0.0;
	let mut avg_y = 0.0;
	let count = (radius as f32 * 2.0 + 1.0).powi(2);
	for y in -radius ..= radius {
		for x in radius ..= radius {
			let score = image.get_pixel((base_x + x) as u32, (base_y + y) as u32)[0];
			avg_x += score * x as f32 / count;
			avg_y += score * y as f32 / count;
		}
	}

	Point::new(base_x as f32 + avg_x, base_y as f32 + avg_y)
}
