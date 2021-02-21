use image::{ImageBuffer, Luma};


pub fn compute_cornerness<C>(image: &ImageBuffer<Luma<u8>, C>, x: u32, y: u32, radius: u32) -> f32
where
	C: std::ops::Deref<Target = [u8]>,
{
	// Compute horizontal and vertical sum of values.
	let mut sum_a = 0;
	for i in 0..=2 * radius {
		if i == radius { continue; }
		sum_a += image.get_pixel(x - radius + i, y)[0] as u32;
	}
	let mut sum_b = 0;
	for i in 0..=2 * radius {
		if i == radius { continue; }
		sum_b += image.get_pixel(x, y - radius + i)[0] as u32;
	}
	let score_straight = (sum_a as f32 - sum_b as f32).abs();

	// Compute sum of diagonals.
	let diagonal = (radius as f32 * 0.7071).round().max(1.0) as u32; // equivalent to: radius / sqrt(2.0)
	//let diagonal = radius;

	let mut sum_a = 0;
	for i in 0..=2 * diagonal {
		if i == diagonal { continue; }
		sum_a += image.get_pixel(x - diagonal + i, y - diagonal + i)[0] as u32;
	}

	let mut sum_b = 0;
	for i in 0..=2 * diagonal {
		if i == diagonal { continue; }
		sum_b += image.get_pixel(x + diagonal - i, y - diagonal + i)[0] as u32;
	}

	let score_diagonal = (sum_a as f32 - sum_b as f32).abs();

	// Report the highest score.
	(score_straight.max(score_diagonal * 1.414) / 255.0 / 2.0 / radius as f32).powi(2)
}

pub fn cornerness_map<C>(image: &ImageBuffer<Luma<u8>, C>, radius: u32) -> ImageBuffer<Luma<f32>, Vec<f32>>
where
	C: std::ops::Deref<Target = [u8]>,
{
	let mut output = ImageBuffer::new(image.width(), image.height());

	for y in radius..image.height() - radius {
		for x in radius..image.width() - radius {
			let cornerness = compute_cornerness(image, x, y, radius);
			*output.get_pixel_mut(x, y) = Luma([cornerness]);
		}
	}

	output
}
