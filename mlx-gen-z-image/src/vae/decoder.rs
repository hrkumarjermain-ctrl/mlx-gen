//! VAE decoder assembly + `Vae::decode`. Port of `Decoder.__call__` / `VAE.decode`:
//! conv_in → mid-block → up-blocks → GroupNorm-out → SiLU → conv_out, with the scale/shift
//! that maps latents to the decoder's input range. NCHW throughout.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use super::conv_layers::{ConvLayer, ConvNormOut};
use super::encoder::{Encoder, VaeEncoderConfig};
use super::mid_block::UNetMidBlock;
use super::up_decoder_block::UpDecoderBlock;
use mlx_gen::nn::silu;
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::vae_tiling::tiled_decode;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder, Result};

/// Per-up-block `(num_resnet_layers, add_upsample)`.
#[derive(Debug, Clone)]
pub struct VaeDecoderConfig {
    pub up_blocks: Vec<(usize, bool)>,
}

impl VaeDecoderConfig {
    /// The production Z-Image VAE decoder: 4 up-blocks of 3 resnets, upsampling on the first 3.
    pub fn default_z_image() -> Self {
        Self {
            up_blocks: vec![(3, true), (3, true), (3, true), (3, false)],
        }
    }
}

pub struct Decoder {
    conv_in: ConvLayer,
    mid_block: UNetMidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: ConvNormOut,
    conv_out: ConvLayer,
    /// Output pixels per latent pixel: 2 per upsampling up-block (×8 for the production decoder).
    spatial_scale: i32,
}

impl Decoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeDecoderConfig) -> Result<Self> {
        // Support an empty top-level prefix (sub-module prefixes are always non-empty).
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let up_blocks = cfg
            .up_blocks
            .iter()
            .enumerate()
            .map(|(i, &(layers, up))| {
                UpDecoderBlock::from_weights(w, &p(&format!("up_blocks.{i}")), layers, up)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: ConvLayer::from_weights(w, &p("conv_in"))?,
            mid_block: UNetMidBlock::from_weights(w, &p("mid_block"))?,
            up_blocks,
            conv_norm_out: ConvNormOut::from_weights(w, &p("conv_norm_out"))?,
            conv_out: ConvLayer::from_weights(w, &p("conv_out"))?,
            spatial_scale: 1 << cfg.up_blocks.iter().filter(|(_, up)| *up).count(),
        })
    }

    /// Output pixels per latent pixel (2 per upsampling up-block).
    pub fn spatial_scale(&self) -> i32 {
        self.spatial_scale
    }

    /// Quantize the decoder's only quantizable Linears — the mid-block attention (conv_in/out,
    /// up-blocks, and norms are conv/norm, not quantized).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mid_block.quantize(bits)
    }

    /// `latents` NCHW → image NCHW (3 channels, spatial ×8).
    pub fn forward(&self, latents: &Array) -> Result<Array> {
        self.forward_tail(&self.forward_head(latents)?)
    }

    /// conv_in → mid-block (incl. its global spatial attention), at latent resolution. The tiled
    /// decode runs this ONCE on the full latent so the attention sees the whole image.
    pub fn forward_head(&self, latents: &Array) -> Result<Array> {
        let h = self.conv_in.forward(latents)?;
        self.mid_block.forward(&h)
    }

    /// up-blocks → GroupNorm-out → SiLU → conv_out: the ×8 upsampling tail, where a full-resolution
    /// decode spikes — the part [`Vae::decode_tiled`] tiles.
    pub fn forward_tail(&self, head: &Array) -> Result<Array> {
        let mut h = head.clone();
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        h = self.conv_norm_out.forward(&h)?;
        h = silu(&h)?;
        self.conv_out.forward(&h)
    }
}

/// The Z-Image VAE. `decode` undoes the latent scale/shift then runs the decoder; `encode`
/// (img2img) runs the encoder and maps the predicted mean into latent space. The encoder is
/// optional so a decode-only `Vae` can still be built from decoder weights alone.
pub struct Vae {
    decoder: Decoder,
    encoder: Option<Encoder>,
    scaling_factor: f32,
    shift_factor: f32,
}

impl Vae {
    pub const SCALING_FACTOR: f32 = 0.3611;
    pub const SHIFT_FACTOR: f32 = 0.1159;

    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeDecoderConfig) -> Result<Self> {
        Ok(Self {
            decoder: Decoder::from_weights(w, prefix, cfg)?,
            encoder: None,
            scaling_factor: Self::SCALING_FACTOR,
            shift_factor: Self::SHIFT_FACTOR,
        })
    }

    /// Like [`Vae::from_weights`] but with caller-supplied latent `scaling_factor` / `shift_factor`.
    /// The AutoencoderKL structure (16-ch, GroupNorm-32, the decoder's scale/shift de-norm math) is
    /// shared across diffusers 16-ch VAEs; only the two latent-normalization constants differ between
    /// families. Z-Image uses the [`Vae::SCALING_FACTOR`] / [`Vae::SHIFT_FACTOR`] defaults; SD3.5
    /// reuses this same module with its own `1.5305` / `0.0609` factors (mlx-gen-sd3, sc-7863).
    pub fn from_weights_with_factors(
        w: &Weights,
        prefix: &str,
        cfg: &VaeDecoderConfig,
        scaling_factor: f32,
        shift_factor: f32,
    ) -> Result<Self> {
        Ok(Self {
            decoder: Decoder::from_weights(w, prefix, cfg)?,
            encoder: None,
            scaling_factor,
            shift_factor,
        })
    }

    /// The latent `scaling_factor` this VAE de-normalizes with (`decode`: `z/scale + shift`).
    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    /// The latent `shift_factor` this VAE de-normalizes with.
    pub fn shift_factor(&self) -> f32 {
        self.shift_factor
    }

    /// Attach the img2img encoder, loaded from `prefix` (the diffusers `encoder.*` tree, remapped
    /// to the crate's internal naming by [`crate::loader::remap_vae_encoder`]).
    pub fn with_encoder(
        mut self,
        w: &Weights,
        prefix: &str,
        cfg: &VaeEncoderConfig,
    ) -> Result<Self> {
        self.encoder = Some(Encoder::from_weights(w, prefix, cfg)?);
        Ok(self)
    }

    /// Quantize the VAE's quantizable Linears (the decoder's — and, if loaded, the encoder's —
    /// mid-block spatial attention) to Q4/Q8. The VAE is otherwise all conv, so this is the full
    /// set the fork's `nn.quantize(vae, …)` hits. Output is pixel-unchanged in practice (the VAE
    /// quant is measurably 0% px on the decode), so this is for memory/`nn.quantize` faithfulness.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.decoder.quantize(bits)?;
        if let Some(encoder) = self.encoder.as_mut() {
            encoder.quantize(bits)?;
        }
        Ok(())
    }

    /// Image NCHW `[1,3,H,W]` (or `[1,3,1,H,W]`) → latent `[1,16,H/8,W/8]`. Port of the fork's
    /// `VAE.encode` composed with `VAEUtil.encode`'s 5-D→4-D fixup: run the encoder, take the
    /// distribution **mean** (first half of the channels), then map to latent space as
    /// `(mean - shift) * scaling`.
    pub fn encode(&self, image: &Array) -> Result<Array> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            Error::Msg("z_image VAE encoder not loaded (img2img unavailable)".into())
        })?;
        let sh = image.shape();
        let image4 = if sh.len() == 5 {
            image.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            image.clone()
        };
        let h = encoder.forward(&image4)?; // [1, 2C, H/8, W/8]
        if h.shape()[1] % 2 != 0 {
            return Err(Error::Msg(format!(
                "z-image vae encode: expected an even (2C: mean|logvar) channel count, got {}",
                h.shape()[1]
            )));
        }
        let c = h.shape()[1] / 2;
        let idx = Array::from_slice(&(0..c).collect::<Vec<i32>>(), &[c]);
        let mean = h.take_axis(&idx, 1)?; // first C channels
        Ok(multiply(
            &subtract(&mean, Array::from_slice(&[self.shift_factor], &[1]))?,
            Array::from_slice(&[self.scaling_factor], &[1]),
        )?)
    }

    /// The tiling geometry of this still-image VAE: the decoder's spatial scale (×8 in production),
    /// singleton frame axis.
    pub fn tiling(&self) -> VaeTiling {
        VaeTiling {
            spatial_scale: self.decoder.spatial_scale(),
            temporal_scale: 1,
            causal_temporal: false,
        }
    }

    /// `(B, C, 1, H, W)` or `(B, C, H, W)` latents → de-normalized `(B, C, H, W)` decoder input.
    fn denormalize(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        let latents4 = if sh.len() == 5 {
            // squeeze the (size-1) frame axis: (B,C,1,H,W) -> (B,C,H,W)
            latents.reshape(&[sh[0], sh[1], sh[3], sh[4]])?
        } else {
            latents.clone()
        };
        Ok(add(
            &multiply(
                &latents4,
                Array::from_slice(&[1.0 / self.scaling_factor], &[1]),
            )?,
            Array::from_slice(&[self.shift_factor], &[1]),
        )?)
    }

    /// `latents`: `(B, C, F, H, W)` (F squeezed) or `(B, C, H, W)` → image `(B, 3, 1, H·8, W·8)`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let decoded = self.decoder.forward(&self.denormalize(latents)?)?;
        let d = decoded.shape();
        Ok(decoded.reshape(&[d[0], d[1], 1, d[2], d[3]])?) // restore frame axis
    }

    /// Memory-bounded decode: the head (conv_in → mid-block attention) runs once on the full latent,
    /// then the ×8 upsampling tail is decoded in overlapping tiles and trapezoidally blended
    /// ([`tiled_decode`]), so the peak is one tile's tail instead of the full-resolution one. Falls
    /// back to [`Self::decode`] when `cfg` doesn't tile this size. Same output shape as `decode`.
    ///
    /// Unlike the Qwen-Image VAE (per-pixel RMS norms), this decoder's up-blocks use **GroupNorm**,
    /// whose statistics span the tile — so a tiled decode is close to, not identical with, the
    /// single-pass one (the classic diffusers tiled-VAE trade-off, hidden by the overlap blend).
    pub fn decode_tiled(
        &self,
        latents: &Array,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let head = self.decoder.forward_head(&self.denormalize(latents)?)?; // [B, C, h, w]
        let s = head.shape();
        let (h, w) = (s[2], s[3]);
        if !cfg.needs_tiling(self.tiling(), 1, h, w) {
            let out = self.decoder.forward_tail(&head)?;
            let o = out.shape();
            return Ok(out.reshape(&[o[0], o[1], 1, o[2], o[3]])?);
        }
        let head5 = head.reshape(&[s[0], s[1], 1, h, w])?;
        let plan = cfg.plan(self.tiling(), 1, h, w);
        // NCTHW: channel axis 1, tiled axes [2, 3, 4] (T is the singleton).
        tiled_decode(&head5, &plan, [2, 3, 4], cancel, |tile| {
            let t = tile.shape();
            let out = self
                .decoder
                .forward_tail(&tile.reshape(&[t[0], t[1], t[3], t[4]])?)?;
            let o = out.shape();
            Ok(out.reshape(&[o[0], o[1], 1, o[2], o[3]])?)
        })
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }
}

/// The native decoder for the Flux1 / Z-Image latent space (the behavior-preserving default of the PiD
/// decode seam, sc-7844). This `Vae` is reused verbatim by every crate in that latent space — Z-Image,
/// FLUX.1, Boogu, Chroma — so this single impl makes all of them PiD-swappable (sc-7846). Delegates to
/// the inherent [`Vae::decode`], which accepts the 4-D `(B, C, H, W)` normalized latent the engines
/// hand the seam (it re-adds the singleton frame axis on output). A PiD decoder for this same latent
/// space (`mlx-gen-pid`, sc-7843) implements the same trait, so a generation can swap between them at
/// the decode call site.
impl LatentDecoder for Vae {
    fn decode(&self, latents: &Array) -> Result<Array> {
        Vae::decode(self, latents)
    }
}
