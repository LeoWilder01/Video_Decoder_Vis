# Fixed latent samples

The dots to the left of Input Latent select immutable initialization samples.
The first dot restores deterministic Gaussian noise (seed 42). Selecting any
sample again discards its in-memory edits and restores its original values.

Place additional `.safetensors` files in this directory, then reload the page.
Each file appears as a dot; hover or keyboard focus shows its filename.
Required tensor key: `latent`; shape: `[16,21,60,104]`; finite float values.
Required safetensors metadata:

- `layout`: `CTHW`
- `model_family`: `wan2.1`
- `stage`: `final_clean`
- `latent_space`: `diffusion`

Available imported samples: `Pirate.safetensors`, `Cooking.safetensors`, and
`Function.safetensors`. All are validated Wan 2.1 clean diffusion-space latents.
FPS was not supplied; playback uses the application's default 16 FPS.

Sample tensors and logs are local data and excluded from Git. Copy this folder's
sample files separately when moving to another computer.
