# Examples

`RTM.ipynb` walks through transmitted surface wave RTM on a synthetic two-block
model with a vertical fault at x = 2500 m. Run it from this folder so that
`data_obs.npy` resolves.

| Section | Purpose |
| --- | --- |
| Observed data | Model vertical-component data for 5 sources and 40 receivers; saved as `data_obs.npy` (`nsrc, nrec, nt`) |
| Part 1 | Forward-propagate the source wavefield in the source-side velocity model |
| Part 2 | Back-propagate the recorded data from the receivers in the receiver-side velocity model |
| Part 3 | Apply the cross-correlation imaging condition |
| Part 4 | Depth-wise normalization of the image |
| Complete workflow | Migrate all shots and stack the images |

For field data, replace `data_obs.npy` with the ambient noise cross-correlations
(virtual sources at the source stations) in the same `(nsrc, nrec, nt)` layout.
