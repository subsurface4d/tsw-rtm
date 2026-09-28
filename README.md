# SeisRTM: Transmitted Surface Wave Reverse Time Migration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/DOI-10.1029%2F2020JB020381-blue.svg)](https://doi.org/10.1029/2020JB020381)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Research code for **"Ambient Noise Surface Wave Reverse Time Migration for Fault Imaging"** (Li, Li, Gu, Gao & Zhang, *JGR: Solid Earth*, 2020).

Surface waves that cross a fault are partly transmitted and partly reflected, and the transmitted waves carry the velocity contrast across it. Transmitted surface wave RTM images the fault by propagating the source wavefield in a source-side velocity model, back-propagating the recorded data (e.g. ambient noise cross-correlations) in a receiver-side velocity model, and cross-correlating the two wavefields. The elastic wave equation is solved with [Deepwave](https://github.com/ar4/deepwave).

## Repository structure

```
tsw-rtm/
├── seisrtm/
│   ├── model/        ElasticModel (vp, vs, rho) on a regular 2D grid
│   ├── survey/       Source, Receiver, Survey, SeismicData
│   ├── propagator/   ElasticPropagator (Deepwave) + Hicks source/receiver interpolation
│   ├── signal/       Butterworth filters, f-k analysis and filtering
│   └── utils/        wavelets, smoothing, wavefield animation
└── examples/         RTM.ipynb tutorial + example observed data
```

## Installation

```bash
git clone https://github.com/subsurface4d/tsw-rtm.git
cd tsw-rtm
conda create -n seisrtm python=3.10 -y && conda activate seisrtm
pip install -e .
```

A CUDA GPU speeds up the propagator but is not required; the example runs on CPU.

## Usage

```python
from seisrtm.model import ElasticModel
from seisrtm.survey import Survey, Source, Receiver
from seisrtm.propagator import ElasticPropagator
from seisrtm.utils import wavelet

model = ElasticModel(ox, oz, dx, dz, nx, nz, vp=vp, vs=vs, rho=rho, free_surface=True, nabc=20)
source = Source(nt=nt, dt=dt, f0=f0)
source.add_source([x_src, z_src], wavelet(nt, dt, f0), 'vz')
receiver = Receiver(nt=nt, dt=dt)
receiver.add_receiver([x_rec, z_rec], 'vz')

F = ElasticPropagator(model, Survey(source=source, receiver=receiver, device='cpu'))
data = F(save_wavefield=True)   # data.data['vz'], data.data['wavefields_vz']
```

The full workflow — observed data, forward and backward propagation, imaging condition, and stacking over shots — is in [`examples/RTM.ipynb`](examples/RTM.ipynb); see [`examples/README.md`](examples/README.md).

## Citation

```bibtex
@article{Li2020TSWRTM,
  title   = {Ambient Noise Surface Wave Reverse Time Migration for Fault Imaging},
  author  = {Li, Haipeng and Li, J. and Gu, N. and Gao, J. and Zhang, H.},
  journal = {Journal of Geophysical Research: Solid Earth},
  volume  = {125}, number = {12}, pages = {e2020JB020381}, year = {2020},
  doi     = {10.1029/2020JB020381}
}
```

## License

MIT — see [LICENSE](LICENSE). Built on [Deepwave](https://github.com/ar4/deepwave).
