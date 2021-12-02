from pathlib import Path

base_path = Path(__file__).parent

_dm_code = base_path.joinpath(Path('dirichlet_multinomial.stan')).read_text()
_dm_multi_code = (
    base_path
    .joinpath(Path('dirichlet_multinomial_multi.stan'))
    .read_text()
)
_bnn_code = base_path.joinpath(Path('beta_binomial.stan')).read_text()

__all__ = ['_dm_code',  '_dm_multi_code', '_bnn_code']
