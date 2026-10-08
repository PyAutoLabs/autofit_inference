"""The config-name grammar {local,ral}_{numpy,jax_cpu}_{fp64,fp32} and its refusals."""

import pytest

from _autofit_inference_cli import CONFIG_NAME_RE, parse_config_name, validate_cli


@pytest.mark.parametrize(
    "name",
    ["local_numpy_fp64", "local_jax_cpu_fp64", "local_jax_cpu_fp32", "ral_numpy_fp64"],
)
def test_grammar_accepts(name):
    assert CONFIG_NAME_RE.match(name)


@pytest.mark.parametrize(
    "name", ["local_jax_gpu_fp64", "a100_jax_cpu_fp64", "local_numba_cpu_fp64", "local_numpy_mp"]
)
def test_grammar_has_no_gpu_leg_and_refuses_others(name):
    with pytest.raises(ValueError):
        parse_config_name(name)


def test_flags_that_disagree_with_the_name_are_refused():
    with pytest.raises(ValueError, match="--backend"):
        validate_cli("local_numpy_fp64", "jax_cpu", None, in_slurm=False)
    with pytest.raises(ValueError, match="--precision"):
        validate_cli("local_jax_cpu_fp64", None, "fp32", in_slurm=False)


def test_numpy_fp32_is_not_a_leg():
    with pytest.raises(ValueError, match="numpy_fp32"):
        validate_cli("local_numpy_fp32", None, None, in_slurm=False)


def test_where_must_match_the_hardware():
    with pytest.raises(ValueError, match="SLURM"):
        validate_cli("ral_numpy_fp64", None, None, in_slurm=False)
    with pytest.raises(ValueError, match="inside SLURM"):
        validate_cli("local_numpy_fp64", None, None, in_slurm=True)
    assert validate_cli("ral_jax_cpu_fp32", None, None, in_slurm=True) == (
        "ral",
        "jax_cpu",
        "fp32",
    )
