from sleep2mi.smoke import run_synthetic_smoke


def test_synthetic_smoke() -> None:
    result = run_synthetic_smoke(seed=67)
    assert result["encoder_parameters"] == 68322
    assert result["embedding_dimension"] == 32

