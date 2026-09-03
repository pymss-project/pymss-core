from .config import load_config
def get_model_from_config(model_type, config_path, model_kwargs_override=None):
    """Instantiate a separation model from a model configuration file."""
    import importlib
    config = load_config(config_path)
    if model_type == "mdx23c":
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        return TFC_TDF_net(config), config
    if model_type == "htdemucs":
        from .modules.demucs4ht import get_model
        return get_model(config), config
    if model_type == "vr": raise ValueError("VR network modules do not use YAML config loading")
    models = {
        "mel_band_roformer": ("bs_roformer", "MelBandRoformer", "model"),
        "mel_band_conformer": ("bs_roformer", "MelBandConformer", "model"),
        "bs_roformer": ("bs_roformer", "BSRoformer", "model"),
        "bs_conformer": ("bs_roformer", "BSConformer", "model"),
        "bs_roformer_hyperace": ("bs_roformer", "BSRoformerHyperACE", "model"),
        "bandit": ("bandit.core.model", "MultiMaskMultiSourceBandSplitRNNSimple", "model"),
        "bandit_v2": ("bandit_v2.bandit", "Bandit", "kwargs"),
        "scnet": ("scnet", "SCNet", "model"),
        "apollo": ("look2hear.apollo", "Apollo", "model"),
    }
    if model_type not in models: raise ValueError(f"Model type {model_type} not supported")
    package, class_name, config_key = models[model_type]
    cls = getattr(importlib.import_module(f".modules.{package}", __package__), class_name)
    model_kwargs = dict(config[config_key])
    model_kwargs.update(model_kwargs_override or {})
    return cls(**model_kwargs), config