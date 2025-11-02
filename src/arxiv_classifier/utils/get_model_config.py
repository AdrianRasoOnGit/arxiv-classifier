#!usr/bin/python3

import inspect
from ..config.models_config import FMLP_CONFIG, FMLP_PROFILES_CONFIG

def get_model_config(profile: str | None = None, verbose: bool = True):

    if profile is None:
        caller = inspect.stack()[1].function.lower()
        if "tfidf" in caller:
            profile = "tfidf"
        elif "ngram" in caller:
            profile = "ngram"
        elif "embeddings" in caller or "embedding" in caller:
            profile = "embeddings"
        else:
            profile = "default"

    # Update the default config of the model
    especifics = FMLP_PROFILES_CONFIG.get(profile, {})
    config = FMLP_CONFIG | especifics | {"profile": profile}

    
    if verbose:
        print(f"Loaded model configuration profile: '{profile.upper()}'")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return config
