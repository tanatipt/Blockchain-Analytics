from dynaconf import Dynaconf

settings = Dynaconf(
    environments = True,
    settings_files = [
        "config/settings.yaml", 
        "config/secret.yaml"
    ]
)