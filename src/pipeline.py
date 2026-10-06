"""Pipeline completo y reproducible: datos -> entrenamiento -> umbrales -> evaluación.

Uso:
    python -m src.pipeline
"""
from src import data, decision, evaluate, train


def main(config_path="config.yaml"):
    pasos = [
        ("1/4 Datos", data.main),
        ("2/4 Entrenamiento", train.main),
        ("3/4 Umbrales de decisión", decision.main),
        ("4/4 Evaluación en test", evaluate.main),
    ]
    for nombre, paso in pasos:
        print(f"\n=== {nombre} ===")
        paso(config_path)


if __name__ == "__main__":
    main()
