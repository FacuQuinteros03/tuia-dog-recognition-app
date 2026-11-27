"""Arquitecturas y configuracion de entrenamiento de la Etapa 2.

Las clases viven en un modulo importable (y no en la notebook) porque los checkpoints .pth se
guardan como modelo completo: torch.load necesita importar la clase con el mismo modulo y nombre.
"""
