"""Translate equivalent precision names for Lightning 1.x and 2.x."""


def compatible_precision(precision, lightning_version):
    if int(lightning_version.split('.')[0]) < 2:
        return {
            '16-mixed': 16,
            'bf16-mixed': 'bf16',
            '32-true': 32,
            '64-true': 64,
        }.get(precision, precision)
    return precision
