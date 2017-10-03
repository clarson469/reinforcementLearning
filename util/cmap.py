from matplotlib.colors import LinearSegmentedColormap

import settings

class CMapError(Exception):
    pass


def colormap(size):
    hex_to_rgb = lambda hexstring: [int(hexstring[i*2:i*2+2], 16) / 255 for i in range(3)]

    if len(settings.colors) > size:
        raise CMapError('Can\'t make colormap onto less values than there are colors. Define fewer colors in "settings.py"')

    return LinearSegmentedColormap.from_list(
        'customCMap',
        [hex_to_rgb(c) for c in settings.colors],
        N=size
    )
