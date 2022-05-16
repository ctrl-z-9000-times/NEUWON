import matplotlib.pyplot as plt
import matplotlib.colors

class Coloration:
    """ Holds each segments color & visibility data, manages colormaps. """
    def __init__(self):
        self.segment_values     = None
        self.visible_segments   = None
        self._color_data        = None
        self.set_colormap(self.get_all_colormaps()[0])
        self.set_background_color('black')

    @classmethod
    def get_all_colormaps(cls):
        return plt.colormaps()

    def set_colormap(self, colormap):
        self.colormap = plt.get_cmap(colormap)
        self._color_data = None

    def set_segment_values(self, segment_values):
        self.segment_values = segment_values
        self._color_data = None

    def set_visible_segments(self, visible_segments):
        self.visible_segments = visible_segments

    def _get(self):
        if self._color_data is None:
            self._color_data = self.colormap(self.segment_values)
        return self._color_data

    def set_background_color(self, color):
        self.background_color = matplotlib.colors.to_rgb(color)
