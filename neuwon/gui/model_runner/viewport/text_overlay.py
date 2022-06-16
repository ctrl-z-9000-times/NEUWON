
class TextOverlay:
    """ Holds the state of the standard text overlay. """
    def __init__(self):
        self.show_text()
        self.show_time()
        self.show_type()

    def show_text(self, text=''):
        self._text = str(text)

    def show_time(self, time=None):
        self._time = time

    def show_type(self, value=True):
        self._show_type = bool(value)

    def _get(self, segment=None):
        return ""
        overlay = self._text
        if self._time is not None:
            overlay += f'\nTime: {self._time}'
        if self._show_type:
            if segment is None:
                neuron_type  = 'None'
                segment_type = 'None'
            else:
                neuron_type  = segment.neuron.neuron_type
                segment_type = segment.segment_type
            overlay += f'\nNeuron Type: {neuron_type}'
            overlay += f'\nSegment Type: {segment_type}'
        return overlay.strip()
