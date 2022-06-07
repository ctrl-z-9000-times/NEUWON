from ..control_panels import *
from .embedded_plot import MatplotlibEmbed
from neuwon import TimeSeries

class DataRecorder(ManagementPanel):
    def __init__(self, parent):
        super().__init__(parent, "Data Recorder")

        self.add_button_duplicate()
        self.add_button_rename()
        self.add_button_delete(row=1)
        self.selector.add_button("Clear Data", self.clear_data, row=1, require_selection=True)
        self.selector.add_button("Start All", self.start_all, row=2)
        self.selector.add_button("Stop All",  self.stop_all,  row=2)
        self.selector.add_button("Start", self.start, row=3, require_selection=True)
        self.selector.add_button("Stop",  self.stop,  row=3, require_selection=True)

        # Note: This has no settings!
        #       DB component must be set immediately upon creation.

        # TODO: Label showing current start/stop state.

        # TODO: Show DB component
        # TODO: Show neuron & segment type.
        # TODO: Button to save data to file.

        self.embed = MatplotlibEmbed(self.get_widget())
        self.embed.frame.grid(row=0, rowspan=3, column=4)
        self.add_callback(self.update_plot)

    def create(self, segment):
        available_components = [
            [
                'voltage',
            ],
        ]
        name, component = self.ask_new_name(available_components)
        self.parameters[name] = {
                'segment':      segment,
                'component':    component,
                'timeseries':   TimeSeries(),}
        self.selector.insert(name)

    def clear_data(self, probe):
        self.parameters[probe]['timeseries'].clear()
        self.update_plot()

    def start_all(self, probe):
        for probe in self.parameters.keys():
            self.start(probe)

    def stop_all(self, probe):
        for probe in self.parameters.keys():
            self.stop(probe)

    def start(self, probe):
        parameters = self.parameters[probe]
        timeseries = parameters['timeseries']
        if timeseries.is_recording():
            self.frame.bell()
            return
        timeseries.record(parameters['segment'], parameters['component'])

    def stop(self, probe):
        timeseries = self.parameters[probe]['timeseries']
        if timeseries.is_stopped():
            self.frame.bell()
            return
        timeseries.stop()

    def update_plot(self):
        probe = self.selector.get()
        if probe is None:
            timeseries = TimeSeries()
        else:
            timeseries = self.parameters[probe]['timeseries']
        self.embed.update(timeseries)
