from .control_panels import *
from .embedded_plot import MatplotlibEmbed
from neuwon import TimeSeries

class SignalEditor(ManagementPanel):
    def __init__(self, frame):
        super().__init__(frame, "Signal", init_settings_panel=False)
        self.set_settings_panel(CustomSettingsPanel(self.get_widget(), "signal_type"))

        options_grid = [
            "Square Wave",
            "Sine Wave",
            "Triangle Wave",
            "Sawtooth Wave",
            "Constant Wave",
            # "Load From File",
            # "Random Noise",
        ]
        self.add_button_create(radio_options={"signal_type": options_grid})
        self.add_button_delete()
        self.add_button_rename()

        self._setup_waveform_settings()

        self.embed = MatplotlibEmbed(self.get_widget())
        self.embed.frame.grid(row=0, rowspan=2, column=3)

        self.settings.add_callback(self._update)

    def _setup_waveform_settings(self):
        waveform_name = "Square Wave"
        settings_panel = SettingsPanel(self.settings.get_widget())
        self.settings.add_panel(waveform_name, settings_panel)
        settings_panel.add_section(waveform_name)
        self._setup_common(settings_panel)
        settings_panel.add_entry("minimum", default=0)
        settings_panel.add_entry("maximum", default=1)
        settings_panel.add_entry("period",  default=10, units='ms')
        settings_panel.add_slider("duty_cycle", (0, 100), default=50, units='%')

        waveform_name = "Sine Wave"
        settings_panel = SettingsPanel(self.settings.get_widget())
        self.settings.add_panel(waveform_name, settings_panel)
        settings_panel.add_section(waveform_name)
        self._setup_common(settings_panel)
        settings_panel.add_entry("minimum", default=0)
        settings_panel.add_entry("maximum", default=1)
        settings_panel.add_entry("period",  default=10, units='ms')

        waveform_name = "Triangle Wave"
        settings_panel = SettingsPanel(self.settings.get_widget())
        self.settings.add_panel(waveform_name, settings_panel)
        settings_panel.add_section(waveform_name)
        self._setup_common(settings_panel)
        settings_panel.add_entry("minimum", default=0)
        settings_panel.add_entry("maximum", default=1)
        settings_panel.add_entry("period",  default=10, units='ms')

        waveform_name = "Sawtooth Wave"
        settings_panel = SettingsPanel(self.settings.get_widget())
        self.settings.add_panel(waveform_name, settings_panel)
        settings_panel.add_section(waveform_name)
        self._setup_common(settings_panel)
        settings_panel.add_entry("minimum", default=0)
        settings_panel.add_entry("maximum", default=1)
        settings_panel.add_entry("period",  default=10, units='ms')

        waveform_name  = "Constant Wave"
        settings_panel = SettingsPanel(self.settings.get_widget())
        self.settings.add_panel(waveform_name, settings_panel)
        settings_panel.add_section(waveform_name)
        self._setup_common(settings_panel)
        settings_panel.add_entry("value")
        settings_panel.add_entry("duration", default=10, units='ms')

    def _setup_common(self, settings_panel):
        settings_panel.add_dropdown("component", lambda: ['TODO'])
        settings_panel.add_radio_buttons("assign_method", ["add", "overwrite"], default="add", title="")
        settings_panel.add_checkbox("loop_forever", default=True)

    def _update(self):
        parameters  = self.settings.get_parameters()
        timeseries  = self.export_timeseries(parameters)
        self.embed.update(timeseries)

    def export(self):
        data = {}
        for name, signal_parameters in self.get_parameters():
            data[name] = self.export_timeseries(signal_parameters)
        return data

    def export_timeseries(self, signal_parameters):
        signal_type = signal_parameters["signal_type"]
        if signal_type == "Constant Wave":
            return TimeSeries().constant_wave(
                    value       = signal_parameters["value"],
                    duration    = signal_parameters["duration"])
        elif signal_type == "Square Wave":
            return TimeSeries().square_wave(
                    minimum     = signal_parameters["minimum"],
                    maximum     = signal_parameters["maximum"],
                    period      = signal_parameters["period"],
                    duty_cycle  = signal_parameters["duty_cycle"] / 100)
        elif signal_type == "Sine Wave":
            return TimeSeries().sine_wave(
                    minimum     = signal_parameters["minimum"],
                    maximum     = signal_parameters["maximum"],
                    period      = signal_parameters["period"])
        elif signal_type == "Triangle Wave":
            return TimeSeries().triangle_wave(
                    minimum     = signal_parameters["minimum"],
                    maximum     = signal_parameters["maximum"],
                    period      = signal_parameters["period"])
        elif signal_type == "Sawtooth Wave":
            return TimeSeries().sawtooth_wave(
                    minimum     = signal_parameters["minimum"],
                    maximum     = signal_parameters["maximum"],
                    period      = signal_parameters["period"])
        else:
            raise NotImplementedError(signal_type)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("SignalEditor Test")
    SignalEditor(root).get_widget().grid()
    root.mainloop()
