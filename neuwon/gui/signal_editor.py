from .control_panels import *
from .embedded_plot import MatplotlibEmbed
from neuwon import TimeSeries

# TODO: Add a virtual DB-component to the options for injecting electric current.
#       The program will automatically convert from amps into delta-voltage.

class SignalEditor(ManagementPanel):
    def __init__(self, parent):
        options_grid = [
            'Square Wave',
            'Sine Wave',
            'Triangle Wave',
            'Sawtooth Wave',
            'Constant Wave',
            # 'Load From File',
            # 'Random Noise',
        ]
        super().__init__(parent, 'Signal',
                         panel=('CustomSettingsPanel', ('signal_type',)))
        # 
        self.add_button_create(radio_options={'signal_type': options_grid})
        self.add_button_delete()
        self.add_button_rename()
        # 
        self._init_settings_panel()
        # 
        self.embed = MatplotlibEmbed(self.get_widget())
        self.embed.frame.grid(row=0, rowspan=2, column=3)
        self.controlled.add_callback(self._update)

    def _init_play_settings(self, settings_panel):
        settings_panel.add_dropdown('component', lambda: ['TODO'])
        settings_panel.add_radio_buttons('assign_method', ['add', 'overwrite'], default='add', title='')
        settings_panel.add_checkbox('loop_forever', default=True)

    def _init_min_max_period(self, settings_panel):
        settings_panel.add_empty_space()
        settings_panel.add_entry('minimum',
                valid_range = (-max_float, max_float),
                default     = 0,)
        settings_panel.add_entry('maximum',
                valid_range = (-max_float, max_float),
                default     = 1,)
        settings_panel.add_entry('period',
                valid_range = (greater_than_zero, max_float),
                default     = 10,
                units       = 'ms')

    def _init_settings_panel(self):
        waveform_name = 'Square Wave'
        settings_panel = self.controlled.add_settings_panel(waveform_name)
        self._init_play_settings(settings_panel)
        settings_panel.add_section(waveform_name + ' Settings')
        self._init_min_max_period(settings_panel)
        settings_panel.add_slider('duty_cycle', (0, 100), default=50, units='%')

        waveform_name = 'Sine Wave'
        settings_panel = self.controlled.add_settings_panel(waveform_name)
        self._init_play_settings(settings_panel)
        settings_panel.add_section(waveform_name + ' Settings')
        self._init_min_max_period(settings_panel)

        waveform_name = 'Triangle Wave'
        settings_panel = self.controlled.add_settings_panel(waveform_name)
        self._init_play_settings(settings_panel)
        settings_panel.add_section(waveform_name + ' Settings')
        self._init_min_max_period(settings_panel)

        waveform_name = 'Sawtooth Wave'
        settings_panel = self.controlled.add_settings_panel(waveform_name)
        self._init_play_settings(settings_panel)
        settings_panel.add_section(waveform_name + ' Settings')
        self._init_min_max_period(settings_panel)

        waveform_name  = 'Constant Wave'
        settings_panel = self.controlled.add_settings_panel(waveform_name)
        self._init_play_settings(settings_panel)
        settings_panel.add_section(waveform_name + ' Settings')
        settings_panel.add_entry('value',
                valid_range = (-max_float, max_float))
        settings_panel.add_entry('duration',
                default     = 10,
                valid_range = (greater_than_zero, max_float),
                units       = 'ms')

    def _update(self):
        parameters  = self.controlled.get_parameters()
        timeseries  = self.export_timeseries(parameters)
        self.embed.update(timeseries)

    def export(self):
        data = {}
        for name, signal_parameters in self.get_parameters():
            data[name] = self.export_timeseries(signal_parameters)
        return data

    def export_timeseries(self, signal_parameters):
        signal_type = signal_parameters['signal_type']
        if signal_type == 'Constant Wave':
            return TimeSeries().constant_wave(
                    value       = signal_parameters['value'],
                    duration    = signal_parameters['duration'])
        elif signal_type == 'Square Wave':
            return TimeSeries().square_wave(
                    minimum     = signal_parameters['minimum'],
                    maximum     = signal_parameters['maximum'],
                    period      = signal_parameters['period'],
                    duty_cycle  = signal_parameters['duty_cycle'] / 100)
        elif signal_type == 'Sine Wave':
            return TimeSeries().sine_wave(
                    minimum     = signal_parameters['minimum'],
                    maximum     = signal_parameters['maximum'],
                    period      = signal_parameters['period'])
        elif signal_type == 'Triangle Wave':
            return TimeSeries().triangle_wave(
                    minimum     = signal_parameters['minimum'],
                    maximum     = signal_parameters['maximum'],
                    period      = signal_parameters['period'])
        elif signal_type == 'Sawtooth Wave':
            return TimeSeries().sawtooth_wave(
                    minimum     = signal_parameters['minimum'],
                    maximum     = signal_parameters['maximum'],
                    period      = signal_parameters['period'])
        else:
            raise NotImplementedError(signal_type)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('SignalEditor Test')
    SignalEditor(root).get_widget().grid()
    root.mainloop()
