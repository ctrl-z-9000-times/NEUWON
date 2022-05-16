from ..control_panels import *
from .embedded_plot import MatplotlibEmbed
from neuwon import TimeSeries

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
        self.embed.frame.grid(row=0, rowspan=2, column=4)
        self.add_callback(self._update)

    def _init_play_settings(self, settings_panel):
        components = [
                'voltage',
                'current',
                # TODO: all of the species concentrations, and their delta's.
        ]
        # TODO: How feasible would it be for the GUI to determine the correct assign_method?
        settings_panel.add_dropdown('component', lambda: components)
        settings_panel.add_radio_buttons('assign_method', ['add', 'overwrite'], default='add', title='')
        settings_panel.add_checkbox('loop_forever', default=True)
        settings_panel.add_entry('delay', valid_range=(0, max_float))

    def _init_min_max_period(self, settings_panel, default=(0, 1)):
        settings_panel.add_empty_space()
        settings_panel.add_entry('extremum_1',
                valid_range = (-max_float, max_float),
                default     = default[0],)
        settings_panel.add_entry('extremum_2',
                valid_range = (-max_float, max_float),
                default     = default[1],)
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
        self._init_min_max_period(settings_panel, default=(1, -1))

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
        signal      = TimeSeries()
        if signal_type == 'Constant Wave':
            signal.constant_wave(
                    value       = signal_parameters['value'],
                    duration    = signal_parameters['duration'])
        elif signal_type == 'Square Wave':
            signal.square_wave(
                    extremum_1  = signal_parameters['extremum_1'],
                    extremum_2  = signal_parameters['extremum_2'],
                    period      = signal_parameters['period'],
                    duty_cycle  = signal_parameters['duty_cycle'] / 100)
        elif signal_type == 'Sine Wave':
            signal.sine_wave(
                    extremum_1  = signal_parameters['extremum_1'],
                    extremum_2  = signal_parameters['extremum_2'],
                    period      = signal_parameters['period'])
        elif signal_type == 'Triangle Wave':
            signal.triangle_wave(
                    extremum_1  = signal_parameters['extremum_1'],
                    extremum_2  = signal_parameters['extremum_2'],
                    period      = signal_parameters['period'])
        elif signal_type == 'Sawtooth Wave':
            signal.sawtooth_wave(
                    extremum_1  = signal_parameters['extremum_1'],
                    extremum_2  = signal_parameters['extremum_2'],
                    period      = signal_parameters['period'])
        else:
            raise NotImplementedError(signal_type)
        delay = signal_parameters['delay']
        if delay > 0:
            delay  = TimeSeries().constant_wave(signal.get_data()[0], delay)
            signal = delay.concatenate(signal)
        return signal
