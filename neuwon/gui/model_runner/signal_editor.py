from ..control_panels import *
from .embedded_plot import MatplotlibEmbed
from neuwon import TimeSeries

options_grid = [[
    'Square Wave',
    'Sine Wave',
    'Triangle Wave',
    'Sawtooth Wave',
    'Constant Wave',
], [
    '', # 'Load From File',
    '', # 'Random Noise',
    '',
    '',
    '',
]]

class SignalGenerator(ManagementPanel):
    def __init__(self, parent):
        self.active = {}
        super().__init__(parent, 'Signal',
                         panel=('CustomSettingsPanel', ('signal_type',)))
        # 
        self.add_button_delete(callback=self.delete)
        self.add_button_rename(callback=self.rename)
        self.selector.add_button('Start All', self.start_all, row=2)
        self.selector.add_button('Stop All',  self.stop_all,  row=2)
        self.selector.add_button('Start',     self.start,     row=3, require_selection=True)
        self.selector.add_button('Stop',      self.stop,      row=3, require_selection=True)
        # 
        self._init_settings_panel()
        # 
        self.embed = MatplotlibEmbed(self.get_widget())
        self.embed.frame.grid(row=0, rowspan=2, column=4)
        self.add_callback(self._update_plot)

    def create(self, segment):
        name = self.ask_new_name()
        self.parameters[name] = {'segment': segment, 'signal_type': 'Sine Wave'}
        self.selector.insert(name)

    def delete(self, name):
        self.stop(name)

    def rename(self, old_name, new_name):
        if old_name in self.active:
            self.active[new_name] = self.active.pop(old_name)

    def start_all(self, name=None):
        for name in self.get_parameters():
            self.start(name)

    def stop_all(self, name=None):
        for name in self.get_parameters():
            self.stop(name)

    def start(self, name):
        if name in self.active:
            return
        parameters = self.get_parameters()[name]
        component  = parameters['component']
        mode       = parameters['assign_method']
        if mode == 'add':
            mode = '+='
        elif mode == 'overwrite':
            mode = '='
        else:
            raise NotImplementedError(mode)
        timeseries = self.export_timeseries(parameters)
        timeseries.play(parameters['segment'], component, mode=mode, loop=parameters['loop_forever'])
        self.active[name] = timeseries

    def stop(self, name):
        if name not in self.active:
            return
        timeseries = self.active.pop(name)
        timeseries.stop()

    def _update_plot(self):
        parameters = self.controlled.get_parameters()
        timeseries = self.export_timeseries(parameters)
        self.embed.update(timeseries)

    def _update_signal_type(self):
        parameters = self.controlled.get_parameters()
        parameters['signal_type'] = parameters['sig_btn']
        self.controlled.set_parameters(parameters)

    def _init_control_settings(self, settings_panel):
        components = [
                'voltage',
                'current',
                # TODO: all of the species concentrations, and their delta's.
        ]
        # TODO: How feasible would it be for the GUI to determine the correct assign_method?
        settings_panel.add_dropdown('component', lambda: components)
        settings_panel.add_radio_buttons('assign_method', ['add', 'overwrite'], default='add', title='')

        row1 = [
                'Square Wave',
                'Sine Wave',
                'Triangle Wave',
                'Sawtooth Wave',
                'Constant Wave',
        ]
        buttons = settings_panel.add_radio_buttons('sig_btn', row1,
                title='')
        for btn in buttons:
            btn.configure(command=self._update_signal_type)

    def _init_common_settings(self, settings_panel):
        settings_panel.add_checkbox('loop_forever', default=True)
        settings_panel.add_entry('delay', valid_range=(0, max_float))

    def _init_min_max_period(self, settings_panel, default=(0, 1)):
        self._init_common_settings(settings_panel)
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
        settings_panel = self.controlled.add_settings_panel('Square Wave')
        self._init_control_settings(settings_panel)
        self._init_min_max_period(settings_panel)
        settings_panel.add_slider('duty_cycle', (0, 100), default=50, units='%')

        settings_panel = self.controlled.add_settings_panel('Sine Wave')
        self._init_control_settings(settings_panel)
        self._init_min_max_period(settings_panel, default=(1, -1))

        settings_panel = self.controlled.add_settings_panel('Triangle Wave')
        self._init_control_settings(settings_panel)
        self._init_min_max_period(settings_panel)

        settings_panel = self.controlled.add_settings_panel('Sawtooth Wave')
        self._init_control_settings(settings_panel)
        self._init_min_max_period(settings_panel)

        settings_panel = self.controlled.add_settings_panel('Constant Wave')
        self._init_control_settings(settings_panel)
        settings_panel.add_entry('value',
                valid_range = (-max_float, max_float))
        settings_panel.add_entry('period',
                title       = 'duration',
                default     = 10,
                valid_range = (greater_than_zero, max_float),
                units       = 'ms')

    def export_timeseries(self, signal_parameters):
        signal_type = signal_parameters['signal_type']
        signal      = TimeSeries()
        if signal_type == 'None':
            return signal
        elif signal_type == 'Constant Wave':
            signal.constant_wave(
                    value       = signal_parameters['value'],
                    duration    = signal_parameters['period'])
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
        elif signal_type == 'Load From File':
            1/0 # TODO
        else:
            raise NotImplementedError(signal_type)
        delay = signal_parameters['delay']
        if delay > 0:
            delay  = TimeSeries().constant_wave(signal.get_data()[0], delay)
            signal = delay.concatenate(signal)
        return signal
