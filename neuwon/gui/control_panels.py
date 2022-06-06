""" General purpose GUI elements for making complex settings menus. """

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from collections.abc import Callable, Iterable, Mapping
import sys
import bisect
import decimal

__all__ = (
        'np',
        'tk', 'ttk',
        'padx', 'pady', 'pad_top',
        'max_float', 'max_int', 'inf',
        'greater_than_zero', 'less_than_one',
        'Toplevel',
        'Panel',
        'SettingsPanel',
        'CustomSettingsPanel',
        'ListSelector',
        'ItemSelector',
        'ManagementPanel',
        'OrganizerPanel',)

padx = 5
pady = 2
pad_top = 10

# Common values for valid_ranges.
inf               = np.inf
max_float         = sys.float_info.max
max_int           = sys.maxsize
greater_than_zero = np.nextafter(0, 1)
less_than_one     = np.nextafter(1, 0)

def Toplevel(title:str):
        window = tk.Toplevel()
        window.title(title)
        window.grid_rowconfigure(   0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        frame = ttk.Frame(window)
        frame.grid(sticky='nesw')
        return window, frame

class Panel:
    def get_widget(self):
        """
        Returns the outermost frame of this GUI element,
        which should be mapped into the parent frame with the ".grid()" method.
        """
        return self.frame
    def get_parameters(self) -> dict:
        raise NotImplementedError(type(self))
    def set_parameters(self, parameters:dict):
        raise NotImplementedError(type(self))
    def add_callback(self, function):
        raise NotImplementedError(type(self))

class SettingsPanel(Panel):
    """ GUI element for editing a table of parameters. """

    # Layout Notes:
    # Widgets are arranged in 3 columns:
    #       Column 1) Description
    #       Column 2) Data entry
    #       Column 3) Units / Misc.
    # 
    # And multiple settings are stacked vertically. If the vertical stack gets
    # too large then it is automatically split into multiple stacks.

    def __init__(self, parent, override_mode=False):
        self.frame       = ttk.Frame(parent)
        self._row_idx    = 0  # Index for appending widgets.
        self._col_idx    = 0  # Index for appending widgets.
        self._parameters = {} # Preserves extra parameters that aren't used by this panel.
        self._variables  = {} # Store the anonymous tkinter variable objects.
        self._defaults   = {}
        self._callbacks  = []
        self._defer_callbacks = False
        self._override_mode = bool(override_mode)
        if self._override_mode:
            self._changed = set()
            self._set_changed_state = {} # Function for each variable.
            self._init_changed_style()

    def _init_changed_style(self):
        color = 'yellow'
        s = ttk.Style()
        s.configure('Changed.TRadiobutton', background=color, highlightcolor=color)
        s.map(      'Changed.TRadiobutton', background=[('active', color)],)
        s.configure('Changed.TCheckbutton', background=color)
        s.map(      'Changed.TCheckbutton', background=[('active', color)],)
        s.map(      'Changed.TCombobox',    fieldbackground=[('readonly', color)],)
        s.configure('Changed.Horizontal.TScale', troughcolor=color)
        s.configure('Changed.TEntry', fieldbackground=color)

    def get_parameters(self):
        assert not self._defer_callbacks
        if self._override_mode:
            for name, variable in self._variables.items():
                if name in self._changed:
                    self._parameters[name] = variable.get()
                else:
                    try:
                        self._parameters.pop(name)
                    except KeyError:
                        pass
        else:
            for name, variable in self._variables.items():
                self._parameters[name] = variable.get()
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters
        self._defer_callbacks = True
        # Set the widget variables so that they display the new values.
        for name, variable in self._variables.items():
            try:
                value = self._parameters[name]
            except KeyError:
                value = self._defaults[name]
                if self._override_mode:
                    self._set_changed_state[name](False)
            else:
                if self._override_mode:
                    self._set_changed_state[name](True)
            variable.set(value)
        self._defer_callbacks = False
        self._call_callbacks()

    def set_defaults(self, parameters):
        self._defaults = parameters
        self._defer_callbacks = True
        # Update the widget variables for the non-overridden parameters to display the new default values.
        if self._override_mode:
            for name, variable in self._variables.items():
                if name not in self._changed:
                    variable.set(self._defaults[name])
        self._defer_callbacks = False
        self._call_callbacks()

    def add_callback(self, function):
        self._callbacks.append(function)

    def _call_callbacks(self, *args):
        if self._defer_callbacks:
            return
        for f in self._callbacks:
            f()

    def _incr_row_idx(self):
        self._row_idx += 1
        if self._row_idx >= 20:
            # If there are too many rows of settings in the settings panel then they
            # go off of the bottom of the screen and they become inaccessible.
            # In that case start a new column of settings.
            self.add_column()

    def add_column(self):
        """ Start a new column for settings, adjacent to the current one. """
        self._row_idx = 0
        self._col_idx += 3
        # Cosmetic: Separate the columns with a vertical bar so that
        # they look visually distinct.
        bar = ttk.Separator(self.frame, orient='vertical')
        bar.grid(row=0, rowspan=999, column=self._col_idx,
                 sticky='ns', padx=padx, pady=pady)
        self._col_idx += 1

    def add_empty_space(self, size=pad_top):
        self.frame.grid_rowconfigure(self._row_idx, minsize=size)
        self._incr_row_idx()

    def add_section(self, title:str):
        """ Cosmetic, add a label and dividing line over a group of settings. """
        if self._row_idx > 0:
            bar = ttk.Separator(self.frame, orient='horizontal')
            bar.grid(row=self._row_idx, column=self._col_idx, columnspan=3,
                     sticky='ew', padx=padx, pady=pady)
            self.frame.grid_rowconfigure(self._row_idx, minsize=pad_top)
            self._row_idx += 1
        label = ttk.Label(self.frame, text=title)
        label.grid(row=self._row_idx, column=self._col_idx, columnspan=3,
                   sticky='w', padx=padx, pady=pady)
        self._incr_row_idx()

    def add_radio_buttons(self, parameter_name, options, variable=None, *,
                          title=None, default=None, callback=None):
        # Clean and save the arguments.
        assert parameter_name not in self._variables
        if variable is None: variable = tk.StringVar()
        self._variables[parameter_name] = variable
        if title is None: title = parameter_name.replace('_', ' ').title()
        self._defaults[parameter_name] = default if default is not None else variable.get()
        variable.set(self._defaults[parameter_name])
        # Create the widgets.
        label   = ttk.Label(self.frame, text=title)
        btn_row = ttk.Frame(self.frame)
        buttons = []
        for idx, x in enumerate(options):
            if isinstance(variable, tk.StringVar):
                value = x
            else:
                value = idx
            button = ttk.Radiobutton(btn_row, text=x, variable=variable, value=value)
            buttons.append(button)
        variable.trace_add('write', self._call_callbacks)
        if callback is not None:
            old_value = self._defaults[parameter_name]
            def callback_wrapper(*args):
                nonlocal old_value
                new_value = variable.get()
                if new_value != old_value:
                    callback(new_value)
                    old_value = new_value
            variable.trace_add('write', callback_wrapper)
            callback(old_value)
        # Arrange the widgets.
        for column, button in enumerate(buttons):
            button.grid(row=0, column=column, pady=pady)
        label  .grid(row=self._row_idx, column=self._col_idx+0, sticky='w', padx=padx, pady=pady)
        btn_row.grid(row=self._row_idx, column=self._col_idx+1, sticky='w', padx=padx, pady=pady,
                columnspan=2) # No units so allow expansion into the units column.
        self._incr_row_idx()
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(parameter_name)
                    for button in buttons:
                        button.configure(style='Changed.TRadiobutton')
                else:
                    variable.set(self._defaults[parameter_name])
                    self._changed.discard(parameter_name)
                    for button in buttons:
                        button.configure(style='TRadiobutton')
            self._set_changed_state[parameter_name] = set_changed_state
            def on_select():
                if (parameter_name not in self._changed) and (variable.get() == self._defaults[parameter_name]):
                    return
                set_changed_state(True)
            for button in buttons:
                button.configure(command=on_select)
                button.bind('<BackSpace>', lambda event: set_changed_state(False))
                button.bind('<Delete>',    lambda event: set_changed_state(False))
        return buttons

    def add_dropdown(self, parameter_name, options_callback, variable=None, *,
                     title=None, default=None, callback=None):
        # Clean and save the arguments.
        if not isinstance(options_callback, Callable) and isinstance(options_callback, Iterable):
            options_list     = list(options_callback)
            options_callback = lambda: options_list
        assert parameter_name not in self._variables
        if variable is None: variable = tk.StringVar()
        self._variables[parameter_name] = variable
        if title is None: title = parameter_name.replace('_', ' ').title()
        if default is None: default = variable.get()
        if default:
            has_default = True
        else:
            has_default = False
            default = 'nothing selected'
        self._defaults[parameter_name] = default
        variable.set(self._defaults[parameter_name])
        # Create the widgets.
        def postcommand():
            options = options_callback()
            options = [str(x) for x in options]
            menu.configure(values=options)
        label = ttk.Label(self.frame, text=title)
        menu  = ttk.Combobox(self.frame, textvar=variable, postcommand=postcommand, width=12)
        menu.configure(state='readonly')
        menu.bind('<<ComboboxSelected>>', lambda event: menu.selection_clear())
        variable.trace_add('write', self._call_callbacks)
        if callback is not None:
            if has_default:
                old_value = variable.get()
            else:
                old_value = None
            def callback_wrapper(*args):
                nonlocal old_value
                new_value = variable.get()
                if new_value != old_value:
                    callback(new_value)
                    old_value = new_value
            variable.trace_add('write', callback_wrapper)
            callback(old_value)
        # Arrange the widgets.
        label.grid(row=self._row_idx, column=self._col_idx+0, sticky='w', padx=padx, pady=pady)
        menu .grid(row=self._row_idx, column=self._col_idx+1, sticky='ew',           pady=pady)
        self._incr_row_idx()
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(parameter_name)
                    menu.configure(style='Changed.TCombobox')
                else:
                    variable.set(self._defaults[parameter_name])
                    self._changed.discard(parameter_name)
                    menu.configure(style='TCombobox')
            self._set_changed_state[parameter_name] = set_changed_state
            def on_select():
                menu.selection_clear()
                if (parameter_name not in self._changed) and (variable.get() == self._defaults[parameter_name]):
                    return
                set_changed_state(True)
            menu.bind('<<ComboboxSelected>>', lambda event: on_select())
            menu.bind('<BackSpace>', lambda event: set_changed_state(False))
            menu.bind('<Delete>',    lambda event: set_changed_state(False))
        return menu

    def add_checkbox(self, parameter_name, variable=None, *,
                     title=None, default=None, callback=None):
        # Clean and save the arguments.
        assert parameter_name not in self._variables
        if variable is None: variable = tk.BooleanVar()
        self._variables[parameter_name] = variable
        if title is None: title = parameter_name.replace('_', ' ').title()
        self._defaults[parameter_name] = default if default is not None else variable.get()
        variable.set(self._defaults[parameter_name])
        # Create the widgets.
        label  = ttk.Label(self.frame, text=title)
        button = ttk.Checkbutton(self.frame, variable=variable,)
        variable.trace_add('write', self._call_callbacks)
        if callback is not None:
            old_value = variable.get()
            def callback_wrapper(*args):
                nonlocal old_value
                new_value = variable.get()
                if new_value != old_value:
                    callback(new_value)
                    old_value = new_value
            variable.trace_add('write', callback_wrapper)
            callback(old_value)
        # Arrange the widgets.
        label .grid(row=self._row_idx, column=self._col_idx+0, sticky='w', padx=padx, pady=pady)
        button.grid(row=self._row_idx, column=self._col_idx+1, sticky='w', padx=padx, pady=pady)
        self._incr_row_idx()
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(parameter_name)
                    button.configure(style='Changed.TCheckbutton')
                else:
                    variable.set(self._defaults[parameter_name])
                    self._changed.discard(parameter_name)
                    button.configure(style='TCheckbutton')
            self._set_changed_state[parameter_name] = set_changed_state
            button.configure(command = lambda:       set_changed_state(True))
            button.bind('<BackSpace>', lambda event: set_changed_state(False))
            button.bind('<Delete>',    lambda event: set_changed_state(False))
        return button

    def add_slider(self, parameter_name, valid_range, variable=None, *, title=None, default=None, units=''):
        # Clean and save the arguments.
        assert parameter_name not in self._variables
        if variable is None: variable = tk.DoubleVar()
        self._variables[parameter_name] = variable
        if title is None: title = parameter_name.replace('_', ' ').title()
        self._defaults[parameter_name] = default if default is not None else variable.get()
        from_, to = valid_range
        # ttk does not support changing the resolution (up/down increments).
        # Reimplement it by creating a new variable, keeping it in sync with
        # the users variable, and applying a scale conversion between them.
        divisions  = 30
        resolution = (to - from_) / divisions
        rescaled = type(variable)()
        bridge_active = False # This flag prevents infinite recursion.
        def bridge_to_tkinter(*args):
            nonlocal bridge_active
            if bridge_active: return
            app_value = variable.get()
            bridge_active = True
            rescaled.set(app_value / resolution)
            bridge_active = False
        def bridge_to_application(*args):
            nonlocal bridge_active
            if bridge_active: return
            ttk_value = rescaled.get()
            bridge_active = True
            variable.set(ttk_value * resolution)
            bridge_active = False
        variable.trace_add('write', bridge_to_tkinter)
        rescaled.trace_add('write', bridge_to_application)
        # Create the widgets.
        label = ttk.Label(self.frame, text=title)
        scale = ttk.Scale(self.frame, variable=rescaled,
                from_   = from_ / resolution,
                to      = to    / resolution,
                orient  = 'horizontal')
        value = ttk.Label(self.frame)
        n_digits = 3 # Try to show this number of digits in the label.
        round_to = max(0, n_digits - len(str(round(to))))
        def update_value_label(*args):
            v = round(variable.get(), round_to)
            if round_to == 0:
                v = int(v)
            value.configure(text=(str(v) + ' ' + units))
        variable.trace_add('write', update_value_label)
        # Set the initial value and force tkinter to call all of the bound events.
        variable.set(self._defaults[parameter_name])
        variable.trace_add('write', self._call_callbacks)
        # Arrange the widgets.
        label.grid(row=self._row_idx, column=self._col_idx+0, sticky='w', padx=padx, pady=pady)
        scale.grid(row=self._row_idx, column=self._col_idx+1, sticky='ew',           pady=pady)
        value.grid(row=self._row_idx, column=self._col_idx+2, sticky='w', padx=padx, pady=pady)
        self._incr_row_idx()
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(parameter_name)
                    scale.configure(style='Changed.Horizontal.TScale')
                else:
                    variable.set(self._defaults[parameter_name])
                    self._changed.discard(parameter_name)
                    scale.configure(style='Horizontal.TScale')
            self._set_changed_state[parameter_name] = set_changed_state
            scale.configure(command = lambda v:     set_changed_state(True))
            scale.bind('<BackSpace>', lambda event: set_changed_state(False))
            scale.bind('<Delete>',    lambda event: set_changed_state(False))
            # By default mouse-1 doesn't focus on the slider, which is needed for the backspace binding.
            scale.bind('<Button-1>', lambda event: scale.focus_set())
        return scale

    def add_entry(self, parameter_name, variable=None, *, title=None, valid_range=(None, None), default=None, units=''):
        # Clean and save the arguments.
        assert parameter_name not in self._variables
        if variable is None: variable = tk.DoubleVar()
        self._variables[parameter_name] = variable
        if title is None: title = parameter_name.replace('_', ' ').title()
        # Create the widgets.
        label = ttk.Label(self.frame, text=title)
        entry = ttk.Entry(self.frame, textvar=variable, justify='right', width=12)
        units = ttk.Label(self.frame, text=units)
        # Arrange the widgets.
        label.grid(row=self._row_idx, column=self._col_idx+0, sticky='w', padx=padx, pady=pady)
        entry.grid(row=self._row_idx, column=self._col_idx+1, sticky='ew',           pady=pady)
        units.grid(row=self._row_idx, column=self._col_idx+2, sticky='w', padx=padx, pady=pady)
        self._incr_row_idx()
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(parameter_name)
                    entry.configure(style='Changed.TEntry')
                else:
                    variable.set(self._defaults[parameter_name])
                    self._changed.discard(parameter_name)
                    entry.configure(style='TEntry')
            self._set_changed_state[parameter_name] = set_changed_state
        # Custom input validation.
        if isinstance(variable, tk.BooleanVar):
            validate_type = bool
        elif isinstance(variable, tk.IntVar):
            validate_type = int
        elif isinstance(variable, tk.DoubleVar):
            validate_type = float
        else:
            validate_type = lambda x: x
        minimum, maximum = valid_range
        def validate_range(x):
            if minimum is not None:
                if x < minimum:
                    raise ValueError()
            if maximum is not None:
                if x > maximum:
                    raise ValueError()
        def clean_input(old_value, new_value):
            try:
                vv = validate_type(new_value)
                validate_range(vv)
            except ValueError:
                vv = old_value
                entry.bell()
            # Cosmetic fix: no negative zeros.
            if isinstance(variable, tk.DoubleVar):
                if vv == 0: vv = abs(vv)
            return vv
        # Perform the input validation when the user moves keyboard focus
        # into/out of the entry box.
        initial_value = None # Save the value from before the user edits it.
        def focus_in(event=None):
            nonlocal initial_value
            initial_value = variable.get()
        def focus_out(event=None):
            entry.selection_clear()
            text = entry.get().strip()
            if self._override_mode and not text:
                # If the user deleted the entry's text then reset to unchanged state.
                set_changed_state(False)
            else:
                vv = clean_input(initial_value, text)
                variable.set(vv)
                if self._override_mode and initial_value != vv:
                    set_changed_state(True)
            self._call_callbacks()
        entry.bind('<FocusIn>',  focus_in)
        entry.bind('<FocusOut>', focus_out)
        entry.bind('<Return>',   focus_out)
        # Up/Down Arrow key controls.
        def arrow_key_control(direction, control_key):
            focus_out()
            if entry.instate(('readonly',)):
                return
            value = vv = variable.get()
            if isinstance(variable, tk.BooleanVar):
                vv = not vv
            elif isinstance(variable, tk.IntVar):
                if not control_key:
                    delta = 1  * direction
                else:
                    delta = 10 * direction
            elif isinstance(variable, tk.DoubleVar):
                if not control_key:
                    delta = .1 * direction
                else:
                    delta = 10 * direction
            # Use controlled accuracy arithmetic to avoid introducing floating
            # point messiness like: "1.1 + .1 = 1.2000000000000002"
            quanta  = decimal.Decimal(10) ** -14
            vv      = decimal.Decimal(vv)   .quantize(quanta, decimal.ROUND_HALF_EVEN)
            delta   = decimal.Decimal(delta).quantize(quanta, decimal.ROUND_HALF_EVEN)
            vv = float(vv + delta)
            vv = clean_input(value, vv)
            variable.set(vv)
            if self._override_mode and value != vv:
                set_changed_state(True)
            self._call_callbacks()
        entry.bind('<Up>',           lambda event: arrow_key_control(+1, False))
        entry.bind('<Down>',         lambda event: arrow_key_control(-1, False))
        entry.bind('<Control-Up>',   lambda event: arrow_key_control(+1, True))
        entry.bind('<Control-Down>', lambda event: arrow_key_control(-1, True))
        # Set the default/initial value and also convert it to the correct datatype.
        default_value = validate_type(default if default is not None else variable.get())
        self._defaults[parameter_name] = default_value
        variable.set(default_value)
        return entry

class CustomSettingsPanel(Panel):
    """ GUI element to show different SettingsPanels depending on the current parameters. """
    def __init__(self, parent, key_parameter:str):
        self.frame    = ttk.Frame(parent)
        self._key     = str(key_parameter)
        self._options = {}
        self._current = None

    def add_panel(self, name:str, panel):
        name = str(name)
        assert name not in self._options
        self._options[name] = panel

    def get_panel(self, name:str):
        return self._options[str(name)]

    def add_settings_panel(self, name:str, **kwargs) -> SettingsPanel:
        """
        Convenience method to create a new SettingsPanel and add it to this
        custom SettingsPanel switcher.
        """
        settings_panel = SettingsPanel(self.frame, **kwargs)
        self.add_panel(name, settings_panel)
        return settings_panel

    def get_parameters(self) -> dict:
        if self._current is not None:
            return self._current.get_parameters()
        else:
            return {}

    def set_parameters(self, parameters:dict):
        if self._current is not None:
            self._current.get_widget().grid_remove()
            self._current = None
        if not parameters:
            return # Leave the panel blank.
        option = parameters[self._key]
        if not option:
            return # Nothing selected, leave the panel blank.
        self._current = self._options[option]
        self._current.get_widget().grid()
        self._current.set_parameters(parameters)

    def set_defaults(self, parameters:dict):
        option = parameters[self._key]
        panel = self._options[option]
        panel.set_defaults(parameters)

    def add_callback(self, function):
        for panel in self._options.values():
            panel.add_callback(function)

class AutoScrollbar(ttk.Scrollbar):
    """
    A scrollbar that hides itself if it's not needed.
    Only works with the grid geometry manager!
    """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            super().set(lo, hi)
    def pack(self, **kw):
        raise NotImplementedError("cannot use pack with this widget")
    def place(self, **kw):
        raise NotImplementedError("cannot use place with this widget")

class ListSelector(Panel):
    """ GUI element for selecting multiple items from a list of options. """
    def __init__(self, parent, options, default=False):
        self.frame    = ttk.Frame(parent)
        button_frame  = ttk.Frame(self.frame)
        select_all    = ttk.Button(button_frame, text='Select All', command=self.select_all)
        deselect_all  = ttk.Button(button_frame, text='Deselect All', command=self.deselect_all)
        listbox_frame = ttk.Frame(self.frame)
        self.listbox  = tk.Listbox(listbox_frame, selectmode='multiple', exportselection=False)
        scrollbar     = AutoScrollbar(listbox_frame)
        button_frame .grid(row=0, column=0, sticky='nesw')
        listbox_frame.grid(row=1, column=0, sticky='nesw')
        select_all   .grid(row=0, column=0, sticky='nesw')
        deselect_all .grid(row=0, column=1, sticky='nesw')
        self.listbox .grid(row=0, column=0, sticky='nesw')
        scrollbar    .grid(row=0, column=1, sticky='ns')
        self.frame   .grid_rowconfigure(1, weight=1)
        listbox_frame.grid_rowconfigure(0, weight=1)
        self.frame   .grid_columnconfigure(0, weight=1)
        listbox_frame.grid_columnconfigure(0, weight=1)
        button_frame .grid_columnconfigure(0, weight=1)
        button_frame .grid_columnconfigure(1, weight=1)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar   .configure(command=self.listbox.yview)
        self.set_parameters({x: default for x in options})
        self._callbacks = []
        self.listbox.bind('<<ListboxSelect>>', self._call_callbacks)

    def get_parameters(self) -> dict:
        return {option: self.listbox.selection_includes(idx)
                for idx, option in enumerate(self.listbox.get(0, tk.END))}

    def set_parameters(self, parameters:dict):
        self.listbox.delete(0, tk.END)
        self.listbox.insert(0, *parameters.keys())
        for idx, selected in enumerate(parameters.values()):
            if selected:
                self.listbox.selection_set(idx)

    def add_callback(self, function):
        self._callbacks.append(function)

    def _call_callbacks(self, *args):
        for f in self._callbacks:
            f()

    def select_all(self):
        self.listbox.selection_set(0, tk.END)
        self._call_callbacks()

    def deselect_all(self):
        self.listbox.selection_clear(0, tk.END)
        self._call_callbacks()

class ItemSelector:
    """
    GUI element for managing lists of user created items. Allows for selecting a
    single item at a time, and tracks the transitions between selected items.
    """
    def __init__(self, parent, on_select_callback, keep_sorted=True):
        self.frame = ttk.Frame(parent)
        self._on_select_callback = on_select_callback
        self._current_selection  = None
        self._keep_sorted        = bool(keep_sorted)
        # The add buttons in a row along the top of the panel.
        self._button_panel = ttk.Frame(self.frame)
        self._button_panel.grid(row=0, column=0, sticky='ew')
        self._buttons_requiring_selection = []
        self._column_idx = [0, 0, 0, 0, 0, 0] # Indexes for appending buttons.
        # 
        listbox_frame = ttk.Frame(self.frame)
        listbox_frame.grid(row=2, column=0, sticky='nesw')
        self.listbox = tk.Listbox(listbox_frame, selectmode='browse', exportselection=False)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        self.listbox.grid(row=0, column=0, sticky='nesw')
        scrollbar = AutoScrollbar(listbox_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar   .configure(command=self.listbox.yview)
        # Resize vertically.
        self.frame   .grid_rowconfigure(2, weight=1)
        listbox_frame.grid_rowconfigure(0, weight=1)
        # Resize whichever widget is smaller horizonally to fill space
        # (buttons or listbox), but do not resize the outer frame.
        listbox_frame.grid_columnconfigure(0, weight=1)

    def _on_select(self, event=None):
        indices = self.listbox.curselection()
        if indices:
            item = self.listbox.get(indices[0])
        else:
            item = None
        if item == self._current_selection:
            return
        if item is None:
            for button in self._buttons_requiring_selection:
                button.configure(state='disabled')
        elif self._current_selection is None:
            for button in self._buttons_requiring_selection:
                button.configure(state='normal')
        old_item = self._current_selection
        self._current_selection = item
        self._on_select_callback(old_item, item)

    def touch(self):
        """ Issue an event as though the user just selected the current item. """
        self._on_select_callback(self._current_selection, self._current_selection)

    def add_button(self, text, command, require_selection=False, row=0):
        button = ttk.Button(self._button_panel, text=text, command=lambda: command(self._current_selection))
        button.grid(row=row, column=self._column_idx[row], sticky='ew', pady=pady)
        self._button_panel.grid_columnconfigure(self._column_idx[row], weight=1) # Stretch buttons horizontally.
        self._column_idx[row] += 1
        if require_selection:
            self._buttons_requiring_selection.append(button)
            if self.get() is None:
                button.configure(state='disabled')
        return button

    def set_list(self, items):
        """ Replace the current contents of this Listbox with the given list of items. """
        if self._keep_sorted: items = sorted(items)
        self.listbox.delete(0, tk.END)
        self.listbox.insert(0, *items)
        self._on_select()

    def get(self):
        return self._current_selection

    def get_list(self):
        return list(self.listbox.get(0, tk.END))

    def _select_idx(self, idx):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)

    def select(self, item):
        idx = self.get_list().index(item)
        self._select_idx(idx)
        self._on_select()

    def clear_selection(self):
        self.listbox.selection_clear(0, tk.END)
        self._on_select()

    def insert(self, item, idx=None):
        if idx is None:
            if self._keep_sorted:
                idx = bisect.bisect(self.get_list(), item)
            else:
                idx = len(self.get_list())
        else:
            idx = round(idx)
            assert not self._keep_sorted
        self.listbox.insert(idx, item)
        self._select_idx(idx)
        self._on_select()

    def rename(self, old_item, new_item):
        idx = self.get_list().index(old_item)
        self.listbox.delete(idx)
        if self._keep_sorted:
            self.insert(new_item)
        else:
            self.insert(new_item, idx)

    def delete(self, item):
        idx = self.get_list().index(item)
        self.listbox.delete(idx)
        self._on_select()

    def move(self, item, direction):
        assert not self._keep_sorted
        items_list  = self.get_list()
        old_idx     = items_list.index(item)
        new_idx     = min(len(items_list) - 1, max(0, old_idx + direction))
        self.listbox.delete(old_idx)
        self.listbox.insert(new_idx, item)
        self._select_idx(new_idx)
        self._on_select()

class ManagementPanel(Panel):
    """ GUI element to use a ItemSelector to control another panel. """
    def __init__(self, parent, title:str, *,
                keep_sorted:bool=True,
                inline_panel=False,
                panel='SettingsPanel',):
        self.title          = str(title).title()
        self.parameters     = {}
        self.selector       = ItemSelector(parent, self._on_select, keep_sorted)
        self.frame          = self.selector.frame
        self.frame.grid_columnconfigure(2, minsize=padx) # Cosmetic spacing between the two halves of the panel.
        self._init_controlled_panel(panel, inline_panel)

    def _init_controlled_panel(self, arguments, inline_panel):
        # Gather the panel_type and arguments.
        if isinstance(arguments, str) or isinstance(arguments, type):
            panel_type = arguments
            args       = ()
            kwargs     = {}
        else:
            arguments = tuple(arguments)
            if len(arguments) == 1:
                panel_type = str(arguments[0])
                args       = ()
                kwargs     = {}
            elif len(arguments) == 2:
                panel_type, args = arguments
                if isinstance(args, dict):
                    kwargs = args
                    args   = ()
                else:
                    kwargs = {}
            elif len(arguments) == 3:
                panel_type, args, kwargs = arguments
            else:
                raise TypeError(arguments)
        # 
        builtin_panel_types = {
            'SettingsPanel':        SettingsPanel,
            'CustomSettingsPanel':  CustomSettingsPanel,
            'OrganizerPanel':       OrganizerPanel,
            'ManagementPanel':      ManagementPanel,
        }
        panel_type = builtin_panel_types.get(panel_type, panel_type)
        # 
        self.panel = self.controlled = panel_type(self.frame, *args, **kwargs)
        if inline_panel:
            self.controlled.get_widget().grid(row=1, column=0, sticky='nesw', pady=pady)
        else:
            self.controlled.get_widget().grid(row=0, rowspan=3, column=1, sticky='nesw', padx=padx, pady=pady)
            self.frame.grid_columnconfigure(1, weight=1)

    def _on_select(self, old_item, new_item):
        # Save the current parameters out of the SettingsPanel.
        if old_item is not None:
            self.parameters[old_item] = self.controlled.get_parameters()
        # Load the newly selected parameters into the SettingsPanel.
        if new_item is not None:
            self.controlled.set_parameters(self.parameters[new_item])
        else:
            self.controlled.set_parameters({})

    def get_parameters(self) -> dict:
        # Save the currently selected item out of the SettingsPanel.
        item = self.selector.get()
        if item is not None:
            self.parameters[item] = self.controlled.get_parameters()
        # Remake the parameters as an ordered-dict using the current ordering of
        # the items in the selector panel.
        ordered_dict = {}
        for item in self.selector.get_list():
            ordered_dict[item] = self.parameters.pop(item)
        ordered_dict.update(self.parameters)
        self.parameters = ordered_dict
        return self.parameters

    def set_parameters(self, parameters:dict):
        # Force the currently selected item out of the settings panel and into
        # the old parameters dict before assigning a new parameters dict.
        self.selector.clear_selection()
        self.parameters = parameters
        self.selector.set_list(self.parameters.keys())

    def add_callback(self, function):
        self.controlled.add_callback(function)

    def ask_new_name(self, options_grid=None):
        title  = f'Create {self.title}'
        prompt = f'Enter new {self.title.lower()} name:'
        response = _askstring(title, prompt, self._default_new_name(),
                                options_grid=options_grid, parent=self.frame)
        if options_grid is None:
            try:
                return self._clean_new_name(response)
            except ValueError:
                return
        else:
            try:
                return (self._clean_new_name(response[0]), response[1])
            except ValueError:
                return (None, None)

    def _default_new_name(self):
        number = len(self.selector.get_list()) + 1
        return f'{self.title}_{number}'

    def _clean_new_name(self, name, old_name=None):
        """ Either returns the cleaned name or raises a ValueError. """
        if not name:
            raise ValueError()
        # If user renames something to the same old name, then fail to validate
        # but also don't show the error message or ring the bell.
        if name == old_name:
            raise ValueError()
        if name in self.parameters:
            self.frame.bell()
            messagebox.showerror(f'{self.title} Name Error',
                    f'{self.title} "{name}" is already defined!')
            raise ValueError()
        return name

    def add_button_create(self, radio_options:dict=None, callback=None, row=0):
        """
        Argument radio_options is a dict containing a single key-value pair.
                The value is a grid of options for the user to select from.
                The key is the parameter_name that their selection is assigned to.
        """
        title  = f'Create {self.title}'
        prompt = f'Enter new {self.title.lower()} name:'
        if radio_options is not None:
            key, options = radio_options.popitem()
            assert not radio_options
            # Arrange the options in a 2D array.
            options = np.array(options, dtype=str)
            if options.ndim < 2:
                options = options.reshape(1, -1)
            options = options.T
            def _callback(name):
                name, choice = _askstring(title, prompt, self._default_new_name(),
                                          options_grid=options, parent=self.frame)
                try:
                    name = self._clean_new_name(name)
                except ValueError:
                    return
                self.parameters[name] = {key: choice}
                self.selector.insert(name)
                if callback is not None: callback(name)
        else:
            def _callback(name):
                name = _askstring(title, prompt, self._default_new_name(),
                                  parent=self.frame)
                try:
                    name = self._clean_new_name(name)
                except ValueError:
                    return
                if self.selector.get() is None:
                    self.parameters[name] = self.controlled.get_parameters()
                else:
                    self.parameters[name] = {}
                self.selector.insert(name)
                if callback is not None: callback(name)
        button = self.selector.add_button('New', _callback, row=row)

    def add_button_delete(self, text:str='Delete', callback=None, require_confirmation=True, row=0):
        text = text.title()
        def _callback(name):
            if require_confirmation:
                confirmation = messagebox.askyesno(f'Confirm {text} {self.title}',
                        f'Are you sure you want to {text.lower()} {self.title.lower()} "{name}"?',
                        parent=self.frame)
                if not confirmation:
                    return
            if callback is not None: callback(name)
            self.selector.delete(name)
            self.parameters.pop(name)
        button = self.selector.add_button(text, _callback, require_selection=True, row=row)
        self.selector.listbox.bind('<Delete>',    lambda event: button.invoke())
        self.selector.listbox.bind('<BackSpace>', lambda event: button.invoke())

    def add_button_rename(self, callback=None, row=0):
        def _callback(name):
            new_name = _askstring(f'Rename {self.title}',
                    f'Rename {self.title.lower()} "{name}" to:',
                    name, parent=self.frame)
            try:
                new_name = self._clean_new_name(new_name, name)
            except ValueError:
                return
            self.parameters[new_name] = self.parameters[name]
            self.selector.rename(name, new_name)
            self.parameters.pop(name)
            if callback is not None: callback(name, new_name)
        self.selector.add_button('Rename', _callback, require_selection=True, row=row)

    def add_button_duplicate(self, callback=None, row=0):
        def _callback(name):
            new_name = _askstring(f'Duplicate {self.title}',
                    f'Enter new {self.title.lower()} name:',
                    parent=self.frame)
            try:
                new_name = self._clean_new_name(new_name)
            except ValueError:
                return
            self.parameters[new_name] = dict(self.parameters[name]) # Should this be a deep-copy?
            self.selector.insert(new_name)
            if callback is not None: callback(name, new_name)
        self.selector.add_button('Duplicate', _callback, require_selection=True, row=row)

    def add_buttons_up_down(self, row=0):
        up   = lambda name: self.selector.move(name, -1)
        down = lambda name: self.selector.move(name, +1)
        self.selector.add_button('Move Up',   up,   require_selection=True, row=row)
        self.selector.add_button('Move Down', down, require_selection=True, row=row)

def _askstring(title, prompt, default='', options_grid=None, *, parent):
        # 
        response = tk.StringVar(value=default)
        def ok_callback(event=None):
            if not response.get().strip():
                entry.bell()
                entry.focus_set()
            else:
                window.destroy()
        def cancel_callback(event=None):
            response.set('')
            window.destroy()
        # Make the widgets.
        window, frame = Toplevel(title)
        label  = ttk.Label(frame, text=prompt)
        entry  = ttk.Entry(frame, textvar=response)
        radio  = ttk.Frame(frame)
        ok     = ttk.Button(frame, text='Ok',     command=ok_callback,)
        cancel = ttk.Button(frame, text='Cancel', command=cancel_callback,)
        # Arrange the widgets.
        label.grid(row=0, columnspan=2, padx=padx, pady=pady)
        entry.grid(row=1, columnspan=2, padx=padx, pady=pady, sticky='ew')
        radio.grid(row=2, columnspan=2, padx=padx, pady=pady)
        ok    .grid(row=3, column=0, padx=2*padx, pady=pady)
        cancel.grid(row=3, column=1, padx=2*padx, pady=pady)
        # 
        if options_grid is not None:
            choice = tk.StringVar(value=options_grid[0][0])
            for row_idx, row_data in enumerate(options_grid):
                for col_idx, value in enumerate(row_data):
                    if not value:
                        continue
                    button = ttk.Radiobutton(radio, text=value, variable=choice, value=value)
                    button.grid(row=row_idx, column=col_idx, sticky='w', padx=padx, pady=pady)
        # 
        entry .bind('<Return>', ok_callback)
        window.bind('<Escape>', cancel_callback)
        window.wait_visibility()
        entry.focus_set()
        entry.select_range(0, tk.END)
        # Make the dialog window modal. This prevents user interaction with
        # any other application window until this dialog is resolved.
        window.grab_set()
        window.transient(parent)
        window.wait_window(window)
        # 
        if options_grid is not None:
            return (response.get().strip(), choice.get())
        else:
            return response.get().strip()

class OrganizerPanel(Panel):
    def __init__(self, parent, tabs=True):
        self.tabs = bool(tabs)
        if self.tabs:
            self._notebook = ttk.Notebook(parent)
        else:
            self._notebook = ttk.Frame(parent)
        self._parameters = {}
        self._tabs       = {}

    def get_widget(self):
        return self._notebook

    def add_tab(self, title:str, panel):
        """ This also saves the panel as an attribute on this object. """
        assert title not in dir(self), 'Duplicate panel or name conflict!'
        setattr(self, title, panel)
        self._tabs[title] = panel
        title = title.replace('_', ' ').title()
        if self.tabs:
            self._notebook.add(panel.get_widget(), text=title,
                                sticky='nesw', padding=(padx, pad_top))
        else:
            column = 2 * len(self._tabs)
            if column > 2:
                div = ttk.Separator(self._notebook, orient='vertical')
                div.grid(row=0, rowspan=2, column=(column - 1),
                         sticky='ns', padx=padx, pady=pad_top)
            label = ttk.Label(self._notebook, text=title)
            label.grid(row=0, column=column, padx=padx, pady=pad_top)
            panel.get_widget().grid(row=1, column=column,
                                    sticky='nesw', padx=padx, pady=pady)

    def current_tab(self):
        assert self.tabs
        index   = self._notebook.index('current')
        panels  = self._notebook.tabs()
        current = panels[index]
        for title, panel in self._tabs.items():
            if str(panel.get_widget()) == current:
                return title
        raise RuntimeError

    def get_parameters(self):
        for title, panel in self._tabs.items():
            self._parameters[title] = panel.get_parameters()
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters
        for title, panel in self._tabs.items():
            panel.set_parameters(parameters.get(title, {}))
