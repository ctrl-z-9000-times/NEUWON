""" General purpose GUI elements for making complex settings menus. """

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, simpledialog
import bisect

padx = 5
pady = 1
pad_top = 10

class Panel:
    def get_widget(self):
        return self.frame
    def get_parameters(self):
        raise NotImplementedError(type(self))
    def set_parameters(self, parameters):
        raise NotImplementedError(type(self))

# IDEA: If horizontal space becomes a problem then make an option for the
# SettingsPanel to compress/interleave the two columns into vertically stacked widgets.

class SettingsPanel(Panel):
    """ GUI element for editing a table of parameters. """
    def __init__(self, parent, override_mode=False):
        self.frame = ttk.Frame(parent)
        self._row_idx = 0 # Index for appending widgets.
        self._parameters = {} # Preserve extra parameters that aren't used by this panel.
        self._variables = {}
        self._defaults = {}
        self._override_mode = bool(override_mode)
        if self._override_mode:
            self._changed = set()
            self._set_changed_state = {}
            color = 'yellow'
            s = ttk.Style()
            s.configure('Changed.TRadiobutton', background=color, highlightcolor=color)
            s.map(      'Changed.TRadiobutton', background=[('active', color)],)
            s.configure('Changed.TCheckbutton', background=color)
            s.map(      'Changed.TCheckbutton', background=[('active', color)],)
            s.configure('Changed.Horizontal.TScale', troughcolor=color)
            s.configure('Changed.TEntry', fieldbackground=color)

    def get_parameters(self):
        for name, variable in self._variables.items():
            if not self._override_mode or name in self._changed:
                self._parameters[name] = variable.get()
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters
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

    def set_defaults(self, parameters):
        self._defaults = parameters
        # Update the widget variables for the non-overridden parameters to display the new default values.
        if self._override_mode:
            for name, variable in self._variables.items():
                if name not in self._changed:
                    variable.set(self._defaults[name])

    def add_empty_space(self, size=pad_top):
        self.frame.rowconfigure(self._row_idx, minsize=size)
        self._row_idx += 1

    def add_section(self, title):
        """ Cosmetic, add a label and dividing line over a group of settings. """
        bar = ttk.Separator(self.frame, orient='horizontal')
        bar.grid(row=self._row_idx, column=0, columnspan=2, sticky='ew', padx=padx, pady=pady)
        self.frame.rowconfigure(self._row_idx, minsize=pad_top)
        self._row_idx += 1
        label = ttk.Label(self.frame, text=title)
        label.grid(row=self._row_idx, column=0, sticky='w', padx=padx, pady=pady)
        self._row_idx += 1

    def add_radio_buttons(self, variable_name, variable, options, *, title=None, default=None):
        # Clean and save the arguments.
        assert variable_name not in self._variables
        self._variables[variable_name] = variable
        if title is None: title = variable_name.replace('_', ' ').title()
        self._defaults[variable_name] = default if default is not None else variable.get()
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
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(variable_name)
                    for button in buttons:
                        button.configure(style="Changed.TRadiobutton")
                else:
                    variable.set(self._defaults[variable_name])
                    self._changed.discard(variable_name)
                    for button in buttons:
                        button.configure(style="TRadiobutton")
            self._set_changed_state[variable_name] = set_changed_state
            def change():
                if (variable_name not in self._changed) and (variable.get() == self._defaults[variable_name]):
                    return
                set_changed_state(True)
            for button in buttons:
                button.configure(command=change)
                button.bind("<BackSpace>", lambda event: set_changed_state(False))
        # Arrange the widgets.
        label  .grid(row=self._row_idx, column=0, sticky='w', padx=padx, pady=pady)
        btn_row.grid(row=self._row_idx, column=1, sticky='w', padx=padx, pady=pady,
                columnspan=2) # No units so allow expansion into the units column.
        self._row_idx += 1
        for column, button in enumerate(buttons):
            button.grid(row=0, column=column)
        return buttons

    def add_checkbox(self, variable_name, variable, *, title=None, default=None):
        # Clean and save the arguments.
        assert variable_name not in self._variables
        self._variables[variable_name] = variable
        if title is None: title = variable_name.replace('_', ' ').title()
        self._defaults[variable_name] = default if default is not None else variable.get()
        # Create the widgets.
        label  = ttk.Label(self.frame, text=title)
        button = ttk.Checkbutton(self.frame, variable=variable,)
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(variable_name)
                    button.configure(style="Changed.TCheckbutton")
                else:
                    variable.set(self._defaults[variable_name])
                    self._changed.discard(variable_name)
                    button.configure(style="TCheckbutton")
            self._set_changed_state[variable_name] = set_changed_state
            button.configure(command=lambda: set_changed_state(True))
            button.bind("<BackSpace>", lambda event: set_changed_state(False))
        # Arrange the widgets.
        label .grid(row=self._row_idx, column=0, sticky='w', padx=padx, pady=pady)
        button.grid(row=self._row_idx, column=1, sticky='w', padx=padx, pady=pady)
        self._row_idx += 1
        return button

    def add_slider(self, variable_name, variable, valid_range, *, title=None, default=None, units=""):
        # Clean and save the arguments.
        assert variable_name not in self._variables
        self._variables[variable_name] = variable
        if title is None: title = variable_name.replace('_', ' ').title()
        self._defaults[variable_name] = default if default is not None else variable.get()
        from_, to = valid_range
        # Create the widgets.
        label = ttk.Label(self.frame, text=title)
        scale = ttk.Scale(self.frame, variable=variable,
                from_=from_, to=to,
                orient = 'horizontal',)
        value = ttk.Label(self.frame)
        def update_value_label(*args):
            v = round(variable.get(), 3)
            value.configure(text=(str(v) + " " + units))
        variable.trace_add("write", update_value_label)
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(variable_name)
                    scale.configure(style="Changed.Horizontal.TScale")
                else:
                    variable.set(self._defaults[variable_name])
                    self._changed.discard(variable_name)
                    scale.configure(style="Horizontal.TScale")
            self._set_changed_state[variable_name] = set_changed_state
            scale.configure(command=lambda v: set_changed_state(True))
            scale.bind("<BackSpace>", lambda event: set_changed_state(False))
            # By default mouse-1 doesn't focus on the slider, which is needed for the backspace binding.
            scale.bind("<Button-1>", lambda event: scale.focus_set())
        # Arrange the widgets.
        label.grid(row=self._row_idx, column=0, sticky='w', padx=padx, pady=pady)
        scale.grid(row=self._row_idx, column=1, sticky='ew',pady=pady)
        value.grid(row=self._row_idx, column=2, sticky='w', padx=padx, pady=pady)
        self._row_idx += 1
        return scale

    def add_entry(self, variable_name, variable, *, title=None, valid_range=(None, None), default=None, units=""):
        # Clean and save the arguments.
        assert variable_name not in self._variables
        self._variables[variable_name] = variable
        if title is None: title = variable_name.replace('_', ' ').title()
        self._defaults[variable_name] = default if default is not None else variable.get()
        # Create the widgets.
        label = ttk.Label(self.frame, text=title)
        entry = ttk.Entry(self.frame, textvar=variable, justify='right')
        units = ttk.Label(self.frame, text=units)
        # Highlight changed values.
        if self._override_mode:
            def set_changed_state(changed):
                if changed:
                    self._changed.add(variable_name)
                    entry.configure(style="Changed.TEntry")
                else:
                    variable.set(self._defaults[variable_name])
                    self._changed.discard(variable_name)
                    entry.configure(style="TEntry")
            self._set_changed_state[variable_name] = set_changed_state
        # Custom input validation.
        minimum, maximum = valid_range
        if isinstance(variable, tk.BooleanVar):
            validate_type = bool
        elif isinstance(variable, tk.IntVar):
            validate_type = int
        elif isinstance(variable, tk.DoubleVar):
            validate_type = float
        else:
            validate_type = lambda x: x
        value = None # Save the initial value from before the user edits it.
        def focus_in(event):
            nonlocal value
            value = variable.get()
        def focus_out(event):
            entry.selection_clear()
            text = entry.get()
            if self._override_mode and not text.strip():
                set_changed_state(False)
            else:
                try:
                    vv = validate_type(text)
                    if minimum is not None and vv <= minimum: raise ValueError()
                    if maximum is not None and vv >= maximum: raise ValueError()
                except ValueError:
                    vv = value
                    entry.bell()
                if vv == 0: vv = abs(vv) # Cosmetic fix: no negative zeros.
                variable.set(vv)
                if self._override_mode and vv != value:
                    set_changed_state(True)
        entry.bind('<FocusIn>', focus_in)
        entry.bind('<FocusOut>', focus_out)
        # Arrange the widgets.
        label.grid(row=self._row_idx, column=0, sticky='w', padx=padx, pady=pady)
        entry.grid(row=self._row_idx, column=1, sticky='w', pady=pady)
        units.grid(row=self._row_idx, column=2, sticky='w', padx=padx, pady=pady)
        self._row_idx += 1
        return entry

class SelectorPanel:
    """ GUI element for managing lists. """
    def __init__(self, parent, on_select_callback):
        self.frame = ttk.Frame(parent)
        self._on_select_callback = on_select_callback
        self._current_selection = None
        # The add buttons in a row along the top of the panel.
        self._button_panel = ttk.Frame(self.frame)
        self._button_panel.grid(row=0, column=0, sticky='nesw')
        self._buttons_requiring_selection = []
        self._column_idx = 0 # Index for appending buttons.
        # 
        self.listbox = tk.Listbox(self.frame, selectmode='browse', exportselection=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        self.listbox.grid(row=1, column=0, sticky='nesw')

    def _on_select(self, event, deselect=False):
        indices = self.listbox.curselection()
        if indices:
            item = self.listbox.get(indices[0])
        else:
            item = None
        if item == self._current_selection:
            return
        if item is None and not deselect:
            return
        if item is None:
            for button in self._buttons_requiring_selection:
                button.configure(state='disabled')
        elif self._current_selection is None:
            for button in self._buttons_requiring_selection:
                button.configure(state='normal')
        self._on_select_callback(self._current_selection, item)
        self._current_selection = item

    def touch(self):
        """ Issue an event as though the user just selected the current item. """
        self._on_select_callback(self._current_selection, self._current_selection)

    def add_button(self, text, command, require_selection=False):
        button = ttk.Button(self._button_panel, text=text, command=lambda: command(self._current_selection))
        button.grid(row=1, column=self._column_idx, sticky='w', pady=pady)
        self._column_idx += 1
        if require_selection:
            self._buttons_requiring_selection.append(button)
            if self.get() is None:
                button.configure(state='disabled')
        return button

    def set_list(self, items):
        """ Replace the current contents of this Listbox with the given list of items. """
        self.listbox.delete(0, tk.END)
        self.listbox.insert(0, *items)
        self._on_select(None, deselect=True)

    def get(self):
        return self._current_selection

    def get_list(self):
        return self.listbox.get(0, tk.END)

    def _select_idx(self, idx):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)

    def select(self, item):
        idx = self.get_list().index(item)
        self._select_idx(idx)
        self._on_select(None)

    def clear_selection(self):
        self.listbox.selection_clear(0, tk.END)
        self._on_select(None, deselect=True)

    def insert(self, item):
        idx = bisect.bisect(self.get_list(), item)
        self.listbox.insert(idx, item)
        self._select_idx(idx)
        self._on_select(None)

    def rename(self, old_item, new_item):
        idx = self.get_list().index(old_item)
        self.listbox.delete(idx)
        self.insert(new_item)

    def delete(self, item):
        idx = self.get_list().index(item)
        self.listbox.delete(idx)
        self._on_select(None, deselect=True)

class ManagementPanel(Panel):
    """ GUI element to use a SelectorPanel to control another panel. """
    def __init__(self, parent, title, init_settings_panel=True):
        self.title      = str(title)
        self.parameters = {}
        self.selector   = SelectorPanel(parent, self._on_select)
        self.frame      = self.selector.frame
        if init_settings_panel:
            self.set_settings_panel(SettingsPanel(self.frame))
        # Cosmetic spacing between the two halves of the panel.
        self.frame.columnconfigure(1, minsize=padx)
        # Display the title and the currently selected item.
        self._title_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self._title_var,
                  relief='raised', padding=padx, anchor='center',
        ).grid(row=0, column=2, sticky='ew', padx=padx, pady=pady)
        self._set_title(None)

    def set_settings_panel(self, panel):
        self.settings = panel
        self.settings.get_widget().grid(row=1, column=2, sticky='nesw', padx=padx, pady=pady)

    def _set_title(self, item):
        if item is None:
            item = "-nothing selected-"
        self._title_var.set(f"{self.title}: {item}")

    def _on_select(self, old_item, new_item):
        self._set_title(new_item)
        # Save the current parameters out of the SettingsPanel.
        if old_item is not None:
            self.parameters[old_item] = self.settings.get_parameters()
        # Load the newly selected parameters into the SettingsPanel.
        if new_item is not None:
            self.settings.set_parameters(self.parameters[new_item])
        else:
            self.settings.set_parameters({})

    def get_parameters(self):
        item = self.selector.get()
        if item is not None:
            self.parameters[item] = self.settings.get_parameters()
        return self.parameters

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.selector.set_list(sorted(self.parameters.keys()))

    def _duplicate_name_error(self, name):
        messagebox.showerror(f"{self.title} Name Error",
                f'{self.title} "{name}" is already defined!')

    def add_button_create(self):
        def _callback(selection):
            name = simpledialog.askstring(f"Create {self.title}", f"Enter {self.title} Name:")
            if name is None:
                return
            name = name.strip()
            if not name:
                return
            if name in self.parameters:
                self._duplicate_name_error(name)
                return
            if self.selector.get() is None:
                self.parameters[name] = self.settings.get_parameters()
            else:
                self.parameters[name] = {}
            self.selector.insert(name)
        self.selector.add_button("New", _callback)

    def add_button_delete(self, text="Delete", require_confirmation=True):
        def _callback(name):
            if require_confirmation:
                confirmation = messagebox.askyesno(f"Confirm {text} {self.title}",
                        f"Are you sure you want to {text.lower()} {self.title.lower()} '{name}'?")
                if not confirmation:
                    return
            self.selector.delete(name)
            self.parameters.pop(name)
        button = self.selector.add_button(text, _callback, require_selection=True)
        self.selector.listbox.bind("<Delete>",    lambda event: button.invoke())
        self.selector.listbox.bind("<BackSpace>", lambda event: button.invoke())

    def add_button_rename(self):
        def _callback(name):
            new_name = simpledialog.askstring(f"Rename {self.title}",
                    f'Rename {self.title} "{name}" to')
            if new_name is None:
                return
            new_name = new_name.strip()
            if not new_name:
                return
            elif new_name == name:
                return
            elif new_name in self.parameters:
                self._duplicate_name_error(new_name)
                return
            self.parameters[new_name] = self.parameters[name]
            self.selector.rename(name, new_name)
            self.parameters.pop(name)
        self.selector.add_button("Rename", _callback, require_selection=True)

    def add_button_duplicate(self):
        def _callback(name):
            new_name = simpledialog.askstring(f"Duplicate {self.title}",
                                              f"Enter {self.title} Name:")
            if new_name is None:
                return
            new_name = new_name.strip()
            if not new_name:
                return
            elif new_name == name:
                return
            elif new_name in self.parameters:
                self._duplicate_name_error(new_name)
                return
            self.parameters[new_name] = dict(self.parameters[name])
            self.selector.insert(new_name)
        self.selector.add_button("Duplicate", _callback, require_selection=True)

class OrganizerPanel(Panel):
    def __init__(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.tabs = {}

    def get_widget(self):
        return self.notebook

    def add_tab(self, title, panel):
        self.tabs[title] = panel
        self.notebook.add(panel.frame, text=title.title(),
                sticky='nesw', padding=(padx, pad_top))

    def get_parameters(self):
        return {title: panel.get_parameters() for title, panel in self.tabs.items()}

    def set_parameters(self, parameters):
        assert set(parameters).issubset(self.tabs)
        for title, panel in self.tabs.items():
            panel.set_parameters(parameters.get(title, {}))
