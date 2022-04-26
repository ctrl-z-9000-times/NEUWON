import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog, font
import bisect

padx = 5
pady = 1
pad_top = 10

class SettingsPanel:
    """ GUI element for editing a table of parameters. """
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.row_idx = 0
        self.variables = []

    def get_parameters(self):
        return {str(v): v.get() for v in self.variables}

    def set_parameters(self, parameters):
        for k,v in parameters.items():
            self.frame.setvar(k, v)

    def add_empty_space(self, size=pad_top):
        self.frame.rowconfigure(self.row_idx, minsize=size)
        self.row_idx += 1

    def add_radio_buttons(self, text, options, variable):
        self.variables.append(variable)
        label   = ttk.Label(self.frame, text=text)
        btn_row = tk.Frame(self.frame)
        for column, x in enumerate(options):
            if isinstance(variable, tk.StringVar):
                value = x
            else:
                value = column
            button = ttk.Radiobutton(btn_row, text=x, variable=variable, value=value)
            button.grid(row=0, column=column)
        label  .grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        btn_row.grid(row=self.row_idx, column=1, sticky='w', padx=padx, pady=pady,
                columnspan=2) # No units so allow expansion into the units column.
        self.row_idx += 1

    def add_checkbox(self, text, variable):
        self.variables.append(variable)
        label  = ttk.Label(self.frame, text=text)
        button = ttk.Checkbutton(self.frame, variable=variable,)
        label .grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        button.grid(row=self.row_idx, column=1, sticky='w', padx=padx, pady=pady)
        self.row_idx += 1

    def add_slider(self, text, variable, from_, to, units=""):
        self.variables.append(variable)
        label = ttk.Label(self.frame, text=text)
        value = ttk.Label(self.frame)
        def value_changed_callback(v):
            v = float(v)
            v = round(v, 3)
            v = str(v) + " " + units
            value.configure(text=v.ljust(5))
        scale = ttk.Scale(self.frame, variable=variable,
                from_=from_, to=to,
                command = value_changed_callback,
                orient = 'horizontal',)
        value_changed_callback(scale.get())
        # 
        label.grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        scale.grid(row=self.row_idx, column=1, sticky='ew',pady=pady)
        value.grid(row=self.row_idx, column=2, sticky='w', padx=padx, pady=pady)
        self.row_idx += 1
        return scale

    def add_entry(self, text, variable, units=""):
        self.variables.append(variable)
        label = ttk.Label(self.frame, text=text)
        entry = ttk.Entry(self.frame, textvar = variable, justify='right')
        units = ttk.Label(self.frame, text=units, justify='left')
        # 
        if isinstance(variable, tk.BooleanVar):
            validate_type = bool
        elif isinstance(variable, tk.IntVar):
            validate_type = int
        elif isinstance(variable, tk.DoubleVar):
            validate_type = float
        else:
            validate_type = lambda x: x
        value = None
        def focus_in(event):
            nonlocal value
            value = variable.get()
        def focus_out(event):
            entry.selection_clear()
            text = entry.get()
            try:
                v = validate_type(text)
            except ValueError:
                variable.set(value)
            else:
                entry.delete(0, tk.END)
                entry.insert(0, str(v))
        entry.bind('<FocusIn>', focus_in)
        entry.bind('<FocusOut>', focus_out)
        # 
        label.grid(row=self.row_idx, column=0, sticky='w', padx=padx, pady=pady)
        entry.grid(row=self.row_idx, column=1, sticky='w', pady=pady)
        units.grid(row=self.row_idx, column=2, sticky='w', padx=padx, pady=pady)
        self.row_idx += 1
        return entry

class SelectorPanel:
    """ GUI element for managing lists. """
    def __init__(self, root, on_select_callback):
        self.frame = ttk.Frame(root)
        self._on_select_callback = on_select_callback
        self._current_selection = None
        # The button_panel is a row of buttons.
        self.button_panel = ttk.Frame(self.frame)
        self.button_panel.grid(row=0, column=0, sticky='nesw')
        self.column_idx = 0 # Index for appending buttons.
        # 
        self.listbox = tk.Listbox(self.frame, selectmode='single', exportselection=True)
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
        self._on_select_callback(self._current_selection, item)
        self._current_selection = item

    def touch(self):
        """ Issue an event as though the user just selected the current item. """
        self._on_select_callback(self._current_selection, self._current_selection)

    def add_button(self, text, command, require_selection=False):
        if require_selection:
            def callback():
                item = self.get()
                if item is None:
                    return
                command(item)
        else:
            def callback():
                item = self.get()
                command(item)
        button = tk.Button(self.button_panel, text=text, command=callback, font=font.BOLD,)
        button.grid(row=1, column=self.column_idx, sticky='w', padx=padx, pady=pady)
        self.column_idx += 1

    def set(self, items):
        """ Replace the current contents of this Listbox with the given list of items. """
        self.listbox.delete(0, tk.END)
        self.listbox.insert(0, *items)
        self._on_select(None, deselect=True)

    def get(self):
        return self._current_selection

    def get_all(self):
        return self.listbox.get(0, tk.END)

    def _select_idx(self, idx):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)

    def select(self, item):
        idx = self.get_all().index(item)
        self._select_idx(idx)
        self._on_select(None)

    def clear(self):
        """ Deselect everything. """
        self.listbox.selection_clear(0, tk.END)
        self._on_select(None, deselect=True)

    def insert_sorted(self, item):
        idx = bisect.bisect(self.get_all(), item)
        self.listbox.insert(idx, item)
        self._select_idx(idx)
        self._on_select(None)

    def rename(self, old_item, new_item):
        idx = self.get_all().index(old_item)
        self.listbox.delete(idx)
        self.listbox.insert(idx, new_item)
        self._select_idx(idx)
        self._on_select(None)

    def delete(self, item):
        idx = self.get_all().index(item)
        self.listbox.delete(idx)
        self._on_select(None, deselect=True)

class ManagementPanel:
    def __init__(self, root, title, on_select_callback):
        self.title = str(title)
        self._on_select_callback = on_select_callback
        self.selector = SelectorPanel(root, self._on_select)
        self.settings = SettingsPanel(self.selector.frame)
        self.settings.frame.grid(row=1, column=1, sticky='nesw')
        self.frame = self.selector.frame

        label = ttk.Label(self.frame,
                textvariable=tk.StringVar(self.frame, name="__title"),
                font=font.BOLD,
                relief='raised',
                padding=padx)
        label.grid(row=0, column=1, sticky='w', padx=padx, pady=pady)

    def _on_select(self, old_item, new_item):
        self.frame.setvar("__title", f"{self.title}: {new_item}")
        self._on_select_callback(old_item, new_item)
