from ..control_panels import *
from neuwon.rxd.nmodl.parser import NmodlParser
from neuwon.rxd.mechanisms import import_python_mechanism
from tkinter import filedialog
import os.path

class MechanismManager(ManagementPanel):
    def __init__(self, root):
        super().__init__(root, 'Mechanism', panel=('CustomSettingsPanel', ('filename',)))
        # 
        self.selector.add_button('Import', self.import_mechanisms)
        self.add_button_delete('Remove')
        self.selector.add_button('Info', self.info_on_mechanism, require_selection=True, row=1)

    def import_mechanisms(self, selected):
        files = filedialog.askopenfilenames(
                title='Import Mechanisms',
                filetypes=[
                    ('any', '*'),
                    ('NMODL', '.mod'),
                    ('Python', '.py')])
        for abspath in files:
            name, ext = os.path.splitext(os.path.basename(abspath))
            if name in self.parameters:
                continue
            self.parameters[name] = {'filename': abspath}
            self._make_settings_panel(abspath)
            self.selector.insert(name)

    def set_parameters(self, parameters):
        for mech_name, mech_parameters in parameters.items():
            filename = mech_parameters['filename']
            try:
                self.controlled.get_panel(filename)
            except KeyError:
                self._make_settings_panel(filename)
        super().set_parameters(parameters)

    def _make_settings_panel(self, filename):
            if filename.endswith('.mod'):
                self._make_nmodl_settings_panel(filename)
            elif filename.endswith('.py'):
                self._make_py_settings_panel(filename)

    def _make_nmodl_settings_panel(self, filename):
        settings_panel = self.controlled.add_settings_panel(filename, override_mode=True)
        parser = NmodlParser(filename, preprocess=False)
        for name, (value, units) in parser.gather_parameters().items():
            settings_panel.add_entry(name, title=name, default=value, units=units)

    def _make_py_settings_panel(self, filename):
        settings_panel = self.controlled.add_settings_panel(filename, override_mode=True)
        mechanism = import_python_mechanism(filename)
        if mechanism is None:
            return
        for name, dtype in mechanism.get_parameters().items():
            settings_panel.add_entry(name)

    def info_on_mechanism(self, selected):
        window, frame = Toplevel(selected + ' Documentation')
        # Display filename in a raised box.
        filename = self.parameters[selected]['filename']
        file_var = tk.StringVar(value=filename)
        file_var.trace_add('write', lambda *args: file_var.set(filename))
        file = ttk.Entry(frame, textvariable=file_var)
        file.grid(row=0, column=0, padx=padx, pady=pad_top, sticky='ew')
        # Show the NMODL file.
        with open(filename, 'rt') as f:
            source_code = f.read()
        docs = tk.Text(frame)
        docs.insert('1.0', source_code)
        docs.configure(state='disabled')
        docs.grid(row=1, column=0, padx=padx, pady=pady, sticky='nsw')

    @classmethod
    def export(cls, parameters):
        sim = {}
        for name, gui in parameters.items():
            gui = dict(gui)
            sim[name] = (gui.pop('filename'), gui)
        return sim

class MechanismSelector(ManagementPanel):
    def __init__(self, root, mechanism_manager, override_mode=False):
        super().__init__(root, 'Mechanism', inline_panel=True,
                         panel=('SettingsPanel', {'override_mode': override_mode}))
        self.mechanisms = mechanism_manager
        # 
        self.selector.add_button('Insert', self.insert_mechanism)
        if not override_mode:
            self.add_button_delete('Remove', require_confirmation=False)
        # 
        self.controlled.add_empty_space()
        self.controlled.add_entry('magnitude', default=1.0)

    def insert_mechanism(self, selected):
        window, frame = Toplevel('Insert Mechanisms')
        mechanisms = sorted(self.mechanisms.parameters)
        listbox = tk.Listbox(frame, selectmode='extended', exportselection=True)
        listbox.grid(row=0, column=0, columnspan=2, padx=padx, pady=pad_top)
        listbox.insert(0, *mechanisms)
        selection = []
        def ok_callback():
            for idx in listbox.curselection():
                selection.append(mechanisms[idx])
            window.destroy()
        ok = ttk.Button(frame, text='Ok',     command=ok_callback,)
        no = ttk.Button(frame, text='Cancel', command=window.destroy,)
        ok.grid(row=1, column=0, padx=2*padx, pady=pad_top, sticky='ew')
        no.grid(row=1, column=1, padx=2*padx, pady=pad_top, sticky='ew')
        # 
        listbox.focus_set()
        listbox.bind('<Double-Button-1>', lambda event: ok_callback())
        window .bind('<Escape>', lambda event: window.destroy())
        # Make the dialog window modal. This prevents user interaction with
        # any other application window until this dialog is resolved.
        window.grab_set()
        window.transient(self.frame)
        window.wait_window(window)
        # 
        for x in selection:
            if x in self.parameters:
                continue
            self.parameters[x] = {}
            self.selector.insert(x)

    @classmethod
    def export(cls, parameters):
        return {k: v['magnitude'] for k, v in parameters.items()}
